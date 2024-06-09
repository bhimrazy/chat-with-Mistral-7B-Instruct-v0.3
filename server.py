from pathlib import Path
from threading import Thread

import litserve as ls
import torch
from litserve.specs.openai import ChatMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from utils import extract_tool_calls_from_buffer

mistral_models_path = Path.home().joinpath("mistral_models", "7B-Instruct-v0.3")


class OpenAISpecLitAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config
        )

        self.mistral_tokenizer = MistralTokenizer.from_file(
            f"{mistral_models_path}/tokenizer.model.v3"
        )

        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def decode_request(self, request: ChatCompletionRequest):
        tools = (
            [tool.dict(exclude_none=True) for tool in request.tools]
            if request.tools
            else []
        )
        messages = [message.dict(exclude_none=True) for message in request.messages]
        completion_request = ChatCompletionRequest(messages=messages, tools=tools)

        tokens = self.mistral_tokenizer.encode_chat_completion(
            completion_request
        ).tokens
        input_ids = torch.tensor([tokens]).to(self.device)

        model_inputs = {
            "input_ids": input_ids,
        }
        return model_inputs

    def predict(self, model_inputs):
        generation_kwargs = dict(
            **model_inputs,
            streamer=self.streamer,
            max_new_tokens=1000,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in self.streamer:
            # print(text, end="", flush=True)
            yield text

    def encode_response(self, output_generator) -> ChatMessage:
        buffer = []
        for output in output_generator:
            buffer.append(output)
            # check if tool calls
            if "".join(buffer).startswith("[TOOL_CALLS]"):
                tool_calls = extract_tool_calls_from_buffer(buffer)
                yield ChatMessage(role="assistant", content="", tool_calls=tool_calls)
                continue

            yield ChatMessage(role="assistant", content=output)

        # parse tool calls from output buffer
        # tool_calls = extract_tool_calls_from_buffer(buffer)
        # print(ChatMessage(role="assistant", content="", tool_calls=tool_calls))
        # yield ChatMessage(role="assistant", content=content, tool_calls=tool_calls)


if __name__ == "__main__":
    server = ls.LitServer(OpenAISpecLitAPI(), spec=ls.OpenAISpec())
    server.run(port=8000)
