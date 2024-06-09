from pathlib import Path

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    ToolMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
    ToolCall,
    FunctionCall,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.model import Transformer

mistral_models_path = Path.home().joinpath("mistral_models", "7B-Instruct-v0.3")

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

completion_request = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris"),
        # AssistantMessage(
        #     content=None,
        #     tool_calls=[
        #         ToolCall(
        #             id="VvvODy9mT",
        #             function=FunctionCall(
        #                 name="get_current_weather",
        #                 arguments='{"location": "Paris, France", "format": "celsius"}',
        #             ),
        #         )
        #     ],
        # ),
        # ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22"),
        # AssistantMessage(
        #     content="The current temperature in Paris, France is 22 degrees Celsius.",
        # ),
        # UserMessage(content="What's the weather like today in San Francisco"),
        # AssistantMessage(
        #     content=None,
        #     tool_calls=[
        #         ToolCall(
        #             id="fAnpW3TEV",
        #             function=FunctionCall(
        #                 name="get_current_weather",
        #                 arguments='{"location": "San Francisco", "format": "celsius"}',
        #             ),
        #         )
        #     ],
        # ),
        # ToolMessage(tool_call_id="fAnpW3TEV", name="get_current_weather", content="20"),
    ],
)

tokens = tokenizer.encode_chat_completion(completion_request).tokens
text = tokenizer.encode_chat_completion(completion_request).text

tok = tokenizer.encode_chat_completion(completion_request)

print("tokens", tokens)
print("text", text)

out_tokens, _ = generate(
    [tokens],
    model,
    max_tokens=64,
    temperature=0.0,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
)
print("out_tokens", out_tokens)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
