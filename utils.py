import base64
import json
import re
import secrets
import string
import uuid

# import logging

# logging.basicConfig(level=logging.INFO)

# def generate_call_id():
#     """
#     Generate a unique call ID in a base64-like format.

#     Returns:
#         str: A unique call ID.
#     """
#     unique_id = uuid.uuid4().bytes
#     base64_id = base64.urlsafe_b64encode(unique_id).rstrip(b"=").decode("ascii")
#     return f"call_{base64_id}"

def generate_call_id():
    """
    Generate a unique call ID with exactly 9 characters in the format a-z, A-Z, 0-9.

    Returns:
        str: A unique call ID.
    """
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(secrets.choice(characters) for _ in range(9))
    return unique_id

def extract_tool_calls_from_buffer(buffer):
    """
    Extract tool calls from the given buffer.

    Args:
        buffer (list): A list of strings containing the buffer data.

    Returns:
        list: A list of formatted tool calls or None if no tool calls found.
    """
    joined_buffer = "".join(buffer)
    try:
        tool_calls_match = re.search(
            r"\[TOOL_CALLS\] (\[.*?\])", joined_buffer, re.DOTALL
        )
        if tool_calls_match:
            tool_calls = json.loads(tool_calls_match.group(1))
            formatted_tool_calls = [
                {
                    "id": generate_call_id(),
                    "function": {
                        "arguments": json.dumps(call["arguments"]),
                        "name": call["name"],
                    },
                    "type": "function",
                }
                for call in tool_calls
            ]
            return formatted_tool_calls
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

    return None
