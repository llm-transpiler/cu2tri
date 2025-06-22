from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def chat_completion_with_reasoning(messages):
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=messages,
        max_tokens=10000,
        # max_completion_tokens=600,
        reasoning_effort="high",
        # stream=True
    )
    return response
msg1 = [
    {"role": "user", "content": {"type": "text", "text": "What's bigger, 9.9 or 9.11?"}},
    {"role": "assistant", "content": {
            "type": "text",
            "text": "9.9 is bigger than 9.11.\n\nTo compare decimal numbers, we need to look at their value, not just count digits. \n\n9.9 = 9 + 0.9 = 9 + 9/10 = 9 + 0.90\n9.11 = 9 + 0.11 = 9 + 11/100\n\nSince 9/10 (or 0.9) equals 90/100, and 11/100 (or 0.11) equals 11/100, we can see that 90/100 > 11/100.\n\nTherefore, 9.9 > 9.11."
        }},
    {"role": "user", "content": {"type": "text", "text": "How about 9.99 or 9.11?"}}
]
msg2 = [
    {"role": "user", "content": "What's bigger, 9.9 or 9.11?"},
    {"role": "assistant", "content": "9.9 is bigger than 9.11.\n\nTo compare decimal numbers, we need to look at their value, not just count digits. \n\n9.9 = 9 + 0.9 = 9 + 9/10 = 9 + 0.90\n9.11 = 9 + 0.11 = 9 + 11/100\n\nSince 9/10 (or 0.9) equals 90/100, and 11/100 (or 0.11) equals 11/100, we can see that 90/100 > 11/100.\n\nTherefore, 9.9 > 9.11."},
    {"role": "user", "content": "How about 9.99 or 9.11?"}
]
resp = chat_completion_with_reasoning(msg2)
print(resp)
print(resp.choices[0].message.content)

# for chunk in chat_completion_with_reasoning([
#     {"role": "user", "content": "What's bigger, 9.9 or 9.11?"}
# ]):
#     if hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
#         print(f"REASONING: {chunk.choices[0].delta.reasoning}")
#     elif chunk.choices[0].delta.content:
#         print(f"CONTENT: {chunk.choices[0].delta.content}")
