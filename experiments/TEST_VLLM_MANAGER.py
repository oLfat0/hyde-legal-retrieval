import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_TOKEN = os.getenv("VLLM_TOKEN")

start = time.time()

client = OpenAI(base_url='https://llm.liaufms.org/v1/gemma-3-12b-it', api_key=VLLM_TOKEN)
resp = client.chat.completions.create(
    model='google/gemma-3-12b-it',
    messages=[{'role': 'user', 'content': 'Hi'}],
)

end = time.time()

tempo = end-start
print(resp.choices[0].message.content)
print(f"Tempo de demora: {tempo/60:.2f}min ({tempo:.2f}s)")