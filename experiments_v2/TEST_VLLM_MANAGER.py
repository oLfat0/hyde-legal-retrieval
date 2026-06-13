import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TOKEN_VLLM = os.getenv("TOKEN_VLLM_GMA3")
print(f"Token: '{TOKEN_VLLM}'")
print(f"Tamanho: {len(TOKEN_VLLM) if TOKEN_VLLM else 'None'}")

start = time.time()

client = OpenAI(base_url='https://llm-lia.ufms.br/v1/gemma-3-12b-it', api_key=TOKEN_VLLM)
resp = client.chat.completions.create(
    model='google/gemma-3-12b-it',
    messages=[{'role': 'user', 'content': 'Hi'}],
)

end = time.time()

tempo = end-start
print(resp.choices[0].message.content)
print(f"Tempo de demora: {tempo/60:.2f}min ({tempo:.2f}s)")