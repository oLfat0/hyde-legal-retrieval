from openai import OpenAI
import time

start = time.time()

API_KEY = "ROpC4lPZm44pQOPuAU4uk87DP0nV-myEjMgXRytohmY"

client = OpenAI(base_url='https://llm.liaufms.org/v1/gemma-3-12b-it', api_key=API_KEY)
resp = client.chat.completions.create(
    model='google/gemma-3-12b-it',
    messages=[{'role': 'user', 'content': 'Hi'}],
)
end = time.time()

print(resp.choices[0].message.content)
print(f"Tempo de Resposta: {(end-start):.2f}s ({(end-start)/60:.2f} min)")