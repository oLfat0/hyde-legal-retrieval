from openai import OpenAI

from dotenv import load_dotenv
import os
import time

load_dotenv()
TOKEN = os.getenv("LLM_API_KEY")

print("Rodando: TESTE_OPEN_ROUTER.py")
start = time.time()

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=TOKEN)
resp = client.chat.completions.create(
    model='google/gemma-3-12b-it',    
    # model='google/gemma-4-26b-a4b-it',
    # model='openai/gpt-4o-mini',    
    # model='anthropic/claude-opus-4.6-fast',    
    # model='z-ai/glm-5.1',    
    # model='google/gemma-4-31b-it',    
    # model='qwen/qwen3.6-plus',   
    # model='z-ai/glm-5v-turbo',   
    # model='openrouter/elephant-alpha',   

    messages=[{'role': 'user', 'content': 'Hi'}],
)
end = time.time()

print(resp.choices[0].message.content)
print(f"\n\nTempo de Resposta: {(end-start):.2f}s ({(end-start)/60:.2f}min)")
