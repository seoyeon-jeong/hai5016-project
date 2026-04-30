import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
    base_url=os.environ["AZURE_FOUNDRY_ENDPOINT"],
)

response = client.chat.completions.create(
    model=os.environ["AZURE_FOUNDRY_MODEL"],
    messages=[
        {"role": "user", "content": "How many R's are in the word 'raspberry'?"},
    ],
)

print(response.choices[0].message.content)
