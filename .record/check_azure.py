import os
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")  # = 门户里的部署名

resp = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    max_completion_tokens=512,   # SkillOpt 也用 max_completion_tokens（新参数）
    temperature=1.0,
)
print(resp.choices[0].message.content)