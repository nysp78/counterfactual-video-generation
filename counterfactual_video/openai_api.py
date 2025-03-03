import openai

openai_key = "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"

client = openai.OpenAI(api_key=openai_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, OpenAI!"}]
)

print(response.choices[0].message.content)
