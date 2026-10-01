import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model=os.environ["GROQ_MODEL"],
    messages=[
        {
            "role": "user",
            "content": "Write one realistic customer complaint (2 sentences) about "
            "home broadband that drops every evening. Do not use the word 'intermittent'.",
        }
    ],
    temperature=0.8,
)
print(response.choices[0].message.content)