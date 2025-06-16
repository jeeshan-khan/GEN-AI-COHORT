from openai import OpenAI
from datetime import datetime
client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT="""
You are a helpful assistant.

Today's date is {datetime.now().strftime("%Y-%m-%d")}
"""
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "give me the current datetime",
        }
    ]
)

print(response.choices[0].message.content)
# import google.generativeai as genai
# import os

# genai.configure(api_key="AIzaSyCRgdqi8elWuIGBpr-ecC7lecM_0q2kwr4")

# model = genai.GenerativeModel("gemini-2.0-flash")

# chat = model.start_chat()

# response = chat.send_message("Explain to me how AI works")

# print(response.text)
