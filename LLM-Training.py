import os
from groq import Groq

client = Groq()

def generate_content(prompt):
    response = client.chat.completions.create(
        messages = [{
            "role" : "user",
            "content":prompt
        }],
        model = "llama-3.1-8b-instant"
    )
    return response

output = generate_content("What will be the future oıf AI jobs in next 5 years")

print(output)

