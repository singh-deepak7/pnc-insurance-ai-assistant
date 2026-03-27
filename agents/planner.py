from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

import time
from datetime import datetime

llm = Ollama(
    model="llama3.2:3b",
    num_predict=150,
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template("""
Break the user question into 2-3 smaller sub-questions.

Return ONLY a numbered list.

Question:
{question}
""")

def plan(question: str):
    now = datetime.now().astimezone()  # local timezone
    formatted_time = now.strftime("%m-%d-%y %H:%M:%S.%f %Z")[:-3]
    print(f"Planning started ... {formatted_time}")
    response = llm.invoke(prompt.format(question=question))

    # Convert to list
    steps = [
        line.strip("123456789. ").strip()
        for line in response.split("\n")
        if line.strip()
    ]

    return steps[:3]  # limit