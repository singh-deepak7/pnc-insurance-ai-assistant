from langchain_community.llms import Ollama

import time
from datetime import datetime

llm = Ollama(
    model="llama3.2:3b",
    num_predict=150,
    temperature=0.3
)

def synthesize(original_question: str, answers: list):
    now = datetime.now().astimezone()  # local timezone
    formatted_time = now.strftime("%m-%d-%y %H:%M:%S.%f %Z")[:-3]
    print(f"Synthesize started ... {formatted_time}")
    combined = "\n".join(answers)

    prompt = f"""
    Combine the answers into a clear final response.

    Keep it concise (4-6 sentences max).

    Question:
    {original_question}

    Partial Answers:
    {combined}

    Final Answer:
    """

    return llm.invoke(prompt)