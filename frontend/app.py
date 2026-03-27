import gradio as gr
import requests
import time

API_URL = "http://127.0.0.1:8000/query"


# ---------------------------
# 🚀 Ask Function (Streaming)
# ---------------------------
def ask(query, history):
    res = requests.post(API_URL, params={"q": query})
    data = res.json()

    answer = data.get("response", "")
    sources = "\n".join([f"📄 {s}" for s in data.get("sources", [])])
    trace = "\n".join(data.get("trace", []))
    confidence = str(data.get("confidence", "N/A"))

    history = history or []

    # Add user message
    history.append({"role": "user", "content": query})

    streamed = ""
    for char in answer:
        streamed += char
        time.sleep(0.01)

        yield (
            history + [{"role": "assistant", "content": streamed}],
            sources,
            trace,
            confidence
        )


# ---------------------------
# 🧹 Clear Chat
# ---------------------------
def clear_chat():
    return [], "", "", ""


# ---------------------------
# 🎨 UI Layout
# ---------------------------
with gr.Blocks(title="AI Insurance Assistant") as demo:

    gr.Markdown("# 🧠 AI-Powered P&C Insurance Assistant")
    gr.Markdown("Ask questions about policies, claims, underwriting, and reports.")

    chatbot = gr.Chatbot(height=400)

    with gr.Row():
        query = gr.Textbox(
            placeholder="Ask something like: What does homeowners insurance cover?",
            label="Your Question",
            scale=4
        )
        submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")

    with gr.Row():
        sources = gr.Textbox(label="📚 Sources", lines=5)
        confidence = gr.Textbox(label="📊 Confidence", lines=1)

    trace = gr.Textbox(label="🧠 Agent Trace", lines=8)

    # ---------------------------
    # 🔗 Actions
    # ---------------------------
    submit_btn.click(
        ask,
        inputs=[query, chatbot],
        outputs=[chatbot, sources, trace, confidence]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, sources, trace, confidence]
    )


# ---------------------------
# ▶️ Run App
# ---------------------------
if __name__ == "__main__":
    demo.launch()