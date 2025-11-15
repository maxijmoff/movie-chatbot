import gradio as gr
import time
from chatbot_core import movie_chatbot

def respond(message, history):
    history = history or []

    try:
        start = time.time()
        answer, _ = movie_chatbot(message)
        end = time.time()
        latency = end - start
        answer_with_latency = f"{answer}\n\nWaktu respon: {latency:.2f} detik"
        history.append((message, answer_with_latency))
        return history
    except Exception as e:
        history.append(("ERROR", str(e)))
        return history


custom_css = """
body, .gradio-container { background-color: #ffffff !important; }

#chatbot {
    height: 550px !important;
    overflow: auto !important;
    background: #ffffff !important;
}

.user {
    background: #FFC4E1 !important;
    color: #000 !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    width: fit-content !important;
    max-width: 70% !important;
    margin: 6px 0 !important;
    align-self: flex-end !important;
}

.bot {
    background: #EDEDED !important;
    color: #000 !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    width: fit-content !important;
    max-width: 70% !important;
    margin: 6px 0 !important;
    align-self: flex-start !important;
}

#pink-button {
    background-color: #FFC4E1 !important;
    color: #000 !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    border: none !important;
}

#pink-button:hover {
    background-color: #FF9DCB !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("🎬 Movie Recommender Chatbot")
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False)
    msg = gr.Textbox(placeholder="Tulis pertanyaanmu...", scale=10)
    btn = gr.Button("Kirim", elem_id="pink-button")
    btn.click(respond, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(respond, inputs=[msg, chatbot], outputs=chatbot)
    btn.click(lambda: "", None, msg)
    msg.submit(lambda: "", None, msg)
demo.launch()