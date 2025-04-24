from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import gradio as gr
blenderbot = "facebook/blenderbot-400M-distill"  
blenderbot_tokenizer = AutoTokenizer.from_pretrained(blenderbot)
blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained(blenderbot)
# Chat function with state (history) and blenderbot is used for chat
def chat_blenderbot(message, history):
    inputs = blenderbot_tokenizer.encode(message, return_tensors='pt')
    outputs = blenderbot_model.generate(inputs)
    response = blenderbot_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    # print(history)
    return "", history

# Build the Gradio Blocks UI
with gr.Blocks(title="BlenderBot Chatbot",theme=gr.themes.Glass()) as demo:
    gr.Markdown("## 🤖 BlenderBot 400M Chatbot")

    chatbot = gr.Chatbot(type='messages')
    message_input = gr.Textbox(placeholder="Type a message...", label="Your Message")
    clear_btn = gr.Button("Clear")

    state = gr.State([])

    message_input.submit(chat_blenderbot, inputs=[message_input, state], outputs=[message_input, chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot, state])

# Launch the app
demo.launch()