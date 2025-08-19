import random
import gradio as gr

"""
This is a simple random response chatbot.
It will randomly respond with "Yes" or "No" to the user's message.
Adapted from https://www.gradio.app/docs/gradio/chatinterface
"""

def random_response(message, history):
    return random.choice(["Yes", "No"])

#demo = gr.ChatInterface(random_response, type="messages", autofocus=False)
#demo.launch()

def yes(message, history):
    return "yes"

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(placeholder="<strong>Your Random Yes-Non</strong><br>Ask Me Anything")
    chatbot.like(vote, None, None)
    gr.ChatInterface(fn=random_response, type="messages", chatbot=chatbot)


if __name__ == "__main__":
    demo.launch()