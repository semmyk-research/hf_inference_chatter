from dotenv import load_dotenv
import os
import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""

## Get Hf token
load_dotenv() # This loads variables from .env into os.environ
hf_token = os.environ.get("HF_TOKEN")


## ValueError: You must provide an api_key to work with fireworks-ai API or log in with `hf auth login`.
## noting: authentication
# using login
#huggingface-cli login
# or using an environment variable
#huggingface-cli login --token $HUGGINGFACE_TOKEN

#model = "deepseek-ai/DeepSeek-V3-0324"
#model = "Qwen/Qwen3-235B-A22B:fireworks-ai"
#model = "meta-llama/Meta-Llama-3-8B-Instruct"
model="openai/gpt-oss-120b"

#provider = "featherless-ai"
provider = "fireworks-ai"

#client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
#client = InferenceClient(provider="featherless-ai", model="models/HuggingFaceH4/zephyr-7b-beta")  ##TypeError: InferenceClient.__init__() got an unexpected keyword argument 'provider'
#client = InferenceClient(model="Qwen/Qwen3-235B-A22B:fireworks-ai")
client = InferenceClient(#model=model, 
                        provider=provider,
                        api_key=hf_token,  #os.environ["HF_TOKEN"],  #["HUGGINGACE_TOKEN"],
                        )

'''
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)
print(completion.choices[0].message)

'''

#'''
def respond(
    #model,
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    #'''
    #completion = client.chat.completions.create(
    completion = client.chat_completion(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    stream=False,  #True,
                    temperature=temperature,
                    top_p=top_p,
                    #messages=[
                    #    {
                    #        "role": "user",
                    #        "content": "What is the capital of France?"
                    #    }
                    #],
                )
    print(f"completion: {completion} \n")
    yield completion.choices[0].message.content 
    ''' #stream=True
    response = ""
    for chunks in completion.choices[0].delta.content:
        response += chunks
    yield completionc
    '''
    #'''

    '''
    response = ""
    for message in client.chat_completion(
        #model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        #token = message.choices[0].content

        response += token
        yield response
    '''
#'''


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=16384, value=512, step=1, label="Max new tokens"),  #maximum=2048, 
        gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.2, #0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    type="messages"
)


if __name__ == "__main__":
    #demo.launch(share=True)
    #demo.launch(ssr_mode=True, node_port=7890)
    demo.launch(ssr_mode=False, node_port=7890)
    #demo.launch(ssr_mode=False, node_port=7890, server_port=7891, share=True)
    #demo.launch(ssr_mode=False, node_port=7890, server_port=7892)
    #demo.launch(ssr_mode=True, node_port=7890, server_port=7891)
    #demo.launch(server_port=7891)
    #demo.launch(ssr_mode=True)
