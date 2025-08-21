#from dotenv import load_dotenv
from typing import Iterable, Literal, Optional
import os
import gradio as gr
from huggingface_hub import InferenceClient, login
from logging_utils import setup_logging
from hf_client import HFChatClient
from provider_validator import is_valid_provider, suggest_providers

"""
This is a simple chatbot that uses the Huggingface_hub InferenceClient to respond to the user's message.
It is a working example: a building block to building complex extensive RAG-KG in the pipeline.
Refactored chatbot using a portable client class around `huggingface_hub.InferenceClient`.

References:
- Hugging Face Hub Inference:  https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
-- Huggingface_hub InferenceClient, as at Aug 2025, does not support openai responses methods.
- Gradio ChatInterface docs: https://www.gradio.app/docs/gradio/chatinterface
- Hugging Face Hub auth: https://huggingface.co/docs/huggingface_hub/v0.34.4/en/quick-start#authentication
"""

# Preserve original implementation (commented out)


# Explicitly disable implicit token propagation; we rely on explicit auth or env var
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


'''
Original in-file HFChatClient (now replaced by import)
'''


def build_ui() -> gr.Blocks:
    """
    Compose the Gradio UI using ChatInterface with tasteful defaults.
    Further Reading: For information on how to customise the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/gradio/chatinterface
    """
    setup_logging()  ## set logging

    chat_client = HFChatClient()  ## initialise the chat client class

    def gradio_chat_fn(
        message: str,
        history: list,
        backend: str,
        model_id: str,
        provider: str,
        endpoint_url: str,
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        api_token: str,
    ):
        backend_literal: Literal["model", "provider", "endpoint"] = (
            "model" if backend == "Hugging Face Model ID" else (
                "provider" if backend == "HF Provider Route" else "endpoint"
            )
        )
        # Include a compact indicator in the first tokens returned
        indicator = f"[Backend: {backend_literal} | Model: {model_id or '-'} | Provider: {provider or '-'} | Endpoint: {endpoint_url or '-'}]\n\n"
        first = True
        for text in chat_client.chat(
            message=message,
            history=history,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            backend=backend_literal,
            model_id=model_id or None,
            provider=provider or None,
            endpoint_url=endpoint_url or None,
            api_token=(api_token or None),
        ):
            if first:
                yield indicator + text
                first = False
            else:
                yield indicator + text

    with gr.Blocks(theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:  #chat_area:  #demo:
        gr.Markdown(
            """
            <div style="text-align:center">
            <h1>🤖 HF Inference ChatBox</h1>
            <p>Chat with models via Hugging Face model ids, provider routing, or custom endpoints.</p>
            </div>
            """
        )
        
        with gr.Row():
            # Backend option
            backend_choice = gr.Dropdown(
                choices=["Hugging Face Model ID", "HF Provider Route", "Inference Endpoint URL"],
                value="HF Provider Route",
                label="Backend",
                )
            model_tb = gr.Textbox(value="openai/gpt-oss-120b", label="Model ID")
            provider_dd = gr.Dropdown(
                choices=["fireworks-ai", "together-ai", "openrouter-ai", "hf-inference"],
                value="fireworks-ai",
                label="Provider",
                allow_custom_value=True,  # let users type new providers as they appear
            )
        
        # Validate provider on change; warn but allow continue
        def on_provider_change(provider_value: str):
            if not provider_value:
                return
            if not is_valid_provider(provider_value):
                sug = suggest_providers(provider_value)
                extra = f" Suggestions: {', '.join(sug)}." if sug else ""
                gr.Warning(
                    f"Provider not on HF provider list. See https://huggingface.co/docs/inference-providers/index.{extra}"
                )

        provider_dd.change(on_provider_change, inputs=provider_dd, outputs=None)

        chatbot = gr.Chatbot(
            placeholder="Ask anything...",
            show_copy_button=True,
            #bubble_full_width=False,  #Future: 'bubble_full_width' parameter is deprecated 
            type="messages",  ## Future deprecation warning
        )

        '''
        # Define model parameter components as variables for additional_inputs
        system_message_tb = gr.Textbox(value="You are a helpful assistant.", label="System message")
        max_token_sl = gr.Slider(minimum=512, maximum=16384, value=4096, step=128, label="Max new tokens")
        temperature_sl = gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="Temperature")
        top_p_sl = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.5, label="Top-p (nucleus sampling)")
        stream_sl = gr.Checkbox(value=True, label="Stream")
        '''

        
        # Clean UI: Model parameters hidden in collapsible accordion
        with gr.Accordion("⚙️ Model Settings", open=False):
            '''
            with gr.Row():
                gr.Markdown("**Backend Configuration**")
            with gr.Row():
                backend_choice.render()
                model_tb.render()
                provider_dd.render()
            with gr.Row():
                endpoint_tb.render()
                api_token_tb.render()
            '''
            
            with gr.Row():
                endpoint_tb = gr.Textbox(value="", label=f"Inference Endpoint URL")
                api_token_tb = gr.Textbox(value="", label="HF API Token (optional)", type="password")

            with gr.Row():
                gr.Markdown("**Generation Parameters**")
            with gr.Row():
                system_message_tb = gr.Textbox(value="You are a helpful assistant.", label="System message")
                max_token_sl = gr.Slider(minimum=512, maximum=16384, value=4096, step=128, label="Max new tokens")
            with gr.Row():
                temperature_sl = gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="Temperature")
                top_p_sl = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Top-p")  #(nucleus sampling)")
                stream_sl = gr.Checkbox(value=True, label="Stream")

        # Logout controls
        def do_logout():
            ok = chat_client.logout()
            # Reset token textbox on successful logout
            msg = "✅ Logged out of Hugging Face and cleared tokens." if ok else "⚠️ Logout failed."
            return gr.update(value=""), gr.update(visible=True, value=msg)
        
        logout_status = gr.Markdown(visible=False)
        logout_btn = gr.Button("Logout from Hugging Face", variant="stop")

        logout_btn.click(fn=do_logout, inputs=None, outputs=[api_token_tb, logout_status])        


        # Chat Interface - clean and simple
        ci = gr.ChatInterface(
            fn=gradio_chat_fn,
            type="messages",
            chatbot=chatbot,
            additional_inputs=[
                backend_choice,
                model_tb,
                provider_dd,
                endpoint_tb,
                ## additional inputs: model parameters
                system_message_tb,
                max_token_sl,
                temperature_sl,
                top_p_sl,
                stream_sl,
                api_token_tb,
            ],
            title="HF Inference ChatBox",
            description=(
                "Chat using `gr.ChatInterface` with multiple backends. "
                "Read more: Gradio ChatInterface and HF auth docs."
            ),
            # Each example is [message, backend, model_id, provider, endpoint_url, system_message, max_tokens, temperature, top_p, stream, api_token]
            #examples=[                
            #    ["Summarise Critical realism explanatory and explanatory ethos.", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
            #    ["Give me a Python snippet to parse JSON.", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
            #    ["What's the difference between temperature and top_p?", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
                #["Summarise explanatory and explanatory ethos.", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
                #["Give me a Python snippet to parse JSON.", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
                #["What's the difference between temperature and top_p?", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
            #],
            #examples_per_page=0,  # Show examples but collapsed by default. ## No longer supported
            save_history=True,
            fill_height=True,
            fill_width=True,
        )

        #'''
        ## Handle examples. Moved away from ChatInterface to allow for more control over the layout
        with gr.Accordion("📚 Examples", open=False):
            gr.Examples(
                # Each example is [message, backend, model_id, provider, endpoint_url, system_message, max_tokens, temperature, top_p, stream, api_token]
                examples=[
                    ["Summarise Critical realism explanatory and explanatory ethos.", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
                    #[{"role": "user", "content": "Summarise Critical realism explanatory and explanatory ethos."}, "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
                    ["Give me a Python snippet to parse JSON.", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
                    ["What is the difference between temperature and top_p?", "Hugging Face Model ID", "openai/gpt-oss-120b", "fireworks-ai", "", "You are a helpful assistant.", 4096, 0.5, 0.2, True, ""],
                    #["Summarise explanatory and explanatory ethos.", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
                    #["Give me a Python snippet to parse JSON.", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
                    #["What's the difference between temperature and top_p?", backend_choice, model_tb, provider_dd, "", system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, ""],
                ],            
                inputs=[ci.textbox, backend_choice, model_tb, provider_dd, endpoint_tb, system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, api_token_tb],
                #inputs=[backend_choice, model_tb, provider_dd, endpoint_tb, system_message_tb, max_token_sl, temperature_sl, top_p_sl, stream_sl, api_token_tb],
                outputs=chatbot,
                label="Example Conversations",
            )
        #'''

        # Now update additional_inputs to reference the components defined above
        ci.additional_inputs[4] = system_message_tb  # Replace the inline Textbox
        ci.additional_inputs[5] = max_token_sl       # Replace the inline Slider
        ci.additional_inputs[6] = temperature_sl     # Replace the inline Slider
        ci.additional_inputs[7] = top_p_sl           # Replace the inline Slider
        ci.additional_inputs[8] = stream_sl          # Replace the inline Checkbox

    return demo  #chat_area  #demo

"""
Run the Gradio-based app
"""
if __name__ == "__main__":
    app = build_ui()
    #app.launch(ssr_mode=False, node_port=7890)
    app.launch(ssr_mode=True, node_port=7890)
    #app.launch(ssr_mode=False, node_port=7890, server_port=7891)
