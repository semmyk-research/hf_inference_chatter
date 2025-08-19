#from dotenv import load_dotenv
from typing import Iterable, Literal, Optional
import os
import gradio as gr
from huggingface_hub import InferenceClient, login

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


class HFChatClient:
    """Encapsulates `huggingface_hub.InferenceClient` setup and chat calls.

    Supports three backends:
    - model: plain Hugging Face model id (e.g., "HuggingFaceH4/zephyr-7b-beta")
    - provider: provider-routed model id (e.g., "Qwen/Qwen2.5-72B-Instruct:fireworks-ai")
    - endpoint: full inference endpoint URL (e.g., "http://localhost:1234").

    The class is stateless across calls except for cached token; it builds a fresh
    client per submitted settings to avoid stale configuration.
    """

    def __init__(
        self,
        default_model: str = "openai/gpt-oss-120b",
        token: Optional[str] = None,
    ) -> None:
        self.default_model = default_model
        
        # Best-effort login to reuse HF CLI auth if available, else fall back to token
        try:
            login()  # Will be a no-op in Spaces or if already logged in                
        except Exception:
            if self.token:
                login(token=self.token)
            else:
                # Respect common env var names; prefer explicit arg when provided
                self.token = (
                    token
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    )
                login(token=self.token)
            # Silent fallback; client will still work if token is passed directly
            pass    

    @staticmethod
    def _normalise_history(
        history: list,
        system_message: str,
        latest_user_message: str,
    ) -> list[dict]:
        """Return OpenAI-style messages list, resilient to tuples-history."""
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        for item in history or []:
            if isinstance(item, dict) and "role" in item and "content" in item:
                # Already messages format
                if item["role"] in ("user", "assistant"):
                    messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": latest_user_message})
        return messages

    @staticmethod
    def _select_model_target(
        backend: Literal["model", "provider", "endpoint"],
        model_id: Optional[str],
        provider: Optional[str],
        endpoint_url: Optional[str],
        default_model: str,
    ) -> str:
        if backend == "endpoint":
            if not endpoint_url:
                raise ValueError("Endpoint URL must be provided when backend is 'endpoint'.")
            return endpoint_url
        if backend == "provider":
            if not model_id or not provider:
                raise ValueError("Both model id and provider must be provided for provider backend.")
            return f"{model_id}:{provider}"
           ## Preferred over
           # #client = InferenceClient(provider="featherless-ai", model="models/HuggingFaceH4/zephyr-7b-beta")  ##TypeError: InferenceClient.__init__() got an unexpected keyword argument 'provider'

        # backend == "model"
        return model_id or default_model

    def chat(
        self,
        message: str,
        history: list,
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        backend: Literal["model", "provider", "endpoint"],
        model_id: Optional[str],
        provider: Optional[str],
        endpoint_url: Optional[str],
        api_token: Optional[str] = None,
    ) -> Iterable[str]:
        """Yield assistant text increments for streaming; else one final string.

        Returns a generator for Gradio ChatInterface compatibility.
        """
        messages = self._normalise_history(history, system_message, message)
        target = self._select_model_target(backend, model_id, provider, endpoint_url, self.default_model)
        client = InferenceClient(model=target, token=api_token or self.token)

        if stream:
            chat_accumulated_text = ""
            for chunk in client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
            ):
                delta = getattr(chunk.choices[0].delta, "content", None) or ""
                if delta:
                    accumulated_text += delta
                    # Preface once with backend indicator for clarity
                    yield chat_accumulated_text
        else:
            result = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=False,
                temperature=temperature,
                top_p=top_p,
            )
            yield result.choices[0].message.content


def build_ui() -> gr.Blocks:
    """
    Compose the Gradio UI using ChatInterface with tasteful defaults.
    Further Reading: For information on how to customise the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/gradio/chatinterface
    """
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

    with gr.Blocks(theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:
        gr.Markdown(
            """
            <div style="text-align:center">
            <h1>🤖 HF Inference ChatBox</h1>
            <p>Chat with models via Hugging Face model ids, provider routing, or custom endpoints.</p>
            </div>
            """
        )

        chatbot = gr.Chatbot(
            placeholder="Ask anything...",
            show_copy_button=True,
            bubble_full_width=False,
        )

        backend_choice = gr.Dropdown(
            choices=["Hugging Face Model ID", "HF Provider Route", "Inference Endpoint URL"],
            value="HF Provider Route",
            label="Backend",
        )
        with gr.Row():
            model_tb = gr.Textbox(value="openai/gpt-oss-120b", label="Model ID")
            provider_dd = gr.Dropdown(
                choices=["fireworks-ai", "together-ai", "openrouter-ai", "hf-inference"],
                value="fireworks-ai",
                label="Provider",
            )
        endpoint_tb = gr.Textbox(value="", label="Inference Endpoint URL (http://...)")
        api_token_tb = gr.Textbox(value="", label="HF API Token (optional)", type="password")

        ci = gr.ChatInterface(
            fn=gradio_chat_fn,
            type="messages",
            chatbot=chatbot,
            additional_inputs=[
                backend_choice,
                model_tb,
                provider_dd,
                endpoint_tb,
                gr.Textbox(value="You are a helpful assistant.", label="System message"),
                gr.Slider(minimum=1, maximum=16384, value=4096, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.05, label="Top-p (nucleus sampling)"),
                gr.Checkbox(value=True, label="Stream"),
                api_token_tb,
            ],

            title="HF Inference ChatBox",
            description=(
                "Chat using `gr.ChatInterface` with multiple backends. "
                "Read more: Gradio ChatInterface and HF auth docs."
            ),
            examples=[
                "Summarise explanatory and explanatory ethos.",
                "Give me a Python snippet to parse JSON.",
                "What's the difference between temperature and top_p?",
            ],
            save_history=True,
            fill_height=True,
            fill_width=True,
        )

    return demo

"""
Run the Gradio-based app
"""
if __name__ == "__main__":
    app = build_ui()
    app.launch(ssr_mode=False, node_port=7890)
    #app.launch(ssr_mode=True, node_port=7890)
    app.launch(ssr_mode=False, node_port=7890, server_port=7891)
