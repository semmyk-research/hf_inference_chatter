from __future__ import annotations

from typing import Iterable, Literal, Optional
import os
from huggingface_hub import InferenceClient, login, logout as hf_logout
from logging_utils import get_logger


## Get logger instance
logger = get_logger(__name__)


class HFChatClient:
    """Encapsulate `huggingface_hub.InferenceClient` setup and chat calls.

    Backends:
    - model: plain HF model id (e.g., "HuggingFaceH4/zephyr-7b-beta")
    - provider: provider-routed id (e.g., "openai/gpt-oss-120b:fireworks-ai")
    - endpoint: full inference endpoint URL (e.g., "http://localhost:1234").
    """

    def __init__(self, default_model: str = "openai/gpt-oss-120b", token: Optional[str] = None) -> None:
        self.default_model = default_model
        self.token = token if token else None   #""  # invalid; preserved
        #self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")  ## not preferred

        # Disable implicit token propagation for determinism
        # Explicitly disable implicit token propagation; we rely on explicit auth or env var
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

        """
        commented out to preserve explicit login to HF
        # Prefer explicit arg; else common env vars
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        try:
            if self.token:
                login(token=self.token)
            else:
                login()
        except Exception as exc:  # non-fatal
            logger.warning("hf_login_failed", extra={"error": str(exc)})
        """
        
        # Privacy-first login: try interactive CLI first; fallback to provided/env token only if needed
        try:
            login()
            logger.info("hf_login", extra={"mode": "cli"})
        except Exception as exc:
            # Respect common env var names; prefer explicit token arg when provided
            fallback_token = self.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if fallback_token:
                try:
                    login(token=fallback_token)
                    self.token = fallback_token
                    logger.info("hf_login", extra={"mode": "token"})
                except Exception as exc_token:
                    logger.warning("hf_login_failed", extra={"error": str(exc_token)})
            else:
                logger.warning("hf_login_failed", extra={"error": str(exc)})
                # Silent fallback; client will still work if token is passed directly
                #pass

    @staticmethod
    def _normalise_history(history: list, system_message: str, latest_user_message: str) -> list[dict]:
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        for item in history or []:
            if isinstance(item, dict) and "role" in item and "content" in item:
                if item["role"] in ("user", "assistant"):
                    messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                usr, asst = item
                if usr:
                    messages.append({"role": "user", "content": usr})
                if asst:
                    messages.append({"role": "assistant", "content": asst})
        messages.append({"role": "user", "content": latest_user_message})
        return messages
	
	## flagged for deprecation or use as openai helper
    @staticmethod
    def _select_target(
        backend: Literal["model", "provider", "endpoint"],
        model_id: Optional[str],
        provider: Optional[str],
        endpoint_url: Optional[str],
        default_model: str,
    ) -> str:
        match backend:
            case "endpoint":
                if not endpoint_url:
                    raise ValueError("Endpoint URL is required for backend 'endpoint'.")
                return endpoint_url
            case "provider":
                if not model_id or not provider:
                    raise ValueError("Model id and provider are required for backend 'provider'.")
                return f"{model_id}:{provider}"  ## HF does not fully support openai AIClient
            case "model":
                return model_id or default_model
            case _:
                raise ValueError("Invalid backend.")

    @staticmethod
    def _initialise_client(self,
        backend: Literal["model", "provider", "endpoint"], 
        model_id: Optional[str] = None, 
        provider: Optional[str] = None, 
        endpoint_url: Optional[str] = None, 
        token: Optional[str] = None) -> InferenceClient:

        #target = self._select_target(backend, model_id, provider, endpoint_url, self.default_model)
        #client = InferenceClient(token=token)  #self.token)
        #logger.log(20, "client: ", extra={"backend":backend})  ## debug
        try:
            match backend:
                case "endpoint" | "model":
                    logger.debug("_initialise_client: initialising with:", extra={"model":model_id})  ## debug
                    client = InferenceClient(model=model_id, token=token)   #endpoint=target)   ##, token=api_token or self.token)
                    #client.model = model_id  ##target
                    #client = client(model=model_id, token=token)
                    logger.log(20, "client: ", extra={"model":model_id})  ## debug
                case "provider":
                    logger.info("_initialise_client: initialising with:", extra={"provider":provider})  ## debug
                    client = InferenceClient(provider=provider, model=model_id, token=token)  ##, token=api_token or self.token)
                    #client = client(model = model_id, provider=provider, token=token)   ##target
                    #client.model = model_id
                    #client.provider = provider
                    logger.log(20, "client: ", extra={"backend":backend})  ## debug
                case _:
                    raise ValueError("Invalid backend.")
            return client
        except Exception as exc:
            logger.log(40, "_initialise_client: client_init_failed", extra={"error": str(exc)})  ## debug
            raise RuntimeError(f"_initialise_client: Failed to initialise client: {exc}")

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
        messages = self._normalise_history(history, system_message, message)
        token = api_token or self.token

        logger.log(20,"chat: initialising client", extra={
            "backend": backend, "model": model_id, "provider": provider, "endpoint": endpoint_url,
            "stream": stream, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        })
        #target = self._select_target(backend, model_id, provider, endpoint_url, self.default_model)
        try:
            client = self._initialise_client(self, backend, model_id, provider, endpoint_url, token)  #api_token)
            logger.log(20, "chat: client initialised")  ## debug
        except Exception as exc:
            ##logger.error
            logger.log(40,"chat client_init_failed", extra={"error": str(exc)})
            raise RuntimeError(f"chat: Failed to initialise client: {exc}")
        
        logger.log(20, "chat_start", extra={
                    "backend": backend, "model": model_id, "provider": provider, "endpoint": endpoint_url,
                    "stream": stream, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                })
        
        if stream:
            acc = ""
            for chunk in client.chat_completion(
                messages=messages,
                #model=client.model,  ## moved back to client initialise
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
            ):
                delta = getattr(chunk.choices[0].delta, "content", None) or ""
                if delta:
                    acc += delta
                    yield acc
            return

        result = client.chat_completion(
            messages=messages,
            #model=client.model,  ## moved back to client initialised
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
            top_p=top_p,
        )
        yield result.choices[0].message.content

    def logout(self) -> bool:
        """Logout from Hugging Face and clear in-process tokens.

        Returns True on success, False otherwise.
        """
        try:
            hf_logout()
        except Exception as exc:
            logger.error("hf_logout_failed", extra={"error": str(exc)})
            return False
        # Clear process environment tokens
        for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            if key in os.environ:
                os.environ.pop(key, None)
        self.token = None
        logger.info("hf_logout_success")
        return True

