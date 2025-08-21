---
title: HF Inference ChatBox
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: mit
short_description: 'HF Inference ChatBox: First Take'
author: https://github.com/semmyk-research
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## Overview

This app is a Gradio ChatInterface demo that wraps the model client logic in a reusable Python class for portability and maintainability.  
You can converse with models via:

- Hugging Face model id (e.g., `HuggingFaceH4/zephyr-7b-beta`)
- HF Provider routing (e.g., `openai/gpt-oss-120b` with provider `fireworks-ai`)
- Custom inference endpoint URL (e.g., `http://localhost:1234`)

UI is built with `gr.ChatInterface` and a minimal, polished theme.


This is a simple chatbot that uses the Huggingface_hub InferenceClient to respond to the user's message.
It is a working example: a building block to building complex extensive RAG-KG in the pipeline.
Refactored chatbot using a portable client class around `huggingface_hub.InferenceClient`.

References:
- Hugging Face Hub Inference:  [https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference)  
  -- Huggingface_hub InferenceClient, as at Aug 2025, does not support openai responses methods.
- Gradio ChatInterface docs: [gradio.app/docs/gradio/chatinterface](https://www.gradio.app/docs/gradio/chatinterface)
- HF Hub authentication quickstart: [huggingface.co/docs/huggingface_hub/v0.34.4/en/quick-start#authentication](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/quick-start#authentication)

## Run

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Set authentication (choose one) [optional]

- Via CLI login (interactive):

```bash
huggingface-cli login
```

- or via environment .env

```bash
HF_TOKEN=hf_xxx  # or HUGGINGFACEHUB_API_TOKEN
```

- or via environment variable:

```bash
export HF_TOKEN=hf_xxx  # or HUGGINGFACEHUB_API_TOKEN
```

Details: see HF auth guide: [link](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/quick-start#authentication).

3) Start the app

```bash
python app.py
```

Open the printed URL, then chat.

## Using Backends

- Backend = "Hugging Face Model ID": provide a plain model id in the Model field.
- Backend = "HF Provider Route": provide a model id and choose a Provider (e.g., `fireworks-ai`). The app builds `model_id:provider` for routing.
- Backend = "Inference Endpoint URL": provide `http://...` in Endpoint; requests go directly to the endpoint.

The first line of each response includes a small indicator of the active backend, model/provider/endpoint for clarity.

## Logout (Shared PCs)

You can explicitly log out of Hugging Face from the UI via the "Logout from Hugging Face" button. This clears the in-process auth and removes `HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN` from the environment so the token is not reused on shared machines.

## Code Structure

- `hf_client.HFChatClient`: small class encapsulating client construction and the `chat` call. It normalizes history to OpenAI-style messages and supports streaming.
- Original implementation is preserved and commented out in `app.py` for reference; no code was deleted.
 - `logging_utils`: structured JSON logging setup used by the app.
 - `tests/test_hf_client.py`: unit tests for backend target selection and history normalization.

## Gradio UI

The app uses `gr.ChatInterface` with additional inputs for backend, model/provider/endpoint, system message, generation params, and token. For component and parameter details, see ChatInterface docs: [link](https://www.gradio.app/docs/gradio/chatinterface).

## Structured Logging

Logs are emitted in JSON to stdout with keys like `ts`, `level`, `logger`, `message`, plus extra fields (e.g., `backend`, `model`). You can pipe them to `jq` or your log shipper.

## Tests

Run unit tests with:

```bash
pytest -q
```
