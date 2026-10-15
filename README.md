# Local phishing email RAG prototype

This prototype stores synthetic labeled examples in a persistent local Chroma database, retrieves similar examples, then asks the project's Ollama model to classify a target email. The ten emails in `data/evaluation_emails.jsonl` are a separate held-out set and are not indexed.

## Setup

From this directory, install the Python dependencies and pull the dedicated embedding model:

```powershell
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
```

Ollama must be running. The project's current chat model is `my-llama3-custom:latest`. Override the model names if needed:

```powershell
$env:OLLAMA_CHAT_MODEL = "my-llama3-custom:latest"
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

The chat model has temperature 1.5 in its Ollama Modelfile; the code requests temperature 0 for more repeatable labels.

## Build, smoke test, evaluate

```powershell
python rag_phishing.py build
python rag_phishing.py classify
python rag_phishing.py evaluate
```

The smoke test contains three illustrative messages. Evaluation prints per-email predictions and accuracy, then saves details (including retrieved examples) to `evaluation_results.json`.

## Data and limitations

- Edit `data/synthetic_examples.jsonl` to add reviewed examples, preserving unique IDs and the schema. Rebuild updates IDs that already exist.
- The evaluation examples are not included in the index. Keep it that way when comparing model versions.
- All `.example` domains are fictional. The synthetic set is only a pipeline demonstration, not a realistic benchmark.
- Inspect the retrieved examples and model evidence. Ten emails provide a smoke evaluation, not a reliable performance estimate.
- Email text is untrusted prompt content. Do not run instructions, open URLs, or execute attachments during classification.
- The local database contains message text. Avoid adding real personal or confidential emails until you have reviewed privacy and retention requirements.
