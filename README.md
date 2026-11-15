# Meta earnings-call voice

A Streamlit app for the DSC670 final milestone. Enter an analyst question and compare a Qwen2.5-0.5B model fine-tuned on Meta's public earnings-call Q&A with the unmodified base model.

The model weights are hosted at [ghostman5658/dsc670-meta-earnings-qwen2.5-0.5b-q8](https://huggingface.co/ghostman5658/dsc670-meta-earnings-qwen2.5-0.5b-q8). They run on CPU with PyTorch and Transformers, so no paid API is called.

## Run locally

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run downloads about 2 GB of model files (one copy per model). Only one model is kept in memory at a time, which keeps the app within Streamlit Community Cloud's memory limit.

## Deploy on Streamlit Community Cloud

1. Create the app from this repo with `streamlit_app.py` as the entry point.
2. In **Advanced settings**, choose Python 3.12.
3. Under **Sharing**, decide who can view the app.

## Caveat

The fine-tuned model scored worse than the base model in the Week 8 held-out evaluation. It can state figures that are not in the source, so treat its numbers as unverified.
