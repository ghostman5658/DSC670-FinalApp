"""Load and query the Week 8 earnings-call models.

Both models are Qwen2.5-0.5B-Instruct weights in bf16 safetensors, stored in
subfolders of REPO_ID. Inference runs on CPU through PyTorch, so no paid API is
called.
"""

import gc
import os
import threading

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "ghostman5658/dsc670-meta-earnings-qwen2.5-0.5b-q8"

# Label shown in the UI -> subfolder in REPO_ID.
MODEL_FOLDERS = {
    "Fine-tuned (Week 8)": "finetuned",
    "Base Qwen2.5-0.5B-Instruct": "base",
}

# Exact system message used for the fine-tune (WK8 training data).
SYSTEM_PROMPT = (
    "You answer the way Meta executives answer on a public quarterly earnings call. "
    "Speak for the company. Answer every part of the question in the order it was asked. "
    "Use spoken sentences. No bullet points, no headlines, and no marketing slogans. "
    "When the analyst's name is known, open with a short thanks."
)

MAX_NEW_TOKENS_LIMIT = 700  # Same cap as the Week 8 held-out reading.
DEFAULT_NEW_TOKENS = 400  # Keeps CPU time reasonable on Streamlit Cloud.
CONTEXT_TOKENS = 2048
MAX_QUESTION_CHARS = 2000

_lock = threading.Lock()
_loaded: dict[str, tuple] = {}


def build_user_message(analyst: str, firm: str, quarter: str, question: str) -> str:
    """Format the user turn the same way the training JSONL does."""
    return f"Analyst: {analyst}, {firm}\nQuarter: {quarter}\n\n{question.strip()}"


def _thread_count() -> int:
    # Streamlit Community Cloud gives roughly 0.08 to 2 CPU cores.
    return max(1, min(2, os.cpu_count() or 1))


def load_model(label: str):
    """Download (once) and load one model. Only one model stays in memory.

    One bf16 copy of the model peaks at about 1.4 GB in this environment. Streamlit
    Community Cloud allows roughly 2.7 GB, so loading the second model evicts the first.
    """
    with _lock:
        if label not in _loaded:
            _loaded.clear()
            gc.collect()
            folder = MODEL_FOLDERS[label]
            local_root = snapshot_download(repo_id=REPO_ID, allow_patterns=[f"{folder}/*"])
            path = os.path.join(local_root, folder)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
            torch.set_num_threads(_thread_count())
            _loaded[label] = (tokenizer, model)
        return _loaded[label]


def answer(
    label: str,
    analyst: str,
    firm: str,
    quarter: str,
    question: str,
    max_new_tokens: int = DEFAULT_NEW_TOKENS,
) -> str:
    """Greedy (do_sample=False) answer from the selected model."""
    tokenizer, model = load_model(label)
    max_new = min(max_new_tokens, MAX_NEW_TOKENS_LIMIT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(analyst, firm, quarter, question)},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    prompt_len = encoded["input_ids"].shape[1]
    if prompt_len + max_new > CONTEXT_TOKENS:
        raise ValueError(
            f"Question is too long ({prompt_len} prompt tokens). Shorten it or lower the token cap."
        )
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
