import re

import streamlit as st

from earnings_model import (
    DEFAULT_NEW_TOKENS,
    MAX_NEW_TOKENS_LIMIT,
    MAX_QUESTION_CHARS,
    MODEL_FOLDERS,
    REPO_ID,
    answer,
)

EXAMPLE = {
    "analyst": "Brian Nowak",
    "firm": "Morgan Stanley",
    "quarter": "Q3 2025",
    "question": (
        "Thanks for taking my questions. I have two for Susan. The first one, Susan, so the pipeline "
        "for core improvements to come in '26 with models and ad ranking models and more types of "
        "compute seems very exciting, and the infrastructure build seems sizable behind that. So can "
        "you help us a little understand some of the early quantifiable signals you're seeing on A/B "
        "tests from some of these improvements to come that sort of make you most excited and give "
        "you confidence you're going to get ROIC from all this CapEx. That's the first one. Second "
        "one's a little faster. How large is the Reality Labs revenue headwind in the 4Q guidance? Thanks."
    ),
}

QUARTER_PATTERN = re.compile(r"^Q[1-4] \d{4}$")

st.set_page_config(page_title="Meta earnings-call voice", page_icon="📈", layout="wide")

st.title("Meta earnings-call voice")
st.write(
    "Ask an analyst-style question and see how a Qwen2.5-0.5B model answers it. The fine-tuned model "
    "was trained on Meta's public earnings-call Q&A (Q3 2018 to Q4 2024). The base model is the "
    "unmodified starting point."
)

st.warning(
    "**Read before using.** In the Week 8 held-out evaluation, the fine-tuned model did worse than the "
    "base model on the held-out quarter. It copies the spoken call style, but it can state figures "
    "that are not in the source. For example, one answer claimed a 29% decline in Reality Labs revenue "
    "that was never in the transcript. Treat every number as unverified."
)

with st.form("question_form"):
    col_a, col_b, col_c = st.columns([2, 2, 1])
    analyst = col_a.text_input("Analyst", value=EXAMPLE["analyst"], max_chars=80)
    firm = col_b.text_input("Firm", value=EXAMPLE["firm"], max_chars=80)
    quarter = col_c.text_input("Quarter", value=EXAMPLE["quarter"], help="Format: Q3 2025")

    question = st.text_area(
        "Analyst question",
        value=EXAMPLE["question"],
        height=160,
        max_chars=MAX_QUESTION_CHARS,
        help=f"Up to {MAX_QUESTION_CHARS} characters.",
    )

    col_mode, col_tokens = st.columns([2, 1])
    mode = col_mode.radio(
        "Models",
        options=["Compare both", *MODEL_FOLDERS.keys()],
        horizontal=True,
    )
    max_tokens = col_tokens.slider(
        "Max answer tokens",
        min_value=100,
        max_value=MAX_NEW_TOKENS_LIMIT,
        value=DEFAULT_NEW_TOKENS,
        step=50,
        help="Longer answers take more CPU time on Streamlit Cloud.",
    )

    submitted = st.form_submit_button("Generate answer", type="primary")

if submitted:
    errors = []
    if not analyst.strip() or not firm.strip():
        errors.append("Enter the analyst name and firm.")
    if not QUARTER_PATTERN.match(quarter.strip()):
        errors.append("Quarter must look like `Q3 2025`.")
    if not question.strip():
        errors.append("Enter a question.")

    if errors:
        for message in errors:
            st.error(message)
    else:
        labels = list(MODEL_FOLDERS.keys()) if mode == "Compare both" else [mode]
        columns = st.columns(len(labels))
        for column, label in zip(columns, labels):
            with column:
                st.subheader(label)
                with st.spinner("Loading model and writing the answer. This can take a few minutes on CPU."):
                    try:
                        text = answer(
                            label,
                            analyst.strip(),
                            firm.strip(),
                            quarter.strip(),
                            question.strip(),
                            max_new_tokens=max_tokens,
                        )
                    except Exception as exc:  # Surface load or inference failures in the UI.
                        st.error(f"Could not generate an answer: {exc}")
                        continue
                st.write(text)
                st.caption(f"{len(text.split())} words")

st.divider()
st.caption(
    f"Models: [{REPO_ID}](https://huggingface.co/{REPO_ID}) (bf16 safetensors, run on CPU with "
    "PyTorch and Transformers). Greedy decoding, so the same input gives the same output. The Week 8 "
    "notebook ran the same weights in float32, so later tokens can differ slightly."
)
