import json
import os
from pathlib import Path

from prompts import build_inference_prompt
from score import parse_response, score_prediction


def _load_done_ids(results_path: str) -> set[str]:
    done = set()
    p = Path(results_path)
    if not p.exists():
        return done
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _load_documents(documents_path: str) -> list[dict]:
    docs = []
    with open(documents_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# GPT-4o-mini backend


def _infer_openai(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""



# HuggingFace backend


def _load_hf_model(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return tokenizer, model


def _infer_hf_batch(tokenizer, model, prompts: list[str]) -> list[str]:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    tokenizer.padding_side = "left"
    enc = tokenizer(
        chat_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # decode only newly generated tokens
    results = []
    for i, ids in enumerate(out):
        n_input = enc["input_ids"].shape[1]
        new_ids = ids[n_input:]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results



# Main entry point

def evaluate_model(
    model_name: str,
    documents_path: str,
    results_path: str,
) -> None:
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    docs = _load_documents(documents_path)
    done_ids = _load_done_ids(results_path)

    pending = [d for d in docs if d["doc_id"] not in done_ids]
    print(
        f"[{model_name}] {len(docs)} total docs, "
        f"{len(done_ids)} already done, {len(pending)} to evaluate."
    )

    out_f = open(results_path, "a", encoding="utf-8")

    try:
        if model_name == "gpt-4o-mini":
            import openai
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            for idx, doc in enumerate(pending):
                prompt = build_inference_prompt(doc["sentences"])
                raw = _infer_openai(client, prompt)
                _write_result(out_f, model_name, doc, raw)
                print(f"  [{idx + 1}/{len(pending)}] {doc['doc_id']}")

        else:
            from config import HF_MODEL_IDS
            model_id = HF_MODEL_IDS[model_name]
            tokenizer, model = _load_hf_model(model_id)

            batch_size = 32
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start: batch_start + batch_size]
                prompts = [build_inference_prompt(d["sentences"]) for d in batch]
                responses = _infer_hf_batch(tokenizer, model, prompts)
                for doc, raw in zip(batch, responses):
                    _write_result(out_f, model_name, doc, raw)
                end_idx = min(batch_start + batch_size, len(pending))
                print(
                    f"  [{end_idx}/{len(pending)}] batch "
                    f"{batch_start // batch_size + 1} done"
                )
    finally:
        out_f.close()

    print(f"[{model_name}] Evaluation complete. Results written to {results_path}")


def _write_result(f, model_name: str, doc: dict, raw: str) -> None:
    prediction = parse_response(raw)
    ground_truth = tuple(doc["contradiction_pair"])
    correct = score_prediction(prediction, ground_truth) if prediction is not None else False

    record = {
        "model": model_name,
        "doc_id": doc["doc_id"],
        "distance": doc["distance"],
        "distractor_count": doc["distractor_count"],
        "n_sentences": doc["n_sentences"],
        "raw_response": raw,
        "prediction": list(prediction) if prediction is not None else None,
        "correct": correct,
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()
