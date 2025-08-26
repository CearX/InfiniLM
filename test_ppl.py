#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from typing import List, Tuple

import requests

try:
    from datasets import load_dataset  # type: ignore
    import torch
    from transformers import MambaForCausalLM, AutoTokenizer
    from tqdm import tqdm
except Exception as e:  # pragma: no cover
    print("[ERROR] 请先安装依赖: pip install -U datasets requests torch transformers tqdm", file=sys.stderr)
    raise


def fetch_logprobs(
    api_base: str,
    model: str,
    text: str,
    timeout: float,
) -> Tuple[List[float], int]:
    """调用 /completions，返回 prompt 的 token logprobs 列表与 token 数。

    要求服务端实现 echo=true 且支持 logprobs>=1，max_tokens=0。
    如果服务端未返回 logprobs，将抛出 RuntimeError。
    """
    url = api_base.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:512]}")
    data = resp.json()
    try:
        choice = data["choices"][0]
        lp = choice["logprobs"]
    except Exception:
        raise RuntimeError(
            "服务未返回 logprobs 字段。请确保服务支持 /completions 的 echo 与 logprobs。"
        )
    token_logprobs = lp.get("token_logprobs")
    if token_logprobs is None:
        raise RuntimeError(
            "logprobs.token_logprobs 为空。请在服务端实现 token 级对数概率返回。"
        )
    # 过滤 None（例如特殊符号），仅聚合有效对数概率
    valid_lps = [x for x in token_logprobs if x is not None]
    return valid_lps, len(valid_lps)


def compute_ppl_on_dataset(
    api_base: str,
    model: str,
    dataset_name: str,
    config: str,
    split: str,
    max_samples: int,
    timeout: float,
) -> float:
    ds = load_dataset(dataset_name, config, split=split)

    total_nll = 0.0
    total_tokens = 0
    processed = 0

    for item in ds:
        # 不同配置字段名不同：wikitext-*-raw-v1 通常字段为 'text'
        text = (item.get("text") or "").strip()
        if not text:
            continue
        try:
            token_logprobs, num_toks = fetch_logprobs(
                api_base, model, text, timeout)
        except Exception as e:
            # 将最早的错误直接抛出，便于用户修复服务端
            raise

        if num_toks == 0:
            continue
        total_tokens += num_toks
        total_nll += -sum(token_logprobs)  # NLL = -log p(token)
        processed += 1

        if 0 < max_samples <= processed:
            break

    if total_tokens == 0:
        raise RuntimeError("未获得任何 token 的 logprobs，无法计算 PPL。")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def compute_ppl_pytorch(model, tokenizer, texts, max_length=1024, max_samples=None):
    """使用 PyTorch Mamba 计算 PPL"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    processed = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PyTorch PPL"):
            # 分词
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)

            if input_ids.size(1) < 2:  # 需要至少2个token来计算loss
                continue

            # 前向传播
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # 累积loss和token数
            total_loss += loss.item() * (input_ids.size(1) - 1)  # 减1因为最后一个token没有target
            total_tokens += input_ids.size(1) - 1
            processed += 1

            # 检查是否达到最大样本数
            if max_samples and max_samples > 0 and processed >= max_samples:
                break

    # 计算PPL
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPL via OpenAI /completions (echo+logprobs)")
    parser.add_argument("--api-base", type=str,
                        default="http://127.0.0.1:8080", help="服务地址")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称（/models 返回的 id）")
    parser.add_argument("--dataset", type=str,
                        default="wikitext", help="HF datasets 名称")
    parser.add_argument("--config", type=str,
                        default="wikitext-2-raw-v1", help="HF datasets 配置名")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集划分，例如 test/validation")
    parser.add_argument("--max-samples", type=int,
                        default=500, help="最多评估多少条样本（>0 生效）")
    parser.add_argument("--timeout", type=float,
                        default=120.0, help="HTTP 请求超时秒数")

    args = parser.parse_args()

    # 计算 Rust 版本 PPL
    print("=== 计算 Rust 版本 PPL ===")
    t0 = time.time()
    ppl_original = compute_ppl_on_dataset(
        api_base=args.api_base,
        model=args.model,
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        max_samples=args.max_samples,
        timeout=args.timeout,
    )
    dt_original = time.time() - t0
    print(f"Rust PPL = {ppl_original:.4f} (time: {dt_original:.2f}s)")

    # 计算 PyTorch 版本 PPL
    print("\n=== 计算 PyTorch 版本 PPL ===")
    print("Loading PyTorch model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/shared/models/mamba-2.8b-hf")
    model = MambaForCausalLM.from_pretrained(
        "/home/shared/models/mamba-2.8b-hf", device_map="cuda")

    # 加载相同的数据集
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    texts = [item["text"] for item in dataset if item["text"].strip()]

    t1 = time.time()
    ppl_pytorch = compute_ppl_pytorch(
        model, tokenizer, texts, max_samples=args.max_samples)
    dt_pytorch = time.time() - t1
    print(f"PyTorch 版本 PPL = {ppl_pytorch:.4f} (time: {dt_pytorch:.2f}s)")

    # 计算差异
    print("\n=== PPL 对比结果 ===")
    print(f"Rust PPL:    {ppl_original:.4f}")
    print(f"PyTorch PPL:   {ppl_pytorch:.4f}")
    diff_abs = abs(ppl_original - ppl_pytorch)
    diff_rel = diff_abs / ppl_pytorch * 100
    print(f"绝对差异:      {diff_abs:.4f}")
    print(f"相对差异:      {diff_rel:.2f}%")
    print(f"Rust 时间:    {dt_original:.2f}s")
    print(f"PyTorch 时间:   {dt_pytorch:.2f}s")


if __name__ == "__main__":
    main()
