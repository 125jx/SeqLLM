"""Video Recommendation Evaluation (Pass@k / Recall@k) with vLLM.

LLaMA-Factory-integrated 版本：在保留 OpenOneRec 官方评估逻辑（beam search /
subtree-constrained / pass@k / recall@k / 标准输出格式）的同时，做了三处适配，
使其能与 `llm_based_user_sequence_modelling` 这套 LLaMA-Factory 训练流水线无缝衔接：

  1. 参数命名与 LLaMA-Factory 对齐
     - ``--model_name_or_path`` 替代 ``--model_path``
     - ``--template``           表示 LLaMA-Factory 训练时使用的 chat 模板名
                                （默认 ``qwen``，与 ``examples/onerec/*.yaml`` 一致）。
     - ``--dataset`` / ``--dataset_dir``
                                通过 ``data/dataset_info.json`` 里注册的数据集名
                                （例如 ``video_rec_eval``）解析出真实 jsonl 路径，
                                避免训练 / 评测两边各自维护一份路径。
     - ``--data_path``          仍然保留，直接指定 jsonl 文件作为兜底入口。

  2. Chat template 解析策略
     - 默认：直接使用 ``tokenizer`` 自带的 ``chat_template``。
       训练时 LLaMA-Factory 在保存 ckpt 时已经把对应 template（如 ``template: qwen``）
       写入 ``chat_template.jinja`` / ``tokenizer_config.json``，因此评测端无须二次设置。
     - 如显式传 ``--chat_template_file path/to/xxx.jinja2``，则覆盖。
       这条主要用于对比官方 OneRec-8B 等外部 checkpoint。

  3. 入口形式
     - 同时接受 CLI 参数与 ``--config YAML``（YAML 字段命名与 LLaMA-Factory 训练
       YAML 风格一致），并允许 CLI 覆盖 YAML。

输出格式与 OpenOneRec 官方对齐:
  <output_dir>/test_generated.json   # 包含 prompt / ground_truth / generations / per-sample 指标
  <output_dir>/debug.json            # passed / failed / no-generation 样本前 N 条
  <output_dir>/summary.json          # 汇总指标 + 配置快照

支持模式:
  - ``--num_beams N``                       beam search
  - ``--constrained subtree``               codebook 子树约束的 step-by-step beam
                                            （需要 ``--codebook_path`` 由 build_codebook.py 产出）

用法（推荐 YAML 风格，对齐 LLaMA-Factory）:
    python scripts/eval_video_rec.py --config examples/onerec/eval_video_rec.yaml

或纯 CLI:
    python scripts/eval_video_rec.py \\
        --model_name_or_path /path/to/ckpt \\
        --template qwen \\
        --dataset video_rec_eval \\
        --dataset_dir /apdcephfs/private_bobjxzhang/llm_based_user_sequence_modelling/data \\
        --output_dir ./results/exp1 \\
        --num_beams 32 --max_new_tokens 3 --k_values 1,32 \\
        --tensor_parallel_size 8 --gpu_memory_utilization 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import yaml
from transformers import AutoTokenizer


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TEMPLATE = "qwen"  # 与 examples/onerec/*.yaml 中的 template 字段一致
DEFAULT_DATASET_DIR = (
    "/apdcephfs_cq11/share_303717182/bobjxzhang/llm_based_user_sequence_modelling/data"
)
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "onerec",
    "results",
)


# ============================================================================
# Utility: SID extraction / metric (官方 utils.py 逻辑)
# ============================================================================


def extract_ids_from_answer(answer: str) -> List[str]:
    """从答案字段提取所有 SID，保留首次出现顺序，并去重。"""
    seen: Set[str] = set()
    out: List[str] = []
    for part in answer.split("<|sid_begin|>"):
        if "<|sid_end|>" in part:
            sid = part.split("<|sid_end|>")[0].strip()
            if sid and sid not in seen:
                out.append(sid)
                seen.add(sid)
    return out


def extract_first_id_from_answer(answer: str) -> Optional[str]:
    ids = extract_ids_from_answer(answer)
    return ids[0] if ids else None


def extract_id_from_generation(generation: str) -> Optional[str]:
    """从模型生成中提取一个 SID（容忍多种格式）。"""
    generation = generation.strip()
    if "</think>" in generation:
        generation = generation.split("</think>")[-1].strip()
    if "<|sid_begin|>" in generation:
        for part in generation.split("<|sid_begin|>"):
            if "<|sid_end|>" in part:
                sid = part.split("<|sid_end|>")[0].strip()
                if sid:
                    return sid
            elif part.strip():
                return part.strip()
    if "<|sid_end|>" in generation:
        generation = generation.split("<|sid_end|>")[0]
    return generation.strip() if generation.strip() else None


def compute_pass_at_k(predicted: List[Optional[str]], gt_ids: List[str], k: int) -> bool:
    if not predicted or not gt_ids:
        return False
    gt_set = set(gt_ids)
    return any(sid in gt_set for sid in predicted[:k] if sid is not None)


def compute_position1_pass_at_k(
    predicted: List[Optional[str]], first_gt: Optional[str], k: int
) -> bool:
    if not predicted or not first_gt:
        return False
    return any(sid == first_gt for sid in predicted[:k] if sid is not None)


def compute_recall_at_k(predicted: List[Optional[str]], gt_ids: List[str], k: int) -> float:
    if not predicted or not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    pred_set = set(p for p in predicted[:k] if p is not None)
    return len(pred_set & gt_set) / len(gt_set)


# ============================================================================
# Dataset path resolution (LLaMA-Factory dataset_info.json)
# ============================================================================


def resolve_data_path(
    data_path: Optional[str],
    dataset: Optional[str],
    dataset_dir: Optional[str],
) -> str:
    """优先用 ``--data_path``；否则从 dataset_info.json 解析注册名。

    LLaMA-Factory 风格：
        dataset_dir/dataset_info.json
          { "video_rec_eval": {"file_name": "/abs/path/to.jsonl", "columns": {...}} }
    """
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"--data_path 不存在: {data_path}")
        return data_path

    if not dataset:
        raise ValueError("必须指定 --data_path 或 --dataset (配合 --dataset_dir)")

    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json 不存在: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if dataset not in info:
        raise KeyError(
            f"dataset_info.json 中没有名为 '{dataset}' 的条目，可用项：{list(info.keys())[:20]}..."
        )

    entry = info[dataset]
    file_name = entry.get("file_name")
    if not file_name:
        raise ValueError(f"dataset_info.json[{dataset}] 缺少 'file_name' 字段")

    # 绝对路径直接用，相对路径相对 dataset_dir
    if not os.path.isabs(file_name):
        file_name = os.path.join(dataset_dir, file_name)
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"dataset_info.json[{dataset}].file_name 指向的文件不存在: {file_name}"
        )
    print(f"[dataset] '{dataset}' → {file_name}")
    return file_name


def resolve_output_dir(output_dir: Optional[str], model_name_or_path: str) -> str:
    """输出目录留空时，按模型路径最后两级自动生成。"""
    if output_dir:
        return output_dir

    norm_path = os.path.normpath(model_name_or_path.rstrip("/"))
    parts = [p for p in norm_path.split(os.sep) if p]
    if len(parts) >= 2:
        run_name = f"{parts[-2]}_{parts[-1]}"
    elif parts:
        run_name = parts[-1]
    else:
        run_name = "unknown_model"
    return os.path.join(DEFAULT_RESULTS_DIR, run_name)


# ============================================================================
# Data loading (Alpaca-style jsonl)
# ============================================================================


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[warn] {path}:{ln_no} json decode failed: {e}")


def _alpaca_to_messages(record: dict) -> Optional[List[dict]]:
    """把一条 Alpaca 记录 (instruction/input) 转成 chat messages。"""
    instruction = (record.get("instruction") or "").strip()
    user_input = (record.get("input") or "").strip()

    messages: List[dict] = []
    if instruction and user_input:
        messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": user_input})
    elif user_input:
        messages.append({"role": "user", "content": user_input})
    elif instruction:
        messages.append({"role": "user", "content": instruction})
    else:
        return None
    return messages


def load_samples(
    data_path: str,
    tokenizer,
    prompt_token: str,
    sample_size: Optional[int],
    enable_thinking: bool,
) -> Dict[str, Dict[str, Any]]:
    samples: Dict[str, Dict[str, Any]] = {}
    n_total = 0
    n_skipped = 0
    for idx, record in enumerate(_iter_jsonl(data_path)):
        if sample_size is not None and len(samples) >= sample_size:
            break
        n_total += 1
        sample_id = str(idx)

        messages = _alpaca_to_messages(record)
        if not messages:
            n_skipped += 1
            continue

        answer = record.get("output")
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            n_skipped += 1
            continue

        try:
            chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking:
                chat_kwargs["enable_thinking"] = True
            prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        except Exception as e:
            print(f"[skip] sample {sample_id}, chat template failed: {e}")
            n_skipped += 1
            continue

        if prompt_token:
            prompt = prompt + prompt_token

        samples[sample_id] = {
            "prompt": prompt,
            "ground_truth": str(answer).strip(),
            "metadata": {
                "source": record.get("source"),
                "instruction": record.get("instruction"),
            },
        }

    if n_skipped:
        print(f"[load_samples] skipped {n_skipped} samples")
    print(f"[load_samples] loaded {len(samples)} / {n_total} samples from {data_path}")
    return samples


# ============================================================================
# vLLM engine
# ============================================================================


def build_llm(args: argparse.Namespace):
    from vllm import LLM

    llm_kwargs: Dict[str, Any] = dict(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_logprobs=args.max_logprobs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
    return LLM(**llm_kwargs)


# ============================================================================
# Codebook (trie) & constrained beam search
# ============================================================================


def load_codebook(path: str) -> Dict[str, Any]:
    print(f"[load_codebook] {path}")
    with open(path, "rb") as f:
        cb = pickle.load(f)
    s = cb.get("stats", {})
    print(f"  unique <s_a>     : {len(cb['a_token_ids'])}")
    print(f"  unique (a,b)     : {len(cb['abc_to_c_token_ids'])}")
    print(f"  unique SIDs      : {s.get('n_unique_sids', '?')}")
    return cb


def generate_constrained_beam_search(
    llm,
    prompts: List[str],
    codebook: Dict[str, Any],
    num_beams: int,
    max_new_tokens: int,
    batch_size: int = 64,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Subtree-constrained beam search (codebook 子树约束)。"""
    from tqdm import tqdm
    from vllm import SamplingParams

    if max_new_tokens != 3:
        print(
            f"[warn] constrained beam 当前实现要求 max_new_tokens=3，收到 {max_new_tokens}，"
            f"强制改为 3"
        )
        max_new_tokens = 3

    a_allowed: List[int] = list(codebook["a_token_ids"])
    ab_to_b: Dict[int, List[int]] = codebook["ab_to_b_token_ids"]
    abc_to_c: Dict[Tuple[int, int], List[int]] = codebook["abc_to_c_token_ids"]

    tokenizer = llm.get_tokenizer()

    def _decode_one(tid: int) -> str:
        return tokenizer.decode([tid], skip_special_tokens=False)

    print(f"    Tokenizing {len(prompts)} prompts (for prompt-length tracking) ...")
    prompt_lengths: List[int] = []
    for p in tqdm(prompts, desc="    Tokenizing", unit="prompt"):
        prompt_lengths.append(len(tokenizer.encode(p, add_special_tokens=True)))

    final_beams: List[List[List[int]]] = [[] for _ in range(len(prompts))]
    final_logp: List[List[float]] = [[] for _ in range(len(prompts))]

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    print(
        f"    [constrained] running {total_batches} batches "
        f"(batch_size={batch_size}, beam={num_beams}) ..."
    )
    pbar = tqdm(total=len(prompts), desc="    Generating", unit="sample")

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        B = len(batch_prompts)

        # ---------- step 0 ----------
        sp0 = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            allowed_token_ids=a_allowed,
            logprobs=num_beams,
        )
        out0 = llm.generate(batch_prompts, sp0, use_tqdm=False)

        beams: List[List[Tuple[List[int], float]]] = []
        for o in out0:
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            top = sorted(lp_dict.items(), key=lambda kv: -kv[1].logprob)[:num_beams]
            beams.append([([tid], lp.logprob) for tid, lp in top])

        # ---------- step 1 ----------
        step1_prompts: List[str] = []
        step1_sps: List[SamplingParams] = []
        step1_owner: List[Tuple[int, int]] = []
        for i, beam_list in enumerate(beams):
            for j, (toks, _lp) in enumerate(beam_list):
                a_tid = toks[0]
                allowed_b = ab_to_b.get(a_tid) or a_allowed[:1]
                step1_prompts.append(batch_prompts[i] + _decode_one(a_tid))
                step1_sps.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        allowed_token_ids=allowed_b,
                        logprobs=min(num_beams, len(allowed_b)),
                    )
                )
                step1_owner.append((i, j))

        out1 = llm.generate(step1_prompts, step1_sps, use_tqdm=False) if step1_prompts else []

        new_beams: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
        for k, o in enumerate(out1):
            i, j = step1_owner[k]
            parent_toks, parent_lp = beams[i][j]
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            for b_tid, lp in lp_dict.items():
                new_beams[i].append((parent_toks + [b_tid], parent_lp + lp.logprob))
        for i in range(B):
            new_beams[i].sort(key=lambda x: -x[1])
            new_beams[i] = new_beams[i][:num_beams]
        beams = new_beams

        # ---------- step 2 ----------
        step2_prompts: List[str] = []
        step2_sps: List[SamplingParams] = []
        step2_owner: List[Tuple[int, int]] = []
        for i, beam_list in enumerate(beams):
            for j, (toks, _lp) in enumerate(beam_list):
                a_tid, b_tid = toks[0], toks[1]
                allowed_c = abc_to_c.get((a_tid, b_tid)) or a_allowed[:1]
                step2_prompts.append(
                    batch_prompts[i] + _decode_one(a_tid) + _decode_one(b_tid)
                )
                step2_sps.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        allowed_token_ids=allowed_c,
                        logprobs=min(num_beams, len(allowed_c)),
                    )
                )
                step2_owner.append((i, j))

        out2 = llm.generate(step2_prompts, step2_sps, use_tqdm=False) if step2_prompts else []

        new_beams2: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
        for k, o in enumerate(out2):
            i, j = step2_owner[k]
            parent_toks, parent_lp = beams[i][j]
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            for c_tid, lp in lp_dict.items():
                new_beams2[i].append((parent_toks + [c_tid], parent_lp + lp.logprob))
        for i in range(B):
            new_beams2[i].sort(key=lambda x: -x[1])
            new_beams2[i] = new_beams2[i][:num_beams]

        for i, beam_list in enumerate(new_beams2):
            final_beams[batch_start + i] = [toks for toks, _ in beam_list]
            final_logp[batch_start + i] = [lp for _, lp in beam_list]
        pbar.update(B)

    pbar.close()

    results: List[List[str]] = []
    for beams in final_beams:
        gens: List[str] = []
        for toks in beams:
            gens.append(tokenizer.decode(toks, skip_special_tokens=False))
        results.append(gens)

    return results, final_logp


# ============================================================================
# SID legal-candidate constraint (与训练 sid_constraint 完全同源的 .npz 表)
# ============================================================================


def load_sid_constraints(path: str, s3_mode: str) -> Dict[str, Any]:
    """加载训练用的 SID 合法候选约束表（scripts/build_sid_constraints.py 产出的 .npz）。

    与 ``src/llamafactory/train/sid_constraint.py`` 中 ``SidConstraintMasker`` 读取的
    是同一个文件、同一套布局，从而保证「训练时约束什么、评测时就只生成什么」：

      * s1（第一位）合法集 = ``valid_s1``
      * s2（第二位）合法集 = ``a2b[s1]``                （依赖已生成的 s1）
      * s3（第三位）合法集 =
            - ``s3_mode = valid_s3`` → 全局 ``valid_s3``  （推荐，与训练默认一致）
            - ``s3_mode = csr/ab2c`` → ``ab2c[(s1, s2)]`` （严格 per-prefix）

    表中存的是 ``[0, slot_size)`` 的「局部 code」，这里统一加上各 slot 的 base 还原成
    真实 vocab token id，供 vLLM 的 ``allowed_token_ids`` 直接使用。
    """
    s3_mode = str(s3_mode).lower()
    if s3_mode == "ab2c":  # 训练 csr 配置里写的是 ab2c，等价于 csr
        s3_mode = "csr"
    if s3_mode not in ("csr", "valid_s3"):
        raise ValueError(
            f"--sid_constraint_s3_mode 必须是 'valid_s3' 或 'csr'/'ab2c'，收到 {s3_mode!r}"
        )

    print(f"[load_sid_constraints] {path} (s3_mode={s3_mode})")
    data = np.load(path, allow_pickle=False)
    meta: Dict[str, Any] = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"]))
        except Exception:  # noqa: BLE001
            meta = {}

    slot_size = int(meta.get("slot_size", 8192))
    a_base = int(meta.get("a_base", 151669))
    b_base = int(meta.get("b_base", 159861))
    c_base = int(meta.get("c_base", 168053))

    valid_s1 = np.asarray(data["valid_s1"], dtype=np.int64)
    valid_s3 = np.asarray(data["valid_s3"], dtype=np.int64)
    a2b_indptr = np.asarray(data["a2b_indptr"], dtype=np.int64)
    a2b_indices = np.asarray(data["a2b_indices"], dtype=np.int64)
    ab2c_keys = np.asarray(data["ab2c_keys"], dtype=np.int64)
    ab2c_indptr = np.asarray(data["ab2c_indptr"], dtype=np.int64)
    ab2c_indices = np.asarray(data["ab2c_indices"], dtype=np.int64)

    # s1 token ids
    a_token_ids: List[int] = [a_base + int(c) for c in valid_s1.tolist()]

    # s2 token ids，按 s1 token id 索引
    ab_to_b: Dict[int, List[int]] = {}
    for code in valid_s1.tolist():
        s, e = int(a2b_indptr[code]), int(a2b_indptr[code + 1])
        ab_to_b[a_base + int(code)] = [b_base + int(x) for x in a2b_indices[s:e].tolist()]

    # s3 token ids
    c_global: List[int] = [c_base + int(c) for c in valid_s3.tolist()]
    abc_to_c: Dict[Tuple[int, int], List[int]] = {}
    if s3_mode == "csr":
        for r in range(int(ab2c_keys.shape[0])):
            key = int(ab2c_keys[r])
            s1, s2 = divmod(key, slot_size)
            s, e = int(ab2c_indptr[r]), int(ab2c_indptr[r + 1])
            abc_to_c[(a_base + s1, b_base + s2)] = [
                c_base + int(x) for x in ab2c_indices[s:e].tolist()
            ]

    print(
        f"  slot_size={slot_size} a_base={a_base} b_base={b_base} c_base={c_base}\n"
        f"  valid_s1={len(a_token_ids)}  a2b_keys={len(ab_to_b)}  "
        f"valid_s3={len(c_global)}  ab2c_prefixes={len(abc_to_c)}"
    )
    return {
        "s3_mode": s3_mode,
        "a_token_ids": a_token_ids,
        "ab_to_b": ab_to_b,
        "c_global": c_global,
        "abc_to_c": abc_to_c,
    }


def generate_sid_constrained_beam_search(
    llm,
    prompts: List[str],
    constraints: Dict[str, Any],
    num_beams: int,
    max_new_tokens: int,
    batch_size: int = 64,
) -> Tuple[List[List[str]], List[List[float]]]:
    """基于训练同源 .npz 约束表的 step-by-step beam search。

    每一位 SID 只在「训练时认定的合法候选」里搜索：s1∈valid_s1、s2∈a2b[s1]、
    s3∈valid_s3（或 ab2c[(s1,s2)]）。逻辑与 ``generate_constrained_beam_search``
    一致，区别仅在候选表来源与 s3 的处理方式。
    """
    from tqdm import tqdm
    from vllm import SamplingParams

    if max_new_tokens != 3:
        print(
            f"[warn] sid-constrained beam 要求 max_new_tokens=3，收到 {max_new_tokens}，强制改为 3"
        )
        max_new_tokens = 3

    s3_mode: str = constraints["s3_mode"]
    a_allowed: List[int] = list(constraints["a_token_ids"])
    ab_to_b: Dict[int, List[int]] = constraints["ab_to_b"]
    c_global: List[int] = list(constraints["c_global"])
    abc_to_c: Dict[Tuple[int, int], List[int]] = constraints["abc_to_c"]

    if not a_allowed:
        raise ValueError("约束表里 valid_s1 为空，无法进行约束解码")

    # 硬约束所需的合法 token 集合（vLLM 的 allowed_token_ids 不能保证过滤掉
    # logprobs 里返回的非法 token，因此扩 beam 时必须在 python 侧再过滤一遍）。
    a_allowed_set = set(a_allowed)
    c_global_set = set(c_global)

    def _allowed_c_set(a_tid: int, b_tid: int) -> set:
        if s3_mode == "valid_s3":
            return c_global_set
        cand = abc_to_c.get((a_tid, b_tid))
        if cand:
            return set(cand)
        return c_global_set or set(a_allowed[:1])

    tokenizer = llm.get_tokenizer()

    def _decode_one(tid: int) -> str:
        return tokenizer.decode([tid], skip_special_tokens=False)

    def _allowed_c(a_tid: int, b_tid: int) -> List[int]:
        if s3_mode == "valid_s3":
            return c_global
        return abc_to_c.get((a_tid, b_tid)) or c_global or a_allowed[:1]

    print(f"    Tokenizing {len(prompts)} prompts (for prompt-length tracking) ...")
    for p in tqdm(prompts, desc="    Tokenizing", unit="prompt"):
        tokenizer.encode(p, add_special_tokens=True)

    final_beams: List[List[List[int]]] = [[] for _ in range(len(prompts))]
    final_logp: List[List[float]] = [[] for _ in range(len(prompts))]
    illegal_dropped = 0  # 被合法集合过滤掉的非法候选总数（用于诊断）

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    print(
        f"    [sid-constrained:{s3_mode}] running {total_batches} batches "
        f"(batch_size={batch_size}, beam={num_beams}) ..."
    )
    pbar = tqdm(total=len(prompts), desc="    Generating", unit="sample")

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        B = len(batch_prompts)

        # ---------- step 0 : s1 ∈ valid_s1 ----------
        sp0 = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            allowed_token_ids=a_allowed,
            logprobs=min(num_beams, len(a_allowed)),
        )
        out0 = llm.generate(batch_prompts, sp0, use_tqdm=False)

        beams: List[List[Tuple[List[int], float]]] = []
        for o in out0:
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            cand = [
                (tid, lp.logprob) for tid, lp in lp_dict.items() if tid in a_allowed_set
            ]
            illegal_dropped += len(lp_dict) - len(cand)
            cand.sort(key=lambda kv: -kv[1])
            beams.append([([tid], lg) for tid, lg in cand[:num_beams]])

        # ---------- step 1 : s2 ∈ a2b[s1] ----------
        step1_prompts: List[str] = []
        step1_sps: List[SamplingParams] = []
        step1_owner: List[Tuple[int, int]] = []
        step1_allowed: List[set] = []
        for i, beam_list in enumerate(beams):
            for j, (toks, _lp) in enumerate(beam_list):
                a_tid = toks[0]
                allowed_b = ab_to_b.get(a_tid) or a_allowed[:1]
                step1_prompts.append(batch_prompts[i] + _decode_one(a_tid))
                step1_sps.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        allowed_token_ids=allowed_b,
                        logprobs=min(num_beams, len(allowed_b)),
                    )
                )
                step1_owner.append((i, j))
                step1_allowed.append(set(allowed_b))

        out1 = llm.generate(step1_prompts, step1_sps, use_tqdm=False) if step1_prompts else []

        new_beams: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
        for k, o in enumerate(out1):
            i, j = step1_owner[k]
            allowed_set = step1_allowed[k]
            parent_toks, parent_lp = beams[i][j]
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            for b_tid, lp in lp_dict.items():
                if b_tid not in allowed_set:
                    illegal_dropped += 1
                    continue
                new_beams[i].append((parent_toks + [b_tid], parent_lp + lp.logprob))
        for i in range(B):
            new_beams[i].sort(key=lambda x: -x[1])
            new_beams[i] = new_beams[i][:num_beams]
        beams = new_beams

        # ---------- step 2 : s3 ∈ valid_s3 (或 ab2c[(s1,s2)]) ----------
        step2_prompts: List[str] = []
        step2_sps: List[SamplingParams] = []
        step2_owner: List[Tuple[int, int]] = []
        step2_allowed: List[set] = []
        for i, beam_list in enumerate(beams):
            for j, (toks, _lp) in enumerate(beam_list):
                a_tid, b_tid = toks[0], toks[1]
                allowed_c_set = _allowed_c_set(a_tid, b_tid)
                allowed_c = list(allowed_c_set)
                step2_prompts.append(
                    batch_prompts[i] + _decode_one(a_tid) + _decode_one(b_tid)
                )
                step2_sps.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        allowed_token_ids=allowed_c,
                        logprobs=min(num_beams, len(allowed_c)),
                    )
                )
                step2_owner.append((i, j))
                step2_allowed.append(allowed_c_set)

        out2 = llm.generate(step2_prompts, step2_sps, use_tqdm=False) if step2_prompts else []

        new_beams2: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
        for k, o in enumerate(out2):
            i, j = step2_owner[k]
            allowed_set = step2_allowed[k]
            parent_toks, parent_lp = beams[i][j]
            lp_dict = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            for c_tid, lp in lp_dict.items():
                if c_tid not in allowed_set:
                    illegal_dropped += 1
                    continue
                new_beams2[i].append((parent_toks + [c_tid], parent_lp + lp.logprob))
        for i in range(B):
            new_beams2[i].sort(key=lambda x: -x[1])
            new_beams2[i] = new_beams2[i][:num_beams]

        for i, beam_list in enumerate(new_beams2):
            final_beams[batch_start + i] = [toks for toks, _ in beam_list]
            final_logp[batch_start + i] = [lp for _, lp in beam_list]
        pbar.update(B)

    pbar.close()
    print(
        f"    [sid-constrained:{s3_mode}] dropped {illegal_dropped} illegal "
        f"candidates returned by vLLM logprobs (hard-filtered to legal slot set)"
    )

    results: List[List[str]] = []
    for beams in final_beams:
        results.append([tokenizer.decode(toks, skip_special_tokens=False) for toks in beams])
    return results, final_logp


def generate_beam_search(
    llm,
    prompts: List[str],
    num_beams: int,
    max_new_tokens: int,
    batch_size: int = 64,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Beam Search 生成（OpenOneRec 官方对齐）。"""
    from tqdm import tqdm
    from vllm.sampling_params import BeamSearchParams

    params = BeamSearchParams(beam_width=num_beams, max_tokens=max_new_tokens)
    tokenizer = llm.get_tokenizer()

    print(f"    Tokenizing {len(prompts)} prompts...")
    prompt_lengths = []
    for text in tqdm(prompts, desc="    Tokenizing", unit="prompt"):
        prompt_lengths.append(len(tokenizer.encode(text, add_special_tokens=True)))

    results: List[List[str]] = []
    all_logprobs: List[List[float]] = []

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"    Generating with {total_batches} batches (batch_size={batch_size})...")

    pbar = tqdm(total=len(prompts), desc="    Generating", unit="sample")
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_lengths = prompt_lengths[batch_start:batch_end]

        prompt_dicts = [{"prompt": text} for text in batch_prompts]
        outputs = llm.beam_search(prompt_dicts, params)

        for idx, output in enumerate(outputs):
            prompt_length = batch_lengths[idx]
            generated_texts = [
                tokenizer.decode(seq.tokens[prompt_length:], skip_special_tokens=True)
                for seq in output.sequences
            ]
            cum_logprobs = [seq.cum_logprob for seq in output.sequences]
            results.append(generated_texts)
            all_logprobs.append(cum_logprobs)

        pbar.update(len(batch_prompts))
    pbar.close()

    return results, all_logprobs


def dedup_generations(
    generations: List[str],
    logprobs: Optional[List[float]] = None,
    mode: str = "sid",
) -> Tuple[List[str], List[float]]:
    """按指定模式去重，保留首次出现顺序。"""
    if mode == "none":
        return generations, (logprobs or [0.0] * len(generations))

    deduped_gens: List[str] = []
    deduped_logprobs: List[float] = []
    seen_keys: Set[str] = set()

    for idx, gen in enumerate(generations):
        if mode == "string":
            key = gen.strip()
        else:  # mode == "sid"
            sid = extract_id_from_generation(gen)
            # 无法提取 SID 的项不参与 SID 去重，保留原始候选，避免误删。
            if sid is None:
                deduped_gens.append(gen)
                deduped_logprobs.append(
                    logprobs[idx] if logprobs and idx < len(logprobs) else 0.0
                )
                continue
            key = sid

        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped_gens.append(gen)
        deduped_logprobs.append(logprobs[idx] if logprobs and idx < len(logprobs) else 0.0)

    return deduped_gens, deduped_logprobs


# ============================================================================
# Evaluation
# ============================================================================


def evaluate(
    samples: Dict[str, Dict[str, Any]],
    predictions: Dict[str, List[str]],
    logprobs: Dict[str, List[float]],
    k_values: List[int],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, list]]:
    pass_counts = {k: 0 for k in k_values}
    position1_pass_counts = {k: 0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}

    per_sample: Dict[str, Dict[str, Any]] = {}
    debug = {"passed_samples": [], "failed_samples": [], "no_generation_samples": []}

    total = 0
    for sid, sample in samples.items():
        gens = predictions.get(sid, [])
        gt_ids = extract_ids_from_answer(sample["ground_truth"])
        first_gt = extract_first_id_from_answer(sample["ground_truth"])

        if not gens:
            sample_metrics = {}
            for k in k_values:
                sample_metrics[f"pass@{k}"] = False
                sample_metrics[f"position1_pass@{k}"] = False
                sample_metrics[f"recall@{k}"] = 0.0
            per_sample[sid] = sample_metrics
            debug["no_generation_samples"].append(
                {"sample_id": sid, "ground_truth": sample["ground_truth"]}
            )
            total += 1
            continue

        pred_ids = [extract_id_from_generation(g) for g in gens]
        sample_metrics: Dict[str, Any] = {}

        for k in k_values:
            p = compute_pass_at_k(pred_ids, gt_ids, k)
            sample_metrics[f"pass@{k}"] = p
            pass_counts[k] += int(p)

            p1 = compute_position1_pass_at_k(pred_ids, first_gt, k)
            sample_metrics[f"position1_pass@{k}"] = p1
            position1_pass_counts[k] += int(p1)

            r = compute_recall_at_k(pred_ids, gt_ids, k)
            sample_metrics[f"recall@{k}"] = r
            recall_sums[k] += r

        per_sample[sid] = sample_metrics
        total += 1

        debug_item = {
            "sample_id": sid,
            "n_ground_truth_sids": len(gt_ids),
            "ground_truth_sids": gt_ids[:10],
            "first_ground_truth_sid": first_gt,
            "top_10_generations": pred_ids[:10],
            "pass_results": {k: sample_metrics[f"pass@{k}"] for k in k_values},
            "position1_pass_results": {k: sample_metrics[f"position1_pass@{k}"] for k in k_values},
        }
        if any(sample_metrics[f"pass@{k}"] for k in k_values):
            debug["passed_samples"].append(debug_item)
        else:
            debug["failed_samples"].append(debug_item)

    metrics: Dict[str, Any] = {"total_samples": total}
    for k in k_values:
        metrics[f"pass@{k}"] = pass_counts[k] / total if total else 0.0
        metrics[f"position1_pass@{k}"] = position1_pass_counts[k] / total if total else 0.0
        metrics[f"recall@{k}"] = recall_sums[k] / total if total else 0.0

    return metrics, per_sample, debug


# ============================================================================
# Result IO
# ============================================================================


def save_generation_file(
    output_dir: str,
    samples: Dict[str, Dict[str, Any]],
    predictions: Dict[str, List[str]],
    logprobs_dict: Dict[str, List[float]],
    metrics: Dict[str, Any],
    per_sample: Dict[str, Dict[str, Any]],
    elapsed_seconds: float,
    model_name: str,
):
    path = os.path.join(output_dir, "test_generated.json")
    out_samples: Dict[str, Any] = {}
    for sid, sample in samples.items():
        item: Dict[str, Any] = {
            "prompt": sample["prompt"],
            "ground_truth": sample["ground_truth"],
            "generations": predictions.get(sid, []),
            "logprobs": logprobs_dict.get(sid, []),
            "metadata": sample.get("metadata", {}),
        }
        item.update(per_sample.get(sid, {}))
        out_samples[sid] = item

    data = {
        "model_name": model_name,
        "task_name": "video",
        "split": "test",
        "total_time": elapsed_seconds,
        "avg_time_per_sample": elapsed_seconds / len(samples) if samples else 0,
        "samples": out_samples,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save] {path}")


def save_debug_file(
    output_dir: str,
    debug: Dict[str, list],
    metrics: Dict[str, Any],
    keep: int = 100,
):
    debug_path = os.path.join(output_dir, "debug.json")
    out = {
        "metrics": metrics,
        "statistics": {
            "total_samples": metrics.get("total_samples", 0),
            "passed_samples_count": len(debug["passed_samples"]),
            "failed_samples_count": len(debug["failed_samples"]),
            "no_generation_samples_count": len(debug["no_generation_samples"]),
        },
        "passed_samples": debug["passed_samples"][:keep],
        "failed_samples": debug["failed_samples"][:keep],
        "no_generation_samples": debug["no_generation_samples"][:keep],
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[save] {debug_path}")


# ============================================================================
# Argument parsing (YAML + CLI)
# ============================================================================


def _merge_yaml_into_args(yaml_path: str, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """读 YAML 作为默认值，CLI 参数覆盖之。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 用 YAML 设置 parser 的 default 值，然后正常 parse CLI
    parser.set_defaults(**cfg)
    return parser.parse_args(_strip_config_arg(sys.argv[1:]))


def _strip_config_arg(argv: List[str]) -> List[str]:
    """去掉 --config xxx 这一对，避免再次解析时报错。"""
    out: List[str] = []
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok == "--config":
            skip = True
            continue
        if tok.startswith("--config="):
            continue
        out.append(tok)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video Recommendation Evaluation (LLaMA-Factory integrated, OpenOneRec aligned)"
    )

    # --- 入口：YAML 配置 ---
    p.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径（参数名同 LLaMA-Factory 训练 YAML 风格）",
    )

    # --- LLaMA-Factory 对齐字段 ---
    p.add_argument(
        "--model_name_or_path",
        default=None,
        help="HF 模型目录（与 LLaMA-Factory 训练 YAML 中的字段同名）",
    )
    p.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help=(
            "LLaMA-Factory 训练时所用 chat 模板名（仅作为标记 / 校验用途）。"
            "实际渲染走 tokenizer 自带 chat_template；如需覆盖请用 --chat_template_file"
        ),
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="dataset_info.json 中的注册名（如 'video_rec_eval'）",
    )
    p.add_argument(
        "--dataset_dir",
        default=DEFAULT_DATASET_DIR,
        help="dataset_info.json 所在目录",
    )

    # --- 兜底：直接传 jsonl ---
    p.add_argument(
        "--data_path",
        default=None,
        help="直接传 jsonl 文件路径（与 --dataset 二选一，优先于 --dataset）",
    )

    # --- Chat template 覆盖 ---
    p.add_argument(
        "--chat_template_file",
        default=None,
        help="可选：jinja2 文件路径，用来覆盖 tokenizer 自带的 chat_template（用于对比外部 ckpt）",
    )

    # --- 输出与采样 ---
    p.add_argument("--output_dir", default=None, required=False, help="评估输出目录")
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="子采样数；不传或 <=0 = 全量",
    )

    # --- Beam search ---
    p.add_argument("--num_beams", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=3)
    p.add_argument(
        "--k_values",
        default="1,32",
        help="逗号分隔的 k 列表；YAML 中可写 list 或 string",
    )

    # --- 其他 ---
    p.add_argument("--prompt_token", default="<|sid_begin|>")
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument(
        "--dedup_mode",
        choices=["none", "string", "sid"],
        default="sid",
        help="生成候选去重模式：none/string/sid",
    )

    # --- 约束解码 ---
    p.add_argument(
        "--constrained",
        choices=["none", "subtree", "sid"],
        default="none",
        help=(
            "约束解码模式：none=不约束；subtree=codebook 子树约束（需 --codebook_path）；"
            "sid=训练同源 .npz 合法候选约束（需 --sid_constraint_path）"
        ),
    )
    p.add_argument(
        "--codebook_path",
        default="",
        help="codebook pickle 路径（仅 --constrained subtree 时需要）",
    )
    p.add_argument(
        "--sid_constraint_path",
        default="",
        help=(
            "SID 合法候选约束 .npz 路径（仅 --constrained sid 时需要），"
            "与训练 sid_constraint_path 同一个文件"
        ),
    )
    p.add_argument(
        "--sid_constraint_s3_mode",
        default="valid_s3",
        help="第三位约束模式：valid_s3（推荐，全局集）或 csr/ab2c（严格 per-prefix）",
    )

    # --- 性能 ---
    p.add_argument("--batch_size", type=int, default=128)

    # --- vLLM ---
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_logprobs", type=int, default=384)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--enable_prefix_caching", action="store_true", default=True)
    p.add_argument("--enable_chunked_prefill", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=None)

    # 先 peek 一下 --config
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        print(f"[config] loading YAML config from: {pre_args.config}")
        args = _merge_yaml_into_args(pre_args.config, p)
    else:
        args = p.parse_args()

    # k_values 兼容 list / str
    if isinstance(args.k_values, list):
        args.k_values = [int(x) for x in args.k_values]
    elif isinstance(args.k_values, str):
        args.k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    elif isinstance(args.k_values, int):
        args.k_values = [args.k_values]

    if args.sample_size is not None and args.sample_size <= 0:
        args.sample_size = None

    # 必填校验
    if not args.model_name_or_path:
        p.error("--model_name_or_path 必填（YAML 或 CLI）")
    if not args.data_path and not args.dataset:
        p.error("必须指定 --data_path 或 --dataset")

    args.output_dir = resolve_output_dir(args.output_dir, args.model_name_or_path)
    print(f"[output_dir] {args.output_dir}")

    return args


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print("Video-Rec Evaluation (LLaMA-Factory integrated)")
    print("=" * 78)
    print(f"  model_name_or_path : {args.model_name_or_path}")
    print(f"  template (LF)      : {args.template}")
    if args.chat_template_file:
        print(f"  chat_template_file : {args.chat_template_file} (will override)")
    if args.dataset:
        print(f"  dataset            : {args.dataset}")
        print(f"  dataset_dir        : {args.dataset_dir}")
    if args.data_path:
        print(f"  data_path          : {args.data_path}")
    print(f"  output_dir         : {args.output_dir}")
    print(f"  sample_size        : {args.sample_size}")
    print(f"  --- Beam Search ---")
    print(f"  num_beams          : {args.num_beams}")
    print(f"  max_new_tokens     : {args.max_new_tokens}")
    print(f"  k_values           : {args.k_values}")
    print(f"  dedup_mode         : {args.dedup_mode}")
    print(f"  constrained        : {args.constrained}")
    if args.constrained == "subtree":
        print(f"  codebook_path      : {args.codebook_path}")
    if args.constrained == "sid":
        print(f"  sid_constraint_path: {args.sid_constraint_path}")
        print(f"  sid_constraint_s3  : {args.sid_constraint_s3_mode}")
    print(f"  --- vLLM ---")
    print(f"  tensor_parallel    : {args.tensor_parallel_size}")
    print(f"  gpu_mem_util       : {args.gpu_memory_utilization}")
    print(f"  dtype              : {args.dtype}")
    print("=" * 78)

    # 1) Resolve data path
    data_path = resolve_data_path(args.data_path, args.dataset, args.dataset_dir)

    # 2) Tokenizer + chat template
    print("\n[1/4] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if args.chat_template_file:
        if not os.path.exists(args.chat_template_file):
            raise FileNotFoundError(f"--chat_template_file 不存在: {args.chat_template_file}")
        with open(args.chat_template_file, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
        print(f"  → overridden by {args.chat_template_file}")
    else:
        # 走 tokenizer 自带 chat_template（训练时 LLaMA-Factory 已经写入）
        print(f"  → using tokenizer built-in chat_template (template: {args.template})")

    # 3) Load samples
    print("\n[2/4] Loading data ...")
    samples = load_samples(
        data_path=data_path,
        tokenizer=tokenizer,
        prompt_token=args.prompt_token,
        sample_size=args.sample_size,
        enable_thinking=args.enable_thinking,
    )
    if not samples:
        raise RuntimeError("No samples loaded; check the data path / sample_size.")

    sample_ids = list(samples.keys())
    prompts = [samples[sid]["prompt"] for sid in sample_ids]

    # 4) Build vLLM
    print(f"\n[3/4] Building vLLM engine (TP={args.tensor_parallel_size}) ...")
    llm = build_llm(args)

    model_name = os.path.basename(args.model_name_or_path.rstrip("/"))

    # 5) Generate
    print(f"\n[4/4] Generation & evaluation ...")
    codebook = None
    sid_constraints = None
    if args.constrained == "subtree":
        if not args.codebook_path:
            raise ValueError("--constrained subtree 需要传 --codebook_path")
        codebook = load_codebook(args.codebook_path)
        print(
            f"  >>> CONSTRAINED Beam Search (subtree mask, beam={args.num_beams}, "
            f"max_new={args.max_new_tokens})"
        )
    elif args.constrained == "sid":
        if not args.sid_constraint_path:
            raise ValueError("--constrained sid 需要传 --sid_constraint_path")
        if not os.path.exists(args.sid_constraint_path):
            raise FileNotFoundError(
                f"--sid_constraint_path 不存在: {args.sid_constraint_path}"
            )
        sid_constraints = load_sid_constraints(
            args.sid_constraint_path, args.sid_constraint_s3_mode
        )
        print(
            f"  >>> SID-CONSTRAINED Beam Search (npz legal candidates, "
            f"s3_mode={sid_constraints['s3_mode']}, beam={args.num_beams}, "
            f"max_new={args.max_new_tokens})"
        )
    else:
        print(
            f"  >>> Beam Search (beam={args.num_beams}, max_new={args.max_new_tokens})"
        )

    t0 = time.time()
    if args.constrained == "subtree":
        gens, logprobs_list = generate_constrained_beam_search(
            llm,
            prompts,
            codebook=codebook,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    elif args.constrained == "sid":
        gens, logprobs_list = generate_sid_constrained_beam_search(
            llm,
            prompts,
            constraints=sid_constraints,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    else:
        gens, logprobs_list = generate_beam_search(
            llm,
            prompts,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.2f}s ({elapsed / max(len(prompts), 1):.4f}s/sample)")

    dedup_total = 0
    min_required = max(args.k_values) if args.k_values else 0
    insufficient_count = 0
    shortest_len = None
    for i in range(len(gens)):
        before = len(gens[i])
        gens[i], logprobs_list[i] = dedup_generations(
            gens[i], logprobs_list[i], mode=args.dedup_mode
        )
        dedup_total += max(0, before - len(gens[i]))
        if len(gens[i]) < min_required:
            insufficient_count += 1
            shortest_len = (
                len(gens[i])
                if shortest_len is None
                else min(shortest_len, len(gens[i]))
            )
    print(
        f"    dedup mode={args.dedup_mode} removed {dedup_total} duplicated candidates"
    )
    if min_required > 0 and insufficient_count > 0:
        print(
            f"    [warn] {insufficient_count}/{len(gens)} samples have <{min_required} candidates "
            f"after dedup (shortest={shortest_len})"
        )
    elif min_required > 0:
        print(
            f"    [ok] all {len(gens)} samples have >={min_required} candidates after dedup"
        )

    predictions = {sid: g for sid, g in zip(sample_ids, gens)}
    logprobs_dict = {sid: lp for sid, lp in zip(sample_ids, logprobs_list)}

    metrics, per_sample, debug = evaluate(
        samples, predictions, logprobs_dict, k_values=args.k_values
    )

    save_generation_file(
        args.output_dir,
        samples,
        predictions,
        logprobs_dict,
        metrics,
        per_sample,
        elapsed,
        model_name,
    )
    save_debug_file(args.output_dir, debug, metrics)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "model_name": model_name,
        "template": args.template,
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "data_path": data_path,
        "num_samples": len(samples),
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "k_values": args.k_values,
        "dedup_mode": args.dedup_mode,
        "prompt_token": args.prompt_token,
        "enable_thinking": args.enable_thinking,
        "constrained": args.constrained,
        "codebook_path": args.codebook_path if args.constrained == "subtree" else None,
        "sid_constraint_path": (
            args.sid_constraint_path if args.constrained == "sid" else None
        ),
        "sid_constraint_s3_mode": (
            args.sid_constraint_s3_mode if args.constrained == "sid" else None
        ),
        "tensor_parallel_size": args.tensor_parallel_size,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"Summary written to: {summary_path}")
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 78)


if __name__ == "__main__":
    main()
