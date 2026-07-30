#!/usr/bin/env python3
"""为 Qwen3-8B 构建三种扩展词表模型。

默认依次生成：
1. Qwen3-8B + SID token（随机初始化）
2. Qwen3-8B + MovieLens token（描述文本语义初始化）
3. Qwen3-8B + Amazon Movies & TV token（描述文本语义初始化）

示例：
    python scripts/build_qwen3_8b_token_variants.py
    --targets 可以设置生成什么模型：sid/movielens/amazon
    --semantic-device 可以设置使用什么设备：cuda/cpu
"""

import argparse
import gc
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


WORKSPACE = Path(
    "/apdcephfs_cq11/share_303717182/bobjxzhang"
)
DEFAULT_BASE_MODEL = WORKSPACE / "Qwen/Qwen3-8B"
DEFAULT_SID_TOKENS = WORKSPACE / "Qwen/OneRec-8B/added_tokens.json"
DEFAULT_MOVIELENS_JSON = (
    WORKSPACE / "data/user_llm/movielens20m/id_descriptions.json"
)
DEFAULT_AMAZON_JSON = (
    WORKSPACE / "data/user_llm/amazon_movies_tv/id_descriptions.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE / "Qwen"

SEMANTIC_DATASETS = {
    "movielens": {
        "default_json": DEFAULT_MOVIELENS_JSON,
        "output_name": "Qwen3-8B-movielens20m-semantic-init",
        "templates": {
            "movie": "The movie is {description}",
            "genre": "The movie genre is {description}",
            "rating": "{description}",
        },
    },
    "amazon": {
        "default_json": DEFAULT_AMAZON_JSON,
        "output_name": "Qwen3-8B-amazon-movies-tv-semantic-init",
        "templates": {
            "item": "The Amazon Movies and TV item is {description}",
            "category": "The Amazon item category is {description}",
            "rating": "{description}",
        },
    },
}


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def check_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录非空：{output_dir}。如需覆盖，请添加 --overwrite。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer(base_model: Path, dtype: torch.dtype):
    print(f"加载 tokenizer：{base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), trust_remote_code=True
    )
    print(f"加载模型：{base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def filter_new_tokens(
    tokenizer, token_phrase_pairs: Sequence[Tuple[str, str]]
) -> Tuple[List[str], List[str]]:
    """去重并过滤 tokenizer 中已经存在的 token。"""
    existing = tokenizer.get_vocab()
    seen = set()
    tokens: List[str] = []
    phrases: List[str] = []
    for token, phrase in token_phrase_pairs:
        if token in seen:
            raise ValueError(f"输入数据中存在重复 token：{token}")
        seen.add(token)
        if token not in existing:
            tokens.append(token)
            phrases.append(phrase)
    return tokens, phrases


def load_sid_tokens(path: Path) -> List[str]:
    """读取 SID token，并保持 OneRec 中的原始 token id 顺序。"""
    with path.open("r", encoding="utf-8") as file:
        token_to_id = json.load(file)

    sid_pattern = re.compile(r"^<s_[abc]_\d+>$")
    sid_tokens = [
        (token, token_id)
        for token, token_id in token_to_id.items()
        if sid_pattern.fullmatch(token)
        or token in {"<|sid_begin|>", "<|sid_end|>"}
    ]
    if not sid_tokens:
        raise ValueError(f"没有在 {path} 中找到 SID token")
    sid_tokens.sort(key=lambda item: item[1])
    return [token for token, _ in sid_tokens]


def load_semantic_pairs(
    path: Path, templates: Dict[str, str]
) -> List[Tuple[str, str]]:
    """从 id_descriptions.json 读取 (token, 语义描述) 对。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    pairs: List[Tuple[str, str]] = []
    for section, items in data.items():
        if not isinstance(items, dict):
            raise TypeError(f"{path} 的 section {section!r} 不是字典")
        template = templates.get(section, "{description}")
        for token, description in items.items():
            if not isinstance(description, str):
                description = json.dumps(description, ensure_ascii=False)
            phrase = template.format(description=description.strip())
            pairs.append((token, phrase))
    if not pairs:
        raise ValueError(f"{path} 中没有 token 描述")
    return pairs


@torch.no_grad()
def compute_semantic_rows(
    tokenizer,
    base_embedding: torch.Tensor,
    phrases: Sequence[str],
    batch_size: int,
    device: str,
    mean_center: bool,
) -> torch.Tensor:
    """用描述子词的基础 embedding 均值初始化每个新 token。"""
    try:
        lookup_embedding = base_embedding.detach().to(device)
    except RuntimeError as error:
        if device == "cpu":
            raise
        print(f"警告：无法把 embedding 放到 {device}（{error}），改用 CPU")
        device = "cpu"
        lookup_embedding = base_embedding.detach()

    rows: List[torch.Tensor] = []
    for start in tqdm(
        range(0, len(phrases), batch_size),
        desc="计算语义初始化",
        unit="batch",
    ):
        batch = phrases[start : start + batch_size]
        encoded = tokenizer(
            list(batch),
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        for phrase, token_ids in zip(batch, encoded):
            if not token_ids:
                raise ValueError(f"描述被编码为空：{phrase!r}")
            ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            rows.append(lookup_embedding.index_select(0, ids).mean(dim=0).cpu())

    semantic_rows = torch.stack(rows)
    if mean_center:
        semantic_rows -= semantic_rows.mean(dim=0, keepdim=True)
    del lookup_embedding
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return semantic_rows


@torch.no_grad()
def install_new_rows(
    model,
    tokenizer,
    new_tokens: Sequence[str],
    input_rows: torch.Tensor,
    output_rows: torch.Tensor,
) -> Tuple[int, int]:
    old_vocab_size = len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != old_vocab_size:
        raise ValueError(
            "tokenizer 长度与模型 input embedding 行数不一致："
            f"{old_vocab_size} != "
            f"{model.get_input_embeddings().weight.shape[0]}"
        )

    num_added = tokenizer.add_tokens(list(new_tokens), special_tokens=False)
    if num_added != len(new_tokens):
        raise RuntimeError(
            f"预计添加 {len(new_tokens)} 个 token，实际添加 {num_added} 个"
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    input_weight = model.get_input_embeddings().weight
    output_embedding = model.get_output_embeddings()
    start = old_vocab_size
    end = start + num_added
    input_weight[start:end].copy_(
        input_rows.to(device=input_weight.device, dtype=input_weight.dtype)
    )

    if (
        output_embedding is not None
        and output_embedding.weight.data_ptr() != input_weight.data_ptr()
    ):
        output_weight = output_embedding.weight
        output_weight[start:end].copy_(
            output_rows.to(
                device=output_weight.device, dtype=output_weight.dtype
            )
        )

    expected_ids = list(range(start, end))
    actual_ids = tokenizer.convert_tokens_to_ids(list(new_tokens))
    if actual_ids != expected_ids:
        raise RuntimeError("新增 token id 不是连续追加到原词表末尾")

    model.config.vocab_size = len(tokenizer)
    return old_vocab_size, num_added


def save_variant(
    model,
    tokenizer,
    output_dir: Path,
    report: Dict,
    max_shard_size: str,
) -> None:
    print(f"保存模型：{output_dir}")
    tokenizer.save_pretrained(str(output_dir))
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    with (output_dir / "token_initialization_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    shard_paths = sorted(glob.glob(str(output_dir / "model-*.safetensors")))
    if shard_paths:
        total_size = sum(os.path.getsize(path) for path in shard_paths)
        print(
            f"已保存 {len(shard_paths)} 个模型分片，"
            f"共 {total_size / 1024**3:.2f} GiB"
        )


def release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_sid_variant(args, dtype: torch.dtype) -> None:
    output_dir = args.output_root / args.sid_output_name
    check_output_dir(output_dir, args.overwrite)
    sid_tokens = load_sid_tokens(args.sid_tokens)
    model, tokenizer = load_model_and_tokenizer(args.base_model, dtype)
    pairs = [(token, "") for token in sid_tokens]
    new_tokens, _ = filter_new_tokens(tokenizer, pairs)
    if not new_tokens:
        raise ValueError("基础 tokenizer 已包含所有 SID token")

    input_embedding = model.get_input_embeddings().weight
    output_embedding = model.get_output_embeddings()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_rows = torch.randn(
        len(new_tokens),
        input_embedding.shape[1],
        dtype=torch.float32,
        generator=generator,
    ) * args.random_std
    output_rows = torch.randn(
        len(new_tokens),
        output_embedding.weight.shape[1],
        dtype=torch.float32,
        generator=generator,
    ) * args.random_std

    old_vocab_size, num_added = install_new_rows(
        model, tokenizer, new_tokens, input_rows, output_rows
    )
    report = {
        "variant": "sid",
        "base_model": str(args.base_model),
        "sid_tokens": str(args.sid_tokens),
        "initialization": "normal_random",
        "random_std": args.random_std,
        "seed": args.seed,
        "old_vocab_size": old_vocab_size,
        "num_added": num_added,
        "new_vocab_size": len(tokenizer),
    }
    save_variant(
        model, tokenizer, output_dir, report, args.max_shard_size
    )
    release_model(model, tokenizer)


def build_semantic_variant(
    target: str, args, dtype: torch.dtype
) -> None:
    config = SEMANTIC_DATASETS[target]
    json_path = (
        args.movielens_json if target == "movielens" else args.amazon_json
    )
    output_dir = args.output_root / config["output_name"]
    check_output_dir(output_dir, args.overwrite)

    pairs = load_semantic_pairs(json_path, config["templates"])
    model, tokenizer = load_model_and_tokenizer(args.base_model, dtype)
    new_tokens, phrases = filter_new_tokens(tokenizer, pairs)
    if not new_tokens:
        raise ValueError(f"基础 tokenizer 已包含 {target} 的所有 token")
    print(f"{target}：读取 {len(pairs)} 条描述，添加 {len(new_tokens)} 个 token")

    semantic_rows = compute_semantic_rows(
        tokenizer=tokenizer,
        base_embedding=model.get_input_embeddings().weight,
        phrases=phrases,
        batch_size=args.batch_size,
        device=args.semantic_device,
        mean_center=not args.no_mean_center,
    )
    old_vocab_size, num_added = install_new_rows(
        model,
        tokenizer,
        new_tokens,
        semantic_rows,
        semantic_rows,
    )
    report = {
        "variant": target,
        "base_model": str(args.base_model),
        "descriptions": str(json_path),
        "initialization": (
            "description_subtoken_mean_pool"
            + ("_with_global_mean_centering" if not args.no_mean_center else "")
        ),
        "old_vocab_size": old_vocab_size,
        "num_added": num_added,
        "new_vocab_size": len(tokenizer),
    }
    save_variant(
        model, tokenizer, output_dir, report, args.max_shard_size
    )
    del semantic_rows
    release_model(model, tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 Qwen3-8B 的 SID、MovieLens 和 Amazon 扩展模型"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["sid", "movielens", "amazon"],
        default=["sid", "movielens", "amazon"],
    )
    parser.add_argument(
        "--base-model", type=Path, default=DEFAULT_BASE_MODEL
    )
    parser.add_argument(
        "--sid-tokens", type=Path, default=DEFAULT_SID_TOKENS
    )
    parser.add_argument(
        "--movielens-json", type=Path, default=DEFAULT_MOVIELENS_JSON
    )
    parser.add_argument(
        "--amazon-json", type=Path, default=DEFAULT_AMAZON_JSON
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--sid-output-name", default="Qwen3-8B-with-sid"
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--semantic-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--random-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-mean-center",
        action="store_true",
        help="关闭语义 embedding 的全局均值中心化",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许写入已有的非空输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.dtype)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for index, target in enumerate(args.targets, start=1):
        print("=" * 80)
        print(f"[{index}/{len(args.targets)}] 构建 {target} 模型")
        print("=" * 80)
        if target == "sid":
            build_sid_variant(args, dtype)
        else:
            build_semantic_variant(target, args, dtype)

    print("全部目标模型构建完成。")


if __name__ == "__main__":
    main()
