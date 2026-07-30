#!/usr/bin/env python3
"""
Bake BehaviorProjector into embed_tokens weights so the model becomes
a standard transformer (no custom projector needed at inference time).

For all behavior token ids (>= behavior_token_start_id):
    embed_tokens.weight[i] = projector(embed_tokens.weight[i])

Usage:
    python scripts/merge_projector_into_embeddings.py \
        --model-path /apdcephfs_cq11/share_303717182/bobjxzhang/Qwen/Qwen3-8B-with-sid \
        --projector-path /apdcephfs_cq11/share_303717182/bobjxzhang/saves/onerec_seq_align_phase1_v2/checkpoint-4000/projector \
        --output-path /apdcephfs_cq11/share_303717182/bobjxzhang/Qwen/align_test \
        --behavior-token-start-id 151669
"""

import argparse
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # llm_based_user_sequence_modelling
workspace_root = os.path.dirname(project_root)  # parent of llm_based_user_sequence_modelling
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_based_user_sequence_modelling.src.llamafactory.train.seq_align.projector import BehaviorProjector
# fromllamafactory.train.seq_align.projector import BehaviorProjector


def merge(model_path: str, projector_path: str, output_path: str, behavior_token_start_id: int) -> None:
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    print(f"Loading projector from {projector_path} ...")
    projector = BehaviorProjector.from_pretrained(projector_path)
    projector = projector.to(dtype=model.dtype)
    projector.eval()

    emb = model.model.embed_tokens.weight  # [vocab_size, hidden_size]
    vocab_size = emb.shape[0]

    if behavior_token_start_id >= vocab_size:
        raise ValueError(
            f"behavior_token_start_id={behavior_token_start_id} >= vocab_size={vocab_size}"
        )

    n_behavior = vocab_size - behavior_token_start_id
    print(f"Merging projector into {n_behavior} behavior token embeddings "
          f"(ids {behavior_token_start_id} ~ {vocab_size - 1}) ...")

    with torch.no_grad():
        beh_embs = emb[behavior_token_start_id:].clone()  # [N, hidden]
        projected = projector(beh_embs)                   # [N, hidden]
        emb[behavior_token_start_id:] = projected

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="LLM checkpoint directory")
    parser.add_argument("--projector-path", required=True, help="projector directory (contains projector.pt + projector_config.json)")
    parser.add_argument("--output-path", required=True, help="where to save the merged model")
    parser.add_argument("--behavior-token-start-id", type=int, default=151675)
    args = parser.parse_args()

    merge(args.model_path, args.projector_path, args.output_path, args.behavior_token_start_id)


if __name__ == "__main__":
    main()
