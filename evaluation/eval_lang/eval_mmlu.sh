#!/bin/bash

# MMLU 评估脚本 (vLLM 加速)
# 57 个学科，约 14,000+ 道选择题

# 模型路径
MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/video_rec_rl_grpo_multinode/video_rec_rl_ckpt_90"
# MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/saves/onerec_sft_100k_mixed_cont/checkpoint-1106"
MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR="/apdcephfs_cq11/share_303717182/bobjxzhang/tmp/eval_results/${MODEL_NAME}-mmlu"
mkdir -p $OUTPUT_DIR

# 代理设置（如需下载数据集）
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

# 运行评估
lm_eval --model vllm \
    --model_args pretrained=${MODEL_PATH},tensor_parallel_size=8,trust_remote_code=True,dtype=auto \
    --tasks mmlu \
    --batch_size auto \
    --output_path ${OUTPUT_DIR} \
    --log_samples

echo "========================================"
echo "MMLU 评估完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "参考成绩:"
echo "  - Qwen2.5-7B:          ~74%"
echo "  - Qwen2.5-7B-Instruct: ~76%"
echo "  - Qwen3-8B:            ~70%"
echo "========================================"
