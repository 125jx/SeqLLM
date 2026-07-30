#!/bin/bash

# C-Eval 评估脚本 (vLLM 加速)
# 52 个学科，约 13,948 道中文选择题

# 模型路径
MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/saves/video_rec_rl/video_rec_rl_loss/video_rec_rl_ckpt_120"
# MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/saves/onerec_sft_100k_mixed_cont/checkpoint-1106"
MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_DIR="/apdcephfs_cq11/share_303717182/bobjxzhang/tmp/eval_results/${MODEL_NAME}-ceval"
mkdir -p $OUTPUT_DIR

# vLLM 配置
TENSOR_PARALLEL=8
GPU_MEMORY_UTIL=0.9

# few-shot 数量：Qwen 官方 C-Eval 成绩是 5-shot，0-shot 会明显偏低
NUM_FEWSHOT=5

# 代理设置（如需下载数据集）
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

# 运行评估
# 对齐官方 base 成绩：纯 5-shot loglikelihood completion
# 不加 --apply_chat_template / --fewshot_as_multiturn（对比 Qwen3-8B base ~78%）
lm_eval --model vllm \
    --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL},trust_remote_code=True,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTIL},max_model_len=8192" \
    --tasks ceval-valid \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR} \
    --log_samples

echo "========================================"
echo "C-Eval 评估完成！（base 对齐：5-shot loglikelihood）"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "参考成绩 (base / 5-shot):"
echo "  - Qwen2.5-7B:          ~81%"
echo "  - Qwen3-8B:            ~78%"
echo "========================================"
