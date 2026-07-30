#!/bin/bash

# ============================================
# AGIEval 评测脚本 (中文 + 英文)
# ============================================
# AGIEval 是一个针对人类认知和问题解决能力设计的基准测试
# 包含高考、法律考试、SAT、LSAT、GRE 等多种考试题目
# ============================================

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============== 配置区域 ==============
MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/video_rec_rl_loss/video_rec_rl_ckpt_120"
# MODEL_PATH="/apdcephfs_cq11/share_303717182/bobjxzhang/saves/qwen8b_pt_only_seq/checkpoint-6000"
OUTPUT_BASE="/apdcephfs_cq11/share_303717182/bobjxzhang/tmp/eval_results"

# 评测模式: "cn" (仅中文), "en" (仅英文), "all" (全部)
EVAL_MODE="cn"

# vLLM 配置
TENSOR_PARALLEL=8
GPU_MEMORY_UTIL=0.9
# ======================================

# 设置代理（如果需要下载数据）
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

# 根据模式选择任务
case $EVAL_MODE in
    "cn")
        TASKS="agieval_cn"
        OUTPUT_DIR="${OUTPUT_BASE}/agieval_cn"
        DESC="AGIEval 中文子集 (高考 + 法律 + 逻辑)"
        ;;
    "en")
        TASKS="agieval_en"
        OUTPUT_DIR="${OUTPUT_BASE}/agieval_en"
        DESC="AGIEval 英文子集 (SAT + LSAT + GRE)"
        ;;
    "all")
        TASKS="agieval"
        OUTPUT_DIR="${OUTPUT_BASE}/agieval_all"
        DESC="AGIEval 完整版 (50个子任务)"
        ;;
    *)
        echo "错误: EVAL_MODE 必须是 cn, en 或 all"
        exit 1
        ;;
esac

mkdir -p $OUTPUT_DIR

echo "============================================"
echo "AGIEval 评测"
echo "============================================"
echo "模型路径: $MODEL_PATH"
echo "评测模式: $EVAL_MODE"
echo "任务说明: $DESC"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "AGIEval 中文子集包含:"
echo "  - agieval_gaokao_chinese    高考语文"
echo "  - agieval_gaokao_english    高考英语"
echo "  - agieval_gaokao_mathqa     高考数学(选择)"
echo "  - agieval_gaokao_mathcloze  高考数学(填空)"
echo "  - agieval_gaokao_physics    高考物理"
echo "  - agieval_gaokao_chemistry  高考化学"
echo "  - agieval_gaokao_biology    高考生物"
echo "  - agieval_gaokao_history    高考历史"
echo "  - agieval_gaokao_geography  高考地理"
echo "  - agieval_logiqa_zh         中文逻辑推理"
echo "  - agieval_jec_qa_ca/kd      法律考试"
echo "============================================"
echo ""

# 运行 lm-eval (使用 vLLM 后端)
lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=$TENSOR_PARALLEL,gpu_memory_utilization=$GPU_MEMORY_UTIL \
    --tasks $TASKS \
    --batch_size auto \
    --output_path $OUTPUT_DIR \
    --log_samples

echo ""
echo "============================================"
echo "评测完成!"
echo "============================================"
echo "结果保存至: $OUTPUT_DIR"
echo ""
echo "查看结果: cat $OUTPUT_DIR/results.json | python3 -m json.tool"
echo "============================================"

# ============================================
# Qwen3-8B 参考成绩 (来自官方)
# ============================================
# AGIEval (全部):
#   - Qwen3-8B Base:     ~65%
#   - Qwen3-8B-Instruct: ~70%
#
# AGIEval 中文子集:
#   - 高考语文: ~70%
#   - 高考数学: ~50-60%
#   - 高考英语: ~85%
#   - 高考理综: ~60-70%
#   - 高考文综: ~70-75%
# ============================================
