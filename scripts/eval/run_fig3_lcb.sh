#!/usr/bin/env bash
# LiveCodeBench v5 eval used for the SFT-U/V sigma-interpolation experiment
# (Fig-3 mirror). Generates + scores (pass@1/pass@4) for ONE model on ONE GPU
# with the exact parameters used in replace/RESULTS_sft_uv_sigma_interp.md.
#
# Usage: run_fig3_lcb.sh <GPU_ID> <MODEL_PATH> <OUT_DIR>
#   e.g. run_fig3_lcb.sh 0 /path/to/models/alpha_0.4 ./outputs/fig3/alpha_0.4
#
# Data prep (once): python tools/download_datasets.py
# NOTE: pass the .json to data.path. A pandas-written parquet of this dataset
# has nested columns >2GB and fails pd.read_parquet; main_generation's JSON
# fallback is the intended path. The scoring fallback needs `polars` installed.
# Run from the ArcherCodeR repo root.
set -u
GPU=${1:?gpu id}
MODEL_PATH=${2:?model path}
OUT_DIR=${3:?output dir}

PY=${PY:-python}
mkdir -p "$OUT_DIR"

export HF_HOME=${HF_HOME:-/shared/huggingface}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$(pwd)

if [[ -f "$OUT_DIR/livecodebench_v5_out.parquet.pass.csv" ]]; then
  echo "[skip] pass.csv already exists"
  exit 0
fi

$PY -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    model.path="$MODEL_PATH" \
    data.path=data/test/livecodebench_v5.json \
    data.output_path="$OUT_DIR/livecodebench_v5_out.parquet" \
    data.batch_size=32 \
    data.n_samples=4 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.85 \
    rollout.free_cache_engine=False \
    rollout.tensor_model_parallel_size=1 \
    rollout.temperature=0.8 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=2048 \
    rollout.response_length=32768 \
    rollout.max_num_batched_tokens=34816
echo "=== $(date) | done. pass@1/pass@4 in $OUT_DIR/livecodebench_v5_out.parquet.pass.csv ==="
