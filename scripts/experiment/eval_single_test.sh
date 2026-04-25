#!/bin/bash

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ]; then
    echo "Usage: $0 <model_name> <module> <port> [load_path]"
    echo "Example: $0 gpt-4o-mini cot 8000"
    echo "Example: $0 qwen3-8b react 8001 ../artifacts/models/qwen3-8b/react/copro_hybrid/optimised_program.json"
    exit 1
fi

MODEL_NAME="$1"
MODULE="$2"
PORT="$3"
LOAD_PATH="$4"

if [ "$LOAD_PATH" != "" ]; then
    LOAD_ARG="--load_path $LOAD_PATH"
    OPTIMIZER_DIR=$(basename "$(dirname "$LOAD_PATH")")
    SESSION_PREFIX="${MODEL_NAME}_${MODULE}_${OPTIMIZER_DIR}_test_eval"
    echo "Using load_path: $LOAD_PATH"
else
    LOAD_ARG=""
    SESSION_PREFIX="${MODEL_NAME}_${MODULE}_baseline_test_eval"
    echo "Evaluating baseline model"
fi

echo "Running evaluation for model: $MODEL_NAME, module: $MODULE, port: $PORT"
echo "Session prefix: $SESSION_PREFIX"
echo "=================================================="

mkdir -p logs

echo "Starting scitab test evaluation..."
python eval.py --model_name "$MODEL_NAME" --module "$MODULE" --port "$PORT" --dataset scitab --split test --num_threads 1 $LOAD_ARG > "logs/${SESSION_PREFIX}_scitab_test.log" 2>&1 &

echo "Starting tabfact test evaluation..."
python eval.py --model_name "$MODEL_NAME" --module "$MODULE" --port "$PORT" --dataset tabfact --split test --num_threads 1 $LOAD_ARG > "logs/${SESSION_PREFIX}_tabfact_test.log" 2>&1 &

echo "Starting pubhealthtab test evaluation..."
python eval.py --model_name "$MODEL_NAME" --module "$MODULE" --port "$PORT" --dataset pubhealthtab --split test --num_threads 1 $LOAD_ARG > "logs/${SESSION_PREFIX}_pubhealthtab_test.log" 2>&1 &

echo "Starting mmsci test evaluation..."
python eval.py --model_name "$MODEL_NAME" --module "$MODULE" --port "$PORT" --dataset mmsci --split test --num_threads 4 $LOAD_ARG > "logs/${SESSION_PREFIX}_mmsci_test.log" 2>&1 &

echo "All evaluations started in parallel!"
echo "Check logs with:"
echo "  tail -f logs/${SESSION_PREFIX}_scitab_test.log"
echo "  tail -f logs/${SESSION_PREFIX}_tabfact_test.log"
echo "  tail -f logs/${SESSION_PREFIX}_pubhealthtab_test.log"
echo "  tail -f logs/${SESSION_PREFIX}_mmsci_test.log"
