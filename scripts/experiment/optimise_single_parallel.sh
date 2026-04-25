#!/bin/bash

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ]; then
    echo "Usage: $0 <model_name> <module> <port> [instruct_model]"
    echo "Example: $0 gpt-4o-mini cot 8000 gpt-4o"
    echo "Example: $0 qwen3-8b predict 8001"
    exit 1
fi

MODEL_NAME="$1"
MODULE="$2"
PORT="$3"
INSTRUCT_MODEL="$4"

if [ "$INSTRUCT_MODEL" != "" ]; then
    INSTRUCT_ARG="--instruct_model $INSTRUCT_MODEL"
    SESSION_PREFIX="${MODEL_NAME}_${MODULE}_${INSTRUCT_MODEL}"
    echo "Using $INSTRUCT_MODEL for instruction generation"
else
    INSTRUCT_ARG=""
    SESSION_PREFIX="${MODEL_NAME}_${MODULE}"
    echo "Using $MODEL_NAME for instruction generation"
fi

echo "Running optimization experiments for module: $MODULE, model: $MODEL_NAME, port: $PORT"
echo "=================================================="

echo "Starting COPRO optimization in tmux session..."
tmux new-session -d -s "${SESSION_PREFIX}_copro" \
    "python optimise.py --module $MODULE --model_name $MODEL_NAME --port $PORT $INSTRUCT_ARG --dataset hybrid --optimiser copro --breadth 6 --num_threads 4 --init_temperature 1.2"

echo "Starting MIPROv2 optimization in tmux session..."
tmux new-session -d -s "${SESSION_PREFIX}_miprov2" \
    "python optimise.py --module $MODULE --model_name $MODEL_NAME --port $PORT $INSTRUCT_ARG --dataset hybrid --optimiser miprov2 --auto_mode medium --save_stats --num_threads 4"

if [ "$INSTRUCT_MODEL" = "" ]; then
    echo "Starting SIMBA optimization in tmux session..."
    tmux new-session -d -s "${SESSION_PREFIX}_simba" \
        "python optimise.py --module $MODULE --model_name $MODEL_NAME --port $PORT --dataset hybrid --optimiser simba --save_stats --num_threads 4"
else
    echo "Skipping SIMBA optimization (instruct_model specified)"
fi

echo "All optimizers started in parallel tmux sessions!"
echo "To check progress, use:"
echo "  tmux list-sessions"
echo "  tmux attach -t ${SESSION_PREFIX}_copro"
echo "  tmux attach -t ${SESSION_PREFIX}_miprov2"
if [ "$INSTRUCT_MODEL" = "" ]; then
    echo "  tmux attach -t ${SESSION_PREFIX}_simba"
fi
