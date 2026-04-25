"""
Evaluate the performance of a DSPy program on a specific table-based fact checking dataset.
"""

import argparse
import json

import dspy
from dspy.evaluate import Evaluate, answer_exact_match

from model_config import TableFactCheckSignature
from utils import (
    create_lm,
    get_dataset_config,
    get_middle_path_from_load_path,
    get_param,
    load_examples,
    load_yaml_config,
    process_evaluation_results,
    resolve_path,
    setup_tracking,
)


def execute_sql(table_data: list, table_name: str, sql_query: str) -> str:
    import duckdb
    import pandas as pd

    try:
        headers = table_data[0]
        rows = table_data[1:]
        df = pd.DataFrame(rows, columns=headers)
        conn = duckdb.connect()
        conn.register(table_name, df)
        result_df = conn.execute(sql_query).fetchdf()
        conn.close()
        return f"Table:\n{result_df.to_string()}"
    except Exception as e:
        return f"SQL Error: {str(e)}"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DSPy program on a table-based fact checking dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--module", type=str, choices=["predict", "cot", "pot", "react", "codeact"], help="which DSPy module to use")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model used for inference")
    parser.add_argument("--dataset", type=str, choices=["tabfact", "scitab", "pubhealthtab", "hybrid", "mmsci"], default="tabfact", help="Dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the prediction results")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the optimised program. If None, evaluate the original program")
    parser.add_argument("--tracking", type=str, choices=["mlflow", "langfuse", "none"], default=None, help="Tracking system to use")
    parser.add_argument("--port", type=int, default=None, help="Port number for local model API server")
    parser.add_argument("--max_iters", type=int, default=None, help="Maximum iterations for Program of Thought, ReAct, CodeAct")
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads for running evaluation")
    return parser.parse_args()


def main(args):
    config = load_yaml_config(args.config)
    params = config["params"]

    args.tracking = get_param(args.tracking, params["tracking"])
    args.max_iters = get_param(args.max_iters, params["max_iters"])
    args.num_threads = get_param(args.num_threads, params["num_threads"])
    args.port = get_param(args.port, params["port"])

    lm = create_lm(args.model_name, args.port, config)
    dspy.configure(lm=lm)

    dataset_config = get_dataset_config(args.dataset.lower(), config)
    with open(resolve_path(dataset_config[f"{args.split}_path"]), "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    evalset = load_examples(eval_data)
    print(f"Loaded {args.dataset} {args.split} set: {len(evalset)} examples")

    if args.module == "predict":
        fact_checker = dspy.Predict(TableFactCheckSignature)
    elif args.module == "cot":
        fact_checker = dspy.ChainOfThought(TableFactCheckSignature)
    elif args.module == "pot":
        fact_checker = dspy.ProgramOfThought(TableFactCheckSignature, max_iters=args.max_iters)
    elif args.module == "react":
        fact_checker = dspy.ReAct(TableFactCheckSignature, tools=[execute_sql], max_iters=args.max_iters)
    else:
        fact_checker = dspy.CodeAct(TableFactCheckSignature, tools=[], max_iters=args.max_iters)

    if args.load_path is not None:
        fact_checker.load(resolve_path(args.load_path).as_posix())
        print(f"Loaded optimised program from {args.load_path}")

    if args.load_path is not None:
        middle_path = get_middle_path_from_load_path(
            args.load_path,
            resolve_path(config["paths"]["models_dir"]),
            args.model_name,
            args.module,
        )
        experiment_name = f"{'_'.join(middle_path.parts)}_{args.dataset}_{args.split}"
    else:
        experiment_name = f"{args.model_name}_{args.module}_baseline_{args.dataset}_{args.split}"

    setup_tracking(args.tracking, experiment_name, config)

    evaluator = Evaluate(
        devset=evalset,
        metric=answer_exact_match,
        num_threads=args.num_threads,
        display_progress=True,
    )
    eval_result = evaluator(fact_checker)
    print(f"{args.split} accuracy: {eval_result.score:.4f}")

    if args.save_path is None:
        args.save_path = config["paths"]["results_dir"]

    save_root = resolve_path(args.save_path)
    if args.load_path is not None:
        middle_path = get_middle_path_from_load_path(
            args.load_path,
            resolve_path(config["paths"]["models_dir"]),
            args.model_name,
            args.module,
        )
        results_dir = save_root / middle_path
    else:
        results_dir = save_root / args.model_name / args.module / "baseline"

    filename = f"{args.dataset}_{args.split}_results.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / filename

    processed_result = process_evaluation_results(eval_result.results)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_result, f, indent=4)

    print(f"Processed {len(processed_result)} results and saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
