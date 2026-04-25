"""
Optimise a DSPy program on train/dev data and then save the optimised program.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import dspy
from dspy.evaluate import answer_exact_match

from model_config import TableFactCheckSignature
from utils import create_lm, get_dataset_config, get_param, load_examples, load_yaml_config, resolve_path, save_json, save_optimizer_stats, setup_tracking


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
    parser = argparse.ArgumentParser(description="Optimise a DSPy program on train data and then save the optimised program")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--module", type=str, choices=["predict", "cot", "pot", "react", "codeact"], help="which DSPy module to use")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model used for inference")
    parser.add_argument("--dataset", type=str, choices=["tabfact", "scitab", "pubhealthtab", "hybrid"], default="hybrid", help="Dataset to use for training")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the optimised program")
    parser.add_argument("--tracking", type=str, choices=["mlflow", "langfuse", "none"], default=None, help="Tracking system to use")
    parser.add_argument("--max_iters", type=int, default=None, help="Maximum iterations for Program of Thought, ReAct, CodeAct")
    parser.add_argument("--num_threads", type=int, default=None, help="Number of threads for running evaluation")
    parser.add_argument("--port", type=int, default=None, help="Port number for local model API server")
    parser.add_argument("--optimiser", type=str, choices=["copro", "miprov2", "simba"], default=None, help="Optimisation method to use")
    parser.add_argument("--instruct_model", type=str, default=None, help="Model used for generating candidate instructions in optimisation")
    parser.add_argument("--save_stats", action="store_true", help="Whether to save optimiser statistics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--seed_instruction", type=str, default=None, help="Seed instruction for instruction optimization")

    copro_group = parser.add_argument_group("COPRO hyperparameters", "Arguments specific to COPRO optimizer")
    copro_group.add_argument("--breadth", type=int, default=None, help="Number of candidates to consider per iteration")
    copro_group.add_argument("--depth", type=int, default=None, help="Number of iteration steps")
    copro_group.add_argument("--init_temperature", type=float, default=None, help="Initial temperature for COPRO")

    miprov2_group = parser.add_argument_group("MIPROv2 hyperparameters", "Arguments specific to MIPROv2 optimizer")
    miprov2_group.add_argument("--auto_mode", type=str, choices=["light", "medium", "heavy"], default=None, help="Auto mode for MIPROv2 to set n and val_size")

    simba_group = parser.add_argument_group("SIMBA hyperparameters", "Arguments specific to SIMBA optimizer")
    simba_group.add_argument("--num_candidates", type=int, default=None, help="Number of candidate programs to generate per iteration")
    simba_group.add_argument("--max_steps", type=int, default=None, help="Maximum number of optimisation steps to run")

    return parser.parse_args()


def main(args):
    config = load_yaml_config(args.config)
    params = config["params"]
    optimisers = config["optimisers"]

    args.tracking = get_param(args.tracking, params["tracking"])
    args.max_iters = get_param(args.max_iters, params["max_iters"])
    args.num_threads = get_param(args.num_threads, params["num_threads"])
    args.port = get_param(args.port, params["port"])
    args.optimiser = get_param(args.optimiser, params["optimiser"])
    args.seed = get_param(args.seed, params["seed"])
    args.seed_instruction = get_param(args.seed_instruction, params["seed_instruction"])

    if args.optimiser == "copro":
        args.breadth = get_param(args.breadth, optimisers["copro"]["breadth"])
        args.depth = get_param(args.depth, optimisers["copro"]["depth"])
        args.init_temperature = get_param(args.init_temperature, optimisers["copro"]["init_temperature"])
    elif args.optimiser == "miprov2":
        args.auto_mode = get_param(args.auto_mode, optimisers["miprov2"]["auto_mode"])
    elif args.optimiser == "simba":
        args.num_candidates = get_param(args.num_candidates, optimisers["simba"]["num_candidates"])
        args.max_steps = get_param(args.max_steps, optimisers["simba"]["max_steps"])

    experiment_name = f"train_{args.model_name}_{args.module}_{args.optimiser}_{args.dataset}"
    if args.instruct_model:
        experiment_name = f"train_{args.model_name}_prompt_{args.instruct_model}_{args.module}_{args.optimiser}_{args.dataset}"
    setup_tracking(args.tracking, experiment_name, config)

    lm = create_lm(args.model_name, args.port, config)
    dspy.configure(lm=lm)
    prompt_model = lm if args.instruct_model is None else create_lm(args.instruct_model, args.port, config)

    dataset_config = get_dataset_config(args.dataset.lower(), config)
    with open(resolve_path(dataset_config["train_path"]), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(resolve_path(dataset_config["dev_path"]), "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    trainset = load_examples(train_data)
    devset = load_examples(dev_data)
    print(f"Loaded {args.dataset} train set: {len(trainset)} examples, dev set: {len(devset)} examples")

    if args.module == "predict":
        fact_checker = dspy.Predict(TableFactCheckSignature.with_instructions(args.seed_instruction))
    elif args.module == "cot":
        fact_checker = dspy.ChainOfThought(TableFactCheckSignature.with_instructions(args.seed_instruction))
    elif args.module == "pot":
        fact_checker = dspy.ProgramOfThought(TableFactCheckSignature.with_instructions(args.seed_instruction), max_iters=args.max_iters)
    elif args.module == "react":
        fact_checker = dspy.ReAct(TableFactCheckSignature.with_instructions(args.seed_instruction), tools=[execute_sql], max_iters=args.max_iters)
    else:
        fact_checker = dspy.CodeAct(TableFactCheckSignature.with_instructions(args.seed_instruction), tools=[], max_iters=args.max_iters)

    if args.optimiser == "copro":
        optimiser = dspy.COPRO(prompt_model=prompt_model, metric=answer_exact_match, breadth=args.breadth, depth=args.depth, init_temperature=args.init_temperature, track_stats=True)
        optimised_fact_checker = optimiser.compile(fact_checker, trainset=trainset, eval_kwargs=dict(num_threads=args.num_threads, display_progress=True))
    elif args.optimiser == "miprov2":
        optimiser = dspy.MIPROv2(prompt_model=prompt_model, metric=answer_exact_match, auto=args.auto_mode, num_threads=args.num_threads, track_stats=True, max_bootstrapped_demos=0, max_labeled_demos=0, seed=args.seed)
        optimised_fact_checker = optimiser.compile(fact_checker, trainset=trainset, seed=args.seed)
    else:
        optimiser = dspy.SIMBA(metric=answer_exact_match, max_demos=0, max_steps=args.max_steps, num_candidates=args.num_candidates, num_threads=args.num_threads, prompt_model=prompt_model)
        optimised_fact_checker = optimiser.compile(fact_checker, trainset=trainset, seed=args.seed)

    save_root = resolve_path(config["paths"]["models_dir"] if args.save_path is None else args.save_path)
    program_name = f"{args.optimiser}_{args.dataset}"
    if args.instruct_model:
        program_name = f"{program_name}_prompt_{args.instruct_model}"
    program_save_path = Path(save_root) / args.model_name / args.module / program_name
    program_save_path.mkdir(parents=True, exist_ok=True)

    state_save_path = program_save_path / "optimised_program.json"
    optimised_fact_checker.save(state_save_path.as_posix(), save_program=False)
    print(f"Optimised program state saved to: {state_save_path}")

    if args.save_stats:
        save_optimizer_stats(optimised_fact_checker, program_save_path / "optimised_stats.json", args.optimiser, experiment_name)

    optimised_fact_checker.save(program_save_path.as_posix(), save_program=True)

    hyperparams = {
        "module": args.module,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "max_iters": args.max_iters,
        "tracking": args.tracking,
        "instruct_model": args.instruct_model,
        "optimiser": args.optimiser,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    if args.optimiser == "copro":
        hyperparams.update({"breadth": args.breadth, "depth": args.depth, "init_temperature": args.init_temperature})
    elif args.optimiser == "miprov2":
        hyperparams.update({"auto_mode": args.auto_mode})
    else:
        hyperparams.update({"num_candidates": args.num_candidates, "max_steps": args.max_steps, "seed": args.seed})

    save_json(hyperparams, program_save_path / "hyperparameters.json")
    print(f"Hyperparameters saved to: {program_save_path / 'hyperparameters.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
