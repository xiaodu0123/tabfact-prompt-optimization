from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

import dspy

from model_config import load_config


def table_formater(table_list):
    """
    Convert a list of lists table to a string representation.
    First row is headers, subsequent rows are data.
    """
    if not table_list or len(table_list) == 0:
        return "Empty table"

    headers = table_list[0]
    rows = table_list[1:]

    table_str = ""
    header = "|| " + " | ".join(map(str, headers)) + " ||\n"
    table_str += header

    for row in rows:
        row_str = "|| " + " | ".join(map(str, row)) + " ||\n"
        table_str += row_str

    return table_str


def load_examples(data) -> List[dspy.Example]:
    """Convert a list of examples in SciTab format into dspy.Example format"""
    examples = []
    for item in data:
        table_headers = item["table_column_names"]
        table_rows = item["table_content_values"]
        complete_table = [table_headers] + table_rows
        caption = item["table_caption"] if item["table_caption"] is not None else ""

        example = dspy.Example(
            claim=item["claim"],
            table=table_formater(complete_table),
            caption=caption,
            answer=str(item["label"]),
        ).with_inputs("claim", "table", "caption")
        examples.append(example)

    return examples


def process_evaluation_results(evaluation_results):
    """
    Process the evaluation results from dspy.Evaluate and save to JSON file
    """
    processed_results = []

    for i, (example, prediction, score) in enumerate(evaluation_results):
        try:
            result = {
                "example_id": i,
                "claim": example.claim,
                "caption": example.caption,
                "table": example.table,
                "gold_label": example.answer,
                "predicted_label": prediction.answer,
                "is_correct": bool(score),
            }
        except Exception as e:
            result = {
                "example_id": i,
                "claim": example.claim,
                "caption": example.caption,
                "table": example.table,
                "gold_label": example.answer,
                "predicted_label": "None",
                "is_correct": False,
                "error": str(e),
            }

        if hasattr(prediction, "items"):
            for key, value in prediction.items():
                if key != "answer":
                    result[key] = value

        processed_results.append(result)

    return processed_results


def setup_tracking(tracking_type, experiment_name, config):
    """Setup tracking system based on user choice"""
    if tracking_type == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(config["tracking"]["mlflow_uri"])
        mlflow.set_experiment(experiment_name)
        mlflow.dspy.autolog(log_traces_from_eval=True, log_evals=True)
        print(f"MLflow tracking enabled for experiment: {experiment_name}")

    elif tracking_type == "langfuse":
        from langfuse import Langfuse
        from openinference.instrumentation.dspy import DSPyInstrumentor

        langfuse = Langfuse(
            secret_key=config["api_keys"]["langfuse_secret_key"],
            public_key=config["api_keys"]["langfuse_public_key"],
            host=config["tracking"]["langfuse_host"],
        )
        langfuse.update_current_trace(session_id=experiment_name)
        DSPyInstrumentor().instrument()
        print(f"Langfuse tracking enabled for experiment: {experiment_name}")

    else:
        print("No tracking enabled")


def create_lm(model_name: str, port: int, config: dict[str, Any]) -> dspy.LM:
    openai_api_key = config["api_keys"]["openai_api_key"]

    if model_name == "gpt-4o":
        return dspy.LM("openai/gpt-4o", api_key=openai_api_key, cache=True)

    if model_name == "gpt-4o-mini":
        return dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key, cache=True)

    if model_name == "gpt-4.1":
        return dspy.LM("openai/gpt-4.1", api_key=openai_api_key, cache=True)

    if model_name == "qwen3-8b":
        return dspy.LM(
            model="hosted_vllm/Qwen/Qwen3-8B",
            api_base=f"http://localhost:{port}/v1",
            model_type="chat",
            chat_template_kwargs={"enable_thinking": False},
        )

    if model_name == "qwen3-32b":
        return dspy.LM(
            model="hosted_vllm/Qwen/Qwen3-32B",
            api_base=f"http://localhost:{port}/v1",
            model_type="chat",
            chat_template_kwargs={"enable_thinking": False},
        )

    if model_name == "gemma3-12b":
        return dspy.LM(
            model="hosted_vllm/google/gemma-3-12b-it",
            api_base=f"http://localhost:{port}/v1",
            model_type="chat",
        )

    if model_name == "gemma3-27b":
        return dspy.LM(
            model="hosted_vllm/google/gemma-3-27b-it",
            api_base=f"http://localhost:{port}/v1",
            model_type="chat",
            cache=True,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_dataset_config(dataset_name: str, config: dict[str, Any]):
    datasets = config["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    return datasets[dataset_name]


def get_param(args_value, config_value):
    return config_value if args_value is None else args_value


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return path.resolve()


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_middle_path_from_load_path(load_path: str, models_root: Path, model_name: str, module: str) -> Path:
    resolved_load_path = resolve_path(load_path)
    if resolved_load_path.is_file():
        resolved_load_path = resolved_load_path.parent

    try:
        relative_path = resolved_load_path.relative_to(models_root)
        if len(relative_path.parts) >= 3:
            return Path(*relative_path.parts[:3])
    except ValueError:
        pass

    return Path(model_name) / module / "unknown"


def load_yaml_config(config_path: str | None = None):
    return load_config(config_path)


def save_optimizer_stats(optimized_program, save_path, optimizer_name, experiment_name):
    """
    Save optimizer statistics to a JSON file.
    """
    stats = {
        "experiment_name": experiment_name,
        "optimizer": optimizer_name,
        "timestamp": datetime.now().isoformat(),
        "optimizer_stats": {},
    }

    try:
        if optimizer_name == "copro":
            stats["optimizer_stats"] = _extract_copro_stats(optimized_program)
        elif optimizer_name == "miprov2":
            stats["optimizer_stats"] = _extract_miprov2_stats(optimized_program)
        elif optimizer_name == "simba":
            stats["optimizer_stats"] = _extract_simba_stats(optimized_program)
        else:
            stats["optimizer_stats"] = {"error": f"Unknown optimizer: {optimizer_name}"}
    except Exception as e:
        stats["optimizer_stats"] = {"extraction_error": str(e)}

    save_json(stats, Path(save_path))
    print(f"Optimizer statistics saved to: {save_path}")


def _extract_miprov2_stats(program):
    stats = {}

    if hasattr(program, "trial_logs"):
        stats["num_trials"] = len(program.trial_logs)
        trial_scores = {}
        for trial_num, log in program.trial_logs.items():
            trial_data = {}
            if "full_eval_score" in log:
                trial_data["full_eval_score"] = log["full_eval_score"]
            if "mb_score" in log:
                trial_data["mb_score"] = log["mb_score"]
            if "total_eval_calls_so_far" in log:
                trial_data["total_eval_calls_so_far"] = log["total_eval_calls_so_far"]
            if trial_data:
                trial_scores[str(trial_num)] = trial_data
        if trial_scores:
            stats["trial_scores"] = trial_scores

    if hasattr(program, "prompt_model_total_calls"):
        stats["prompt_model_total_calls"] = program.prompt_model_total_calls

    if hasattr(program, "candidate_programs"):
        stats["num_full_eval_candidates"] = len(program.candidate_programs)
        top_candidates = []
        for i, candidate in enumerate(program.candidate_programs[:5]):
            top_candidates.append({"rank": i + 1, "score": candidate.get("score", 0)})
        stats["top_full_eval_candidates"] = top_candidates

    if hasattr(program, "mb_candidate_programs"):
        stats["num_minibatch_candidates"] = len(program.mb_candidate_programs)

    return stats


def _extract_copro_stats(program):
    stats = {}

    if hasattr(program, "candidate_programs"):
        stats["num_candidates"] = len(program.candidate_programs)
        candidates_info = []
        for i, candidate in enumerate(program.candidate_programs[:10]):
            candidate_info = {
                "rank": i + 1,
                "score": candidate.get("score", 0),
                "depth": candidate.get("depth", 0),
            }
            if "instruction" in candidate:
                candidate_info["instruction"] = candidate["instruction"]
            if "prefix" in candidate:
                candidate_info["prefix"] = candidate["prefix"]
            candidates_info.append(candidate_info)
        stats["top_candidates"] = candidates_info

    if hasattr(program, "results_best"):
        stats["results_best_summary"] = _summarize_copro_results(program.results_best)

    if hasattr(program, "results_latest"):
        stats["results_latest_summary"] = _summarize_copro_results(program.results_latest)

    if hasattr(program, "total_calls"):
        stats["total_optimization_calls"] = program.total_calls

    return stats


def _extract_simba_stats(program):
    stats = {}

    if hasattr(program, "candidate_programs"):
        stats["num_candidates"] = len(program.candidate_programs)
        candidates_info = []
        for i, candidate in enumerate(program.candidate_programs[:5]):
            candidates_info.append({"rank": i + 1, "score": candidate.get("score", 0)})
        stats["top_candidates"] = candidates_info

    if hasattr(program, "trial_logs"):
        stats["num_batches"] = len(program.trial_logs)
        batch_scores = {}
        for batch_num, log in program.trial_logs.items():
            if "train_score" in log:
                batch_scores[str(batch_num)] = log["train_score"]
        if batch_scores:
            stats["batch_scores"] = batch_scores

    return stats


def _summarize_copro_results(results_dict):
    if not results_dict:
        return {}

    summary = {}
    for predictor_id, metrics in results_dict.items():
        predictor_summary = {}
        for metric_name, values in metrics.items():
            if isinstance(values, list) and values:
                predictor_summary[metric_name] = {
                    "final_value": values[-1],
                    "max_value": max(values),
                    "num_depths": len(values),
                }
        summary[f"predictor_{predictor_id}"] = predictor_summary

    return summary
