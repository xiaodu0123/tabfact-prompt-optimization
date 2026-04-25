"""
Process PubHealthTab dataset: convert to unified data format and create sampled subsets with balanced label distribution.
"""

import os
import argparse
import random
import jsonlines

from utils import load_json, save_json, show_label_distribution


LABEL_MAP = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not enough info",
}

def process_pubhealth_data(input_file_path, output_file_path):
    """
    Convert PubHealthTab JSONL to a standardized JSON format.

    Filters out instances that have both vertical and horizontal headers that
    differ from each other (ambiguous table orientation), then normalizes the
    table schema to match the TabFact/SciTab format.
    """
    data = []
    with jsonlines.open(input_file_path) as reader:
        for line in reader:
            data.append(line)

    print(f"Loaded {len(data)} instances from {input_file_path}")
    processed = []

    for item in data:
        table = item.get('table', {})
        header_v = table.get('header_vertical', [])
        header_h = table.get('header_horizontal', [])

        # Skip instances with conflicting vertical and horizontal headers
        if len(header_v) > 0 and len(header_h) > 0 and header_v != header_h:
            continue

        column_names = header_h.copy()
        rows = table.get('rows', []).copy()

        # Pad rows and column names to a consistent width
        if rows:
            max_row_len = len(max(rows, key=len))
            df_width = max(max_row_len, len(column_names))

            while len(column_names) < df_width:
                column_names.insert(0, "")
            for row in rows:
                while len(row) < df_width:
                    row.insert(0, "")

        processed.append({
            "id": item["_id"],
            "claim": item["claim"],
            "table_caption": table.get("caption"),
            "table_column_names": column_names,
            "table_content_values": rows,
            "table_website": table.get("website"),
            "table_html_code": table.get("html_code"),
            "label": LABEL_MAP.get(item["label"], item["label"].lower()),
        })

    save_json(processed, output_file_path)
    print(f"Saved {len(processed)} processed instances to {output_file_path}")
    return processed


def balance_supports(data, target_supports_count, seed=42):
    """Downsample 'supports' instances to reduce label imbalance."""
    supports = [item for item in data if item['label'] == 'supports']
    others = [item for item in data if item['label'] != 'supports']
    random.seed(seed)
    sampled_supports = random.sample(supports, min(target_supports_count, len(supports)))
    balanced = sampled_supports + others
    random.shuffle(balanced)
    return balanced


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    train_data = process_pubhealth_data(
        f"{args.input_dir}/pubhealthtab_trainset.jsonl",
        f"{args.output_dir}/pubhealthtab_train.json",
    )
    eval_data = process_pubhealth_data(
        f"{args.input_dir}/pubhealthtab_devset.jsonl",
        f"{args.output_dir}/pubhealthtab_dev.json",
    )
    test_data = process_pubhealth_data(
        f"{args.input_dir}/pubhealthtab_testset.jsonl",
        f"{args.output_dir}/pubhealthtab_test.json",
    )

    print("\n=== PubHealthTab Train set label distribution ===")
    show_label_distribution(train_data)

    # Balance training set by downsampling 'supports'
    train_balanced = balance_supports(train_data, target_supports_count=350, seed=42)
    save_json(train_balanced, f"{args.output_dir}/pubhealthtab_train_balanced.json")
    print("\n=== PubHealthTab Train set label distribution (Balanced) ===")
    show_label_distribution(train_balanced)

    # Build a balanced eval set
    eval_balanced = balance_supports(eval_data, target_supports_count=40, seed=42)
    save_json(eval_balanced, f"{args.output_dir}/pubhealthtab_dev_balanced.json")
    print("\n=== PubHealthTab Dev set label distribution (Balanced) ===")
    show_label_distribution(eval_balanced)

    # 200-instance training sample
    train_sampled = random.sample(train_balanced, min(200, len(train_balanced)))
    save_json(train_sampled, f"{args.output_dir}/pubhealthtab_train_sampled.json")
    print("\n=== PubHealthTab Train set label distribution (Sampled) ===")
    show_label_distribution(train_sampled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PubHealthTab dataset")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing raw PubHealthTab JSONL files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write processed output files")
    main(parser.parse_args())
