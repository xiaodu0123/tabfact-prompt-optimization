"""
Process TabFact dataset: convert to unified data format and create sampled subsets with balanced label distribution.
"""

import argparse
import random
import os
import json
import pandas as pd

from utils import load_json, save_json, show_label_distribution


def process_dataset(base_path, input_tsv_path, output_json_path):

    df = pd.read_csv(input_tsv_path, sep='\t')

    result = []

    for _, row in df.iterrows():
        table_df = pd.read_csv(base_path + row['context'])
        table_content = [table_df.columns.tolist()] + table_df.values.tolist()
        target = ""
        if isinstance(row['targetValue'], bool):
            target = "supports" if row['targetValue'] else "refutes"

        new_row = {
            "id": row['id'],
            'claim': row['utterance'],
            'table': table_content,
            'table_caption': row['caption'],
            'label': target
        }

        result.append(new_row)

    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Processed {len(result)} entries and saved to {output_json_path}")

def process_into_scitab_format(base_path, input_tsv_path, output_json_path):

    df = pd.read_csv(input_tsv_path, sep='\t')

    result = []

    for _, row in df.iterrows():
        table_df = pd.read_csv(base_path + row['context'])
        table_column_names = table_df.columns.tolist()
        table_content_values = table_df.values.tolist()
        target = ""
        if isinstance(row['targetValue'], bool):
            target = "supports" if row['targetValue'] else "refutes"
        
        new_row = {
            "id": row['id'],
            "claim": row['utterance'],
            "table_caption": row['caption'],
            "table_column_names": table_column_names,
            "table_content_values": table_content_values,
            "label": target
        }

        result.append(new_row)
    
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Processed {len(result)} entries and saved to {output_json_path}")

def sample_split(data, size, seed):
    random.seed(seed)
    return random.sample(data, min(size, len(data)))


def main(args):
    # Convert TabFact data to SciTab format
    base_path = args.base_path
    input_tsv_path = args.input_tsv
    output_json_path = args.output_json
    process_into_scitab_format(base_path, input_tsv_path, output_json_path)

    # Create sampled subsets for TabFact
    processed_dir = args.processed_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainset = load_json(f"{processed_dir}/tabfact_train.json")
    devset = load_json(f"{processed_dir}/tabfact_val.json")
    testset = load_json(f"{processed_dir}/tabfact_test.json")

    sampled_train = sample_split(trainset, 200, seed=42)
    sampled_val = sample_split(devset, 400, seed=42)
    sampled_test = sample_split(testset, 400, seed=42)

    save_json(sampled_train, f"{output_dir}/sampled_train.json")
    save_json(sampled_val, f"{output_dir}/sampled_val.json")
    save_json(sampled_test, f"{output_dir}/sampled_test.json")

    print("=== Label distribution (sampled train set) ===")
    show_label_distribution(sampled_train)
    print("=== Label distribution (sampled val set) ===")
    show_label_distribution(sampled_val)
    print("=== Label distribution (sampled test set) ===")
    show_label_distribution(sampled_test)

    # Create a 100-instance training subset with a fixed seed for reproducibility
    random.seed(41)
    train_100 = random.sample(trainset, 100)
    random.shuffle(train_100)
    save_json(train_100, f"{output_dir}/tabfact_train_100.json")
    print("=== Label distribution (train set with 100 instances) ===")
    show_label_distribution(train_100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample subsets from TabFact dataset")
    parser.add_argument("--base_path", required=True,
                        help="Base path for TabFact dataset")
    parser.add_argument("--input_tsv", required=True,
                        help="Path to the TabFact data file (.tsv)")
    parser.add_argument("--output_json", required=True,
                        help="Path to the processed TabFact data (.json)")
    parser.add_argument("--processed_dir", required=True,
                        help="Directory containing processed TabFact JSON files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write sampled output files")
    main(parser.parse_args())
