"""
Process MMSci-Table dataset: convert to unified data format
"""

import re
import os
import argparse

from utils import load_json, save_json, show_label_distribution


TOKENS_TO_REMOVE = [
    "[BOLD] ", "[ITALIC] ", "[EMPTY]", "[CONTINUE]",
    "<bold>", "</bold>", "<italic>", "</italic>",
]

SCIGEN_TEST_FILES = ["test-Other", "test-CL"]


def clean_text(text, tokens_to_remove=TOKENS_TO_REMOVE):
    """
    Remove special tokens from text and normalize whitespace.
    """
    for token in tokens_to_remove:
        text = str(text).replace(token, '')
    return re.sub(r'\s+', ' ', text).strip()


def extract_label(answer):
    """
    Extract the gold answer label from an MMSci answer field.
    """
    if isinstance(answer, dict):
        return answer.get("answer")
    if isinstance(answer, str):
        match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', answer)
        if match:
            return match.group(1)
    return None


def load_scigen_test_data(scigen_dir):
    """
    Load all SciGen test data into a dict.
    """
    scigen_test_data = {}
    for split_name in SCIGEN_TEST_FILES:
        path = os.path.join(scigen_dir, f"{split_name}.json")
        scigen_test_data[split_name] = load_json(path)
        print(f"Loaded SciGen split '{split_name}': {len(scigen_test_data[split_name])} items")
    
    return scigen_test_data


def process_mmsci_data(eval_data, scigen_test_data):
    """
    Convert MMSci eval data to SciTab format.

    Args:
        eval_data: List of MMSci eval instances.
        scigen_test_data: Dict of SciGen test data keyed by split name and index.

    Returns:
        List of processed instances in SciTab format.
    """
    processed_samples = []

    for idx, item in enumerate(eval_data):
        # Resolve the corresponding SciGen table entry
        scigen_path = item["images"][0].replace("scigen/", "").replace(".jpg", "")
        filename, index = scigen_path.split("/")
        scigen_item = scigen_test_data[filename][index]

        assert item["table_caption"][0] == scigen_item["table_caption"], (
            f"Caption mismatch at index {idx}: "
            f"{item['table_caption'][0]!r} != {scigen_item['table_caption']!r}"
        )

        column_names = [
            clean_text(col) for col in scigen_item["table_column_names"]
        ]
        content_values = [
            [clean_text(cell) for cell in row]
            for row in scigen_item["table_content_values"]
        ]

        processed_samples.append({
            "id": str(idx),
            "claim": item["statement"],
            "table_caption": item["table_caption"][0],
            "table_column_names": column_names,
            "table_content_values": content_values,
            "label": extract_label(item["answer"]),
        })

    return processed_samples


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Load SciGen test splits
    scigen_test_data = load_scigen_test_data(args.scigen_dir)

    # Load and filter MMSci eval data to TFV category only
    eval_data = load_json(args.eval_data)
    eval_data = [item for item in eval_data if 'scigen_for_TFV' in item['category']]
    print(f"Loaded {len(eval_data)} MMSci TFV instances from {args.eval_data}")

    processed = process_mmsci_data(eval_data, scigen_test_data)
    save_json(processed, args.output)
    print(f"Saved {len(processed)} processed instances to {args.output}")

    print("\n=== MMSci label distribution ===")
    show_label_distribution(processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MMSci-Table dataset into SciTab format")
    parser.add_argument("--eval_data", required=True,
                        help="Path to the raw MMSCi-Table test data")
    parser.add_argument("--scigen_dir", required=True,
                        help="Directory containing SciGen test data (.json)")
    parser.add_argument("--output", required=True,
                        help="Path to save the processed MMSci-Table test data (.json)")
    main(parser.parse_args())
