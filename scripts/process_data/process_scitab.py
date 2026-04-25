"""
Process SciTab dataset into balanced train/dev/test splits
"""

import argparse
import random
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split

from utils import load_json, save_json, show_label_distribution


def create_balanced_splits(data, train_size=210, random_state=42):
    """
    Create balanced train/dev/test splits where each label is equally
    represented in every split.

    Args:
        data: List of dataset instances, each with a 'label' field
        train_size: Total number of training instances (divided equally across labels)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_set, dev_set, test_set)
    """
    label_groups = defaultdict(list)
    for instance in data:
        label_groups[instance['label']].append(instance)

    num_labels = len(label_groups)
    train_per_label = train_size // num_labels

    # How many instances per label remain after taking train_per_label for train
    remaining_per_label = [
        (len(instances) - train_per_label) // 2
        for instances in label_groups.values()
    ]
    dev_test_per_label = min(remaining_per_label)

    train_set, dev_set, test_set = [], [], []
    for label, instances in label_groups.items():
        train_instances, temp = train_test_split(
            instances, train_size=train_per_label, random_state=random_state
        )
        dev_instances, test_instances = train_test_split(
            temp,
            train_size=dev_test_per_label,
            test_size=dev_test_per_label,
            random_state=random_state,
        )
        train_set.extend(train_instances)
        dev_set.extend(dev_instances)
        test_set.extend(test_instances)

    print(f"Train: {len(train_set)}  Dev: {len(dev_set)}  Test: {len(test_set)}")
    for name, split in [("Train", train_set), ("Dev", dev_set), ("Test", test_set)]:
        label_counts = Counter(x['label'] for x in split)
        print(f"  {name}: {dict(label_counts)}")

    return train_set, dev_set, test_set


def main(args):
    data = load_json(args.input_file)

    train, dev, test = create_balanced_splits(data, train_size=210, random_state=42)

    save_json(train, f"{args.output_dir}/scitab_train.json", indent=4)
    save_json(dev,   f"{args.output_dir}/scitab_dev.json",   indent=4)
    save_json(test,  f"{args.output_dir}/scitab_test.json",  indent=4)

    # 100-instance training sample
    random.seed(42)
    train_100 = random.sample(train, min(100, len(train)))
    save_json(train_100, f"{args.output_dir}/scitab_train_100.json")
    print("\n=== Label distribution (train set with 100 instances) ===")
    show_label_distribution(train_100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SciTab dataset into balanced splits")
    parser.add_argument("--input_file", required=True,
                        help="Path to the cleaned SciTab data")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save processed data")
    main(parser.parse_args())
