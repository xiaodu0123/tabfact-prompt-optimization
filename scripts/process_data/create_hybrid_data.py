"""
Create hybrid training data by combining instances from PubHealthTab, TabFact,
and SciTab datasets.
"""

import argparse
import random
import os

from utils import load_json, save_json, show_label_distribution


def main(args):
    pubhealth_data = load_json(args.pubhealth_train)
    tabfact_data   = load_json(args.tabfact_train)
    scitab_data    = load_json(args.scitab_train)

    random.seed(42)
    hybrid_train_data = (
        random.sample(pubhealth_data, 40) 
        + random.sample(tabfact_data, 20) 
        + random.sample(scitab_data, 40)
    )

    random.shuffle(hybrid_train_data)
    show_label_distribution(hybrid_train_data)
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(hybrid_train_data, f"{args.output_dir}/hybrid_train.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create hybrid training datasets")
    parser.add_argument("--pubhealth_train", required=True,
                        help="Path to processed PubHealthTab training JSON")
    parser.add_argument("--tabfact_train", required=True,
                        help="Path to processed TabFact training JSON")
    parser.add_argument("--scitab_train", required=True,
                        help="Path to processed SciTab training JSON")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write hybrid output files")
    main(parser.parse_args())
