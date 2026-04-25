"""
Utility functions for data preprocessing
"""

import json
import re
from collections import Counter


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path, indent=4):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def show_label_distribution(data_or_path):
    """
    Show the label distribution of a dataset.

    Args:
        data_or_path: Either a list of dataset instances or a path to a JSON file
    """
    if isinstance(data_or_path, str):
        data = load_json(data_or_path)
    else:
        data = data_or_path

    labels = [instance['label'] for instance in data]
    label_counts = Counter(labels)
    total = len(data)

    print(f"Label Distribution (Total: {total} instances)")
    print("-" * 50)
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count / total * 100:.1f}%)")


def html_table_to_markdown(html_string):
    """
    Convert an HTML table to a markdown-like format with || delimiters.

    Args:
        html_string: HTML string containing a table

    Returns:
        str: Table formatted as rows of "|| cell1 | cell2 | ... ||"
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')
    if not table:
        return ""

    rows = []
    for row in table.find_all('tr'):
        cells = []
        for cell in row.find_all(['th', 'td']):
            text = cell.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            cells.append(text)
        if cells:
            rows.append("|| " + " | ".join(cells) + " ||")

    return "\n".join(rows)
