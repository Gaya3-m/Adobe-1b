import json
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from scipy.stats import spearmanr
import string

def normalize(text):
    if not isinstance(text, str):
        return ""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def align_items(system_output, ground_truth):
    """Create aligned vectors for comparison, with normalized titles"""
    all_items = set()
    for item in ground_truth['extracted_sections']:
        key = (item['document'], normalize(item['section_title']), item['page_number'])
        all_items.add(key)
    for item in system_output['extracted_sections']:
        key = (item['document'], normalize(item['section_title']), item['page_number'])
        all_items.add(key)
    truth_scores = []
    system_scores = []
    for key in all_items:
        # Find in ground truth
        truth_score = 0
        for item in ground_truth['extracted_sections']:
            if (item['document'], normalize(item['section_title']), item['page_number']) == key:
                truth_score = 1
                break
        # Find in system output
        system_score = 0
        for rank, item in enumerate(system_output['extracted_sections']):
            if (item['document'], normalize(item['section_title']), item['page_number']) == key:
                system_score = 1 - (rank / len(system_output['extracted_sections']))
                break
        truth_scores.append(truth_score)
        system_scores.append(system_score)
    return truth_scores, system_scores

def calculate_metrics(system_output, ground_truth):
    y_true, y_pred = align_items(system_output, ground_truth)
    y_pred_binary = [1 if x > 0 else 0 for x in y_pred]
    # Handle empty cases
    if not y_true or not y_pred_binary:
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'spearman_correlation': 0,
            'num_relevant_items': 0,
            'num_selected_items': 0
        }
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'spearman_correlation': spearman_corr if not np.isnan(spearman_corr) else 0,
        'num_relevant_items': sum(y_true),
        'num_selected_items': sum(y_pred_binary)
    }

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <system_output.json> <ground_truth.json>")
        return
    system_output = load_json(sys.argv[1])
    ground_truth = load_json(sys.argv[2])
    if not system_output.get('extracted_sections') or not ground_truth.get('extracted_sections'):
        print("Warning: One of the files has no extracted_sections.")
    metrics = calculate_metrics(system_output, ground_truth)
    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print(f"Spearman Rank Correlation: {metrics['spearman_correlation']:.3f}")
    print(f"\nRelevant Items: {metrics['num_relevant_items']}")
    print(f"Selected Items: {metrics['num_selected_items']}")

if __name__ == "__main__":
    main()