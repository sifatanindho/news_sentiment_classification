import json

def find_best_result(results):
    best_entry = None
    best_score = 0

    for entry in results:
        devtest_rw_accuracy = entry['devtest_rw']['accuracy']
        devtest_rw_roc_auc = entry['devtest_rw']['roc_auc']
        score = (devtest_rw_accuracy + devtest_rw_roc_auc) / 2

        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry

def get_devtest_rw_full_metrics(best_result):
    accuracy = best_result['devtest_rw']['accuracy']
    roc_auc = best_result['devtest_rw']['roc_auc']
    return accuracy, roc_auc

def get_label_specific_metrics(best_result):
    label_specific_metrics = {}
    for label in ['0', '1', '2']:
        label_metrics = best_result['devtest_rw'][label]
        label_specific_metrics[label] = {
            'f1-score': label_metrics['f1-score'],
            'recall': label_metrics['recall'],
            'precision': label_metrics['precision']
        }
    return label_specific_metrics