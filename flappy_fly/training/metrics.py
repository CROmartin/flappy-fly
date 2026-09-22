import numpy as np


def classification(labels, probability, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(probability) >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {'samples': len(labels), 'precision_flap': precision, 'recall_flap': recall,
            'f1_flap': 2 * precision * recall / max(1e-12, precision + recall),
            'accuracy': (tp + tn) / max(1, len(labels)), 'confusion_matrix': [[tn, fp], [fn, tp]],
            'threshold': threshold}


def episode_metrics(rows):
    scores = np.asarray([r['score'] for r in rows])
    times = np.asarray([r['survival_time'] for r in rows])
    return {'episodes': len(rows), 'mean_score': float(scores.mean()), 'median_score': float(np.median(scores)),
            'max_score': int(scores.max()), 'mean_survival_time': float(times.mean()),
            'median_survival_time': float(np.median(times)), 'pipes_passed': int(scores.sum()),
            'percent_passing_one': float(100 * (scores >= 1).mean()),
            'truncated_episodes': sum(r['truncated'] for r in rows)}
