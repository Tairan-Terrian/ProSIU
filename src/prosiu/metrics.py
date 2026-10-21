"""The four fixed evaluation metrics, expressed in percentage points."""

import torch

from .common import DEVICE


def balanced_accuracy(logits, labels):
    correct = (logits >= 0) == labels.bool()
    return 50 * (correct[labels == 0].double().mean() + correct[labels == 1].double().mean()).item()


def evaluate_scores(score, truth, req, test_ids, records, utility=None):
    labels = (truth > 0).double()
    groups = {
        "CA": torch.tensor(req["forgotten"], device=DEVICE),
        "RA": torch.tensor(req["retained"], device=DEVICE),
        "TA": test_ids,
    }
    metrics = {name: balanced_accuracy(score[ids], labels[ids]) for name, ids in groups.items()}
    difference = truth[records["first"]] - truth[records["second"]]
    utility = score if utility is None else utility
    predicted = utility[records["first"]] - utility[records["second"]]
    valid = difference != 0
    credit = ((predicted > 0) == (difference > 0)).double()
    credit[predicted == 0] = 0.5
    metrics["PA"] = 100 * credit[valid].mean().item()
    return metrics, {
        "interaction": score.cpu(),
        "program_score": predicted.cpu(),
        "program_truth": difference.cpu(),
        "program_first": records["first"].cpu(),
        "program_second": records["second"].cpu(),
    }
