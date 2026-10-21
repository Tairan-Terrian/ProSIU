"""GPU diagnostics for withdrawal scope and hidden program-history dependence."""
from collections import defaultdict
from pathlib import Path
import math
import random

import torch

from prosiu.common import ASSAYS, DEVICE, SEEDS, read_json, rng, write_json
from prosiu.data import load, observed, request
from prosiu.epistatic_program_selection import program_pairs
from followup.programs import proposals

ROOT = Path("/media/yjz/htr/ProSIU-main")
OUTPUT = Path("/media/yjz/htr/ProSIU/studies/motivation_20261020")
FRACTIONS = [.05, .10, .15, .20]
HISTORY_LENGTHS = [2, 4, 8, 16]


def nested_request(rows, seed, fraction):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["partition"] == "train":
            groups[tuple(row["positions"])].append(i)
    generator = random.Random(650100 + seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    size = sum(len(v) for v in groups.values())
    forgotten, shifts = [], []
    for key in keys:
        direction = generator.choice([-1., 1.])
        forgotten.extend(groups[key])
        shifts.extend([direction] * len(groups[key]))
        if len(forgotten) >= math.ceil(fraction * size):
            break
    train = sorted(i for values in groups.values() for i in values)
    retained = sorted(set(train) - set(forgotten))
    return {"seed": seed, "retained": retained, "forgotten": forgotten,
            "shifts": shifts}


def mutation_table(source):
    wt = source["wt"]
    identity = {}
    anchors = []
    for anchor in source["anchors"]:
        changed = [(p, aa) for p, aa in enumerate(anchor["sequence"])
                   if aa != wt[p]]
        if changed:
            identity[changed[0]] = len(identity)
            anchors.append(identity[changed[0]])
    pairs = [[identity[(p, row["q"][3][p])] for p in row["positions"]]
             for row in source["rows"]]
    return (torch.tensor(pairs, device=DEVICE),
            torch.tensor(anchors, device=DEVICE))


def scope_diagnostic(source, assay, seed, mutations, anchors):
    rows = []
    for fraction in FRACTIONS:
        req = nested_request(source["rows"], seed, fraction)
        if fraction == .20:
            main = request(source["rows"], seed)
            assert req["forgotten"] == main["forgotten"]
            assert req["retained"] == main["retained"]
            assert req["shifts"] == main["shifts"]
        forgotten = torch.tensor(req["forgotten"], device=DEVICE)
        retained = torch.tensor(req["retained"], device=DEVICE)
        components = mutations[forgotten].unique()
        removed_anchors = torch.isin(anchors, components).sum().item()
        removed_pairs = torch.isin(mutations[retained], components).any(1).sum().item()
        rows.append({
            "assay": assay, "seed": seed, "requested_fraction": fraction,
            "forgotten_count": len(forgotten), "retained_count": len(retained),
            "anchor_count": len(anchors), "removed_anchors": removed_anchors,
            "removed_retained_pairs": removed_pairs,
            "anchor_removal_percent": 100 * removed_anchors / len(anchors),
            "pair_removal_percent": 100 * removed_pairs / len(retained),
        })
    return rows


def replay_diagnostic(data, truth, source, assay, seed):
    req = request(source["rows"], seed)
    stream = observed(data, truth, req)
    training = torch.tensor(sorted(req["retained"] + req["forgotten"]), device=DEVICE)
    retained = torch.tensor(req["retained"], device=DEVICE)
    forgotten = torch.tensor(req["forgotten"], device=DEVICE)
    original = program_pairs(stream, training, training, seed)
    full_history = original["history"]
    generator = rng(652900 + seed)
    pools = []
    for history in full_history:
        candidates = retained[~torch.isin(retained, history)]
        pools.append(candidates[
            torch.randperm(len(candidates), generator=generator, device=DEVICE)[:64]])
    pools = torch.stack(pools)
    assert not torch.isin(pools, forgotten).any()
    indices = torch.combinations(torch.arange(16, device=DEVICE), r=2)
    records, traces = [], {}
    for length in HISTORY_LENGTHS:
        history = full_history[:, :length]
        valid_history = ~torch.isin(history, forgotten)
        before = proposals(stream, pools, history)
        after = proposals(stream, pools, history, valid_history=valid_history)
        first, second = before[:, indices[:, 0]], before[:, indices[:, 1]]
        new_first, new_second = after[:, indices[:, 0]], after[:, indices[:, 1]]
        old_difference = truth[first] - truth[second]
        new_difference = truth[new_first] - truth[new_second]
        eligible = (first != second) & (old_difference != 0)
        changed = (first != new_first) | (second != new_second)
        dependent = (~valid_history).any(1)[:, None].expand_as(eligible)
        assert not (changed & ~dependent).any()
        n = eligible.sum().item()
        count = (changed & eligible).sum().item()
        common = eligible & (new_difference != 0)
        reversal = ((old_difference * new_difference < 0) & common).sum().item()
        records.append({
            "assay": assay, "seed": seed, "history_length": length,
            "state_count": len(pools), "program_count": 16,
            "eligible_comparisons": n, "changed_comparisons": count,
            "changed_percent": 100 * count / n,
            "dependent_comparisons": (dependent & eligible).sum().item(),
            "changed_proposals": (before != after).sum().item(),
            "total_proposals": before.numel(),
            "reversed_outcome_order": reversal,
            "common_nontied_comparisons": common.sum().item(),
            "collapsed_comparisons": ((new_first == new_second) & eligible).sum().item(),
        })
        traces[str(length)] = {
            "history": history.cpu(), "valid_history": valid_history.cpu(),
            "selected_before": before.cpu(), "selected_after": after.cpu(),
            "eligible": eligible.cpu(), "changed": changed.cpu(),
        }
    folder = OUTPUT / assay / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"pools": pools.cpu(), "replays": traces}, folder / "replay.pt")
    return records


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    scope, replay = [], []
    for assay in ASSAYS:
        source = read_json(ROOT / "data/processed" / f"{assay}.json")
        data, truth = load(assay, ROOT / "data")
        mutations, anchors = mutation_table(source)
        for seed in SEEDS:
            a = scope_diagnostic(source, assay, seed, mutations, anchors)
            b = replay_diagnostic(data, truth, source, assay, seed)
            scope.extend(a)
            replay.extend(b)
            write_json(OUTPUT / assay / f"seed_{seed}" / "result.json",
                       {"scope": a, "replay": b})
            print(assay, seed, {
                "anchor_removal_20": a[-1]["anchor_removal_percent"],
                "pair_removal_20": a[-1]["pair_removal_percent"],
                "changed_comparisons_h16": b[-1]["changed_percent"],
            }, flush=True)
    write_json(OUTPUT / "results.json", {
        "assays": ASSAYS, "seeds": SEEDS, "fractions": FRACTIONS,
        "history_lengths": HISTORY_LENGTHS, "scope": scope, "replay": replay,
        "device": torch.cuda.get_device_name(),
        "protocol": (Path(__file__).parent / "protocol.txt").read_text(),
    })


if __name__ == "__main__":
    main()
