"""Epistatic Program Selection with frozen language-generated proposals."""

import torch

from .common import ASSETS, DEVICE, read_json, rng


def program_pairs(data, candidate_ids, context_ids, seed):
    """Run fixed language programs on common, source-valid observation states."""
    programs = read_json(ASSETS / "programs.json")
    weights = torch.tensor([p["weights"] for p in programs], device=DEVICE, dtype=torch.float64)
    ranks = torch.tensor([p["rank"] for p in programs], device=DEVICE)
    generator = rng(650300 + seed)
    pool, histories = [], []
    for _ in range(128):
        history = context_ids[
            torch.randperm(len(context_ids), generator=generator, device=DEVICE)[:16]
        ]
        available = candidate_ids[~torch.isin(candidate_ids, history)]
        pool.append(
            available[torch.randperm(len(available), generator=generator, device=DEVICE)[:64]]
        )
        histories.append(history)
    pool, histories = torch.stack(pool), torch.stack(histories)
    px, hx = data["x"][pool], data["x"][histories]
    similarity = torch.exp(-torch.cdist(px, hx).square() / 8)
    density = similarity.sum(-1)
    local = (similarity @ data["eps"][histories].unsqueeze(-1)).squeeze(-1) / (1 + density)
    effects = (data["y"][:, 1:3] - data["y"][:, :1]) / data["scale"]
    observations = torch.stack(
        [
            data["offset"][pool],
            local,
            1 / (1 + density),
            -(effects[:, 0] - effects[:, 1]).abs()[pool],
            (
                (data["positions"][:, 0] - data["positions"][:, 1]).abs().double()
                / (data["positions"].max() + 1)
            )[pool],
        ],
        -1,
    )
    priority = observations @ weights.T
    order = priority.argsort(1, descending=True)
    selected = pool.gather(1, order.gather(1, ranks[None, None].expand(128, 1, -1)).squeeze(1))
    pairs = torch.combinations(torch.arange(len(programs), device=DEVICE), r=2)
    first, second = selected[:, pairs[:, 0]].flatten(), selected[:, pairs[:, 1]].flatten()
    state = torch.arange(128, device=DEVICE)[:, None].expand(-1, len(pairs)).flatten()
    distinct = first != second
    first, second, state = first[distinct], second[distinct], state[distinct]
    choose = torch.randperm(len(first), generator=generator, device=DEVICE)[:2048]
    return {
        "first": first[choose],
        "second": second[choose],
        "history": histories,
        "state": state[choose],
        "candidate_pools": pool,
        "selected_program_candidates": selected,
    }


def expand_pairs(records, ids, seed):
    generator = rng(651100 + seed)
    extra = ids[torch.randint(len(ids), (8192, 2), generator=generator, device=DEVICE)]
    extra = extra[extra[:, 0] != extra[:, 1]]
    return {
        "first": torch.cat([records["first"], extra[:, 0]]),
        "second": torch.cat([records["second"], extra[:, 1]]),
    }
