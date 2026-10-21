"""Shared CUDA execution for the appendix, isolated from the fixed main results."""
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from followup import core
from followup.run import foundations
from prosiu.common import DEVICE, configuration, read_json, write_json, rng
from prosiu.data import observed, request
from prosiu.epistatic_program_selection import program_pairs, expand_pairs

OUTPUT = core.RESEARCH / 'studies/appendix_20261020'
MAIN = core.RESEARCH / 'studies/followup_main12'
METHODS = ['SCRUB', 'TARF', 'Entanglement', 'ProSIU']


def tensors(ids):
    return torch.tensor(ids, device=DEVICE, dtype=torch.long)


def request_for(source, seed, fraction=.2, granularity='groups', count=None):
    groups = defaultdict(list)
    for i, row in enumerate(source['rows']):
        if row['partition'] == 'train':
            groups[tuple(row['positions'])].append(i)
    randomizer = random.Random(650100 + seed)
    keys = sorted(groups)
    randomizer.shuffle(keys)
    train = sorted(i for indices in groups.values() for i in indices)
    target = math.ceil(len(train) * fraction) if count is None else count
    forgotten, shifts = [], []
    for key in keys:
        direction = randomizer.choice([-1., 1.])
        forgotten.extend(groups[key])
        shifts.extend([direction] * len(groups[key]))
        if len(forgotten) >= target:
            break
    if granularity in ['record', 'random']:
        candidates = list(train)
        randomizer.shuffle(candidates)
        forgotten = candidates[:target]
        shifts = [randomizer.choice([-1., 1.]) for _ in forgotten]
    return make_request(source, seed, forgotten, shifts, train)


def make_request(source, seed, forgotten, shifts, train):
    focused = {p for i in forgotten for p in source['rows'][i]['positions']}
    retained = sorted(set(train) - set(forgotten))
    return {'seed': seed, 'partition': 'train', 'forgotten': list(forgotten), 'retained': retained,
            'shifts': list(shifts), 'adjacent': [i for i, r in enumerate(source['rows'])
                                               if any(p in focused for p in r['positions'])],
            'affected_fraction': len(forgotten) / len(train)}


def output_file(experiment, assay, seed, condition, method):
    return OUTPUT / experiment / assay / f'seed_{seed}' / condition / method / 'result.json'


def completed(experiment, assay, seed, condition, methods=METHODS):
    return all(output_file(experiment, assay, seed, condition, m).exists() for m in methods)


def record(experiment, ep, condition, method, params, reference, diagnostics, extra=None):
    metrics, equivalence = core.evaluate(ep, params, reference)
    score, utility = core.predictions(ep, params)
    if experiment == 'E18':
        ids = tensors(ep['req']['forgotten'])
        metrics['CA'] = 100 * ((score[ids] >= 0) == (ep['truth'][ids] > 0)).double().mean().item()
    path = output_file(experiment, ep['assay'], ep['seed'], condition, method)
    result = {'experiment': experiment, 'assay': ep['assay'], 'seed': ep['seed'],
              'condition': condition, 'method': method, 'metrics': metrics,
              'equivalence': equivalence, 'diagnostics': diagnostics,
              'counts': {k: len(ep['req'][k]) for k in ['retained', 'forgotten']}}
    if extra:
        result.update(extra)
    write_json(path, result)
    torch.save({'parameters': {h: v.cpu() for h, v in params.items()},
                'score': score.cpu(), 'utility': utility.cpu(),
                'evaluation': {k: v.cpu() for k, v in ep['evaluation'].items()},
                'test_ids': ep['data']['partitions']['test'].cpu(),
                'request': ep['req']}, path.with_name('predictions.pt'))
    print(json.dumps({k: result[k] for k in ['experiment', 'assay', 'seed', 'condition', 'method', 'metrics']}), flush=True)
    return result


def comparison(experiment, ep, condition, methods=METHODS, evaluation_ep=None, extra=None,
               original=None, reference=None):
    if completed(experiment, ep['assay'], ep['seed'], condition, methods):
        return
    if original is None:
        original, reference, _ = foundations(ep)
    results = []
    for method in methods:
        path = output_file(experiment, ep['assay'], ep['seed'], condition, method)
        if path.exists():
            continue
        params, diagnostics = {}, {}
        for head, item in ep['heads'].items():
            start = core.timed()
            if method == 'ProSIU':
                starts, geometry = core.initializations(item['x'], ep['req'], item['after'], original[head])
                params[head], info = core.fit(item['after'], starts['ProSIU'])
                info['geometry'] = geometry
            elif method == 'Retained fit':
                params[head], info = core.fit(item['after'], original[head])
            elif method == 'Retained retraining':
                params[head], info = core.fit(item['after'], strict=True, iterations=1000)
            else:
                setting = read_json(core.RESEARCH / 'studies/epistasis_correction/frozen' / method / f"{ep['assay']}.json")['configuration']
                params[head], info = core.update(method, item['after'], item['revoked'], original[head], setting, ep['seed'])
            info['seconds'] = core.timed() - start
            diagnostics[head] = info
        evaluation = ep if evaluation_ep is None else evaluation_ep
        results.append(record(experiment, evaluation, condition, method, params, reference, diagnostics, extra))
    return results


def saved_main(assay, seed, method='ProSIU'):
    return torch.load(MAIN / 'E01' / assay / f'seed_{seed}' / ('final_' + method) / 'parameters.pt',
                      map_location=DEVICE, weights_only=True)


def common_evaluation(ep, common):
    return dict(ep, req=common['req'], evaluation=common['evaluation'])


def replace_objectives(ep, old_pairs, new_pairs, configs=None):
    ep = dict(ep, heads={h: dict(item) for h, item in ep['heads'].items()})
    train = tensors(sorted(ep['req']['retained'] + ep['req']['forgotten']))
    retain = tensors(ep['req']['retained'])
    forgotten = tensors(ep['req']['forgotten'])
    affected = torch.isin(old_pairs['first'], forgotten) | torch.isin(old_pairs['second'], forgotten)
    revoked = {k: old_pairs[k][affected] for k in ['first', 'second']}
    for head, item in ep['heads'].items():
        cfg = item['cfg'] if configs is None else configs[head]
        item['cfg'] = cfg
        for name, ids, pairs in [('before', train, old_pairs), ('after', retain, new_pairs), ('revoked', forgotten, revoked)]:
            item[name] = core.ExtendedObjective(item['x'], ep['stream']['eps'], ids, pairs, cfg)
    return ep


def write_diagnostic(experiment, assay, seed, rows, extra=None):
    result = {'experiment': experiment, 'assay': assay, 'seed': seed, 'rows': rows}
    if extra:
        result.update(extra)
    write_json(OUTPUT / experiment / assay / f'seed_{seed}' / 'result.json', result)
    print(experiment, assay, seed, len(rows), 'diagnostic rows', flush=True)
