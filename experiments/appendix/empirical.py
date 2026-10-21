"""Predeclared request, evidence, representation and objective interventions."""
import argparse
import copy
import itertools
import math

import torch

from .common import *
from followup.programs import pairs_from_states
from prosiu.features import build_context_features


def e18(data, truth, maps, source, assay, seed):
    first = request_for(source, seed, count=1)
    grouped = request_for(source, seed, fraction=.05)
    cases = [('One record', request_for(source, seed, count=1, granularity='record')),
             ('One position pair', first), ('Position groups', grouped),
             ('Random records', request_for(source, seed, count=len(grouped['forgotten']), granularity='random'))]
    for name, req in cases:
        ep = core.episode(data, truth, maps, source, assay, seed, req=req)
        comparison('E18', ep, name, extra={'requested_granularity': name})


def e19(data, truth, maps, source, assay, seed):
    for fraction in [.01, .05, .1, .2, .4, .6]:
        req = request_for(source, seed, fraction)
        ep = core.episode(data, truth, maps, source, assay, seed, req=req)
        comparison('E19', ep, f'{fraction:.2f}', extra={'fraction': fraction, 'actual_fraction': req['affected_fraction']})


def e20(data, truth, maps, source, assay, seed):
    for amplitude in [0., .25, .5, 1., 2.]:
        req = request_for(source, seed)
        req['shifts'] = [x * amplitude for x in req['shifts']]
        ep = core.episode(data, truth, maps, source, assay, seed, req=req)
        comparison('E20', ep, f'{amplitude:.2f}', extra={'amplitude': amplitude})


def e21(data, truth, maps, source, assay, seed):
    for kind in ['Positive', 'Negative', 'Mixed', 'Gaussian', 'Heteroscedastic']:
        req = request_for(source, seed)
        f = tensors(req['forgotten'])
        shift = torch.tensor(req['shifts'], device=DEVICE)
        if kind == 'Positive':
            shift = torch.ones_like(shift)
        elif kind == 'Negative':
            shift = -torch.ones_like(shift)
        elif kind in ['Gaussian', 'Heteroscedastic']:
            shift = torch.randn(len(f), device=DEVICE, generator=rng(852100 + seed))
            if kind == 'Heteroscedastic':
                distance = (data['positions'][f, 1] - data['positions'][f, 0]).double()
                scales = .5 + distance / len(source['wt'])
                shift = shift * scales / scales.square().mean().sqrt()
        req['shifts'] = shift.cpu().tolist()
        ep = core.episode(data, truth, maps, source, assay, seed, req=req)
        comparison('E21', ep, kind, extra={'offset_kind': kind})


def e23(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed)
    train, retain, forgotten = [tensors(x) for x in [
        sorted(ep['req']['retained'] + ep['req']['forgotten']), ep['req']['retained'], ep['req']['forgotten']]]
    old = program_pairs(ep['stream'], train, train, seed)
    new = program_pairs(ep['stream'], retain, retain, seed)
    replay = pairs_from_states(ep['stream'], old, retained=retain, seed=seed)
    for multiplicity in [1, 2, 4, 8]:
        def amplified(records, ids):
            replicated = {k: records[k].repeat(multiplicity) for k in ['first', 'second']}
            return expand_pairs(replicated, ids, seed)
        old_pairs = amplified(old, train)
        new_pairs = amplified(new, retain)
        full = replace_objectives(ep, old_pairs, new_pairs)
        original, reference, _ = foundations(full)
        live = ~torch.isin(old_pairs['first'], forgotten) & ~torch.isin(old_pairs['second'], forgotten)
        versions = {'Raw records': old_pairs, 'Endpoint removal': {k: v[live] for k, v in old_pairs.items()},
                    'State rebuilding': amplified(replay, retain), 'Full reconstruction': new_pairs}
        count = min(len(v['first']) for v in versions.values())
        for matched in [False, True]:
            for name, pairs in versions.items():
                condition = f'{multiplicity}x_' + ('matched' if matched else 'natural')
                if output_file('E23', assay, seed, condition, name).exists():
                    continue
                if matched:
                    order = torch.randperm(len(pairs['first']), device=DEVICE, generator=rng(852300 + seed))[:count]
                    pairs = {k: v[order] for k, v in pairs.items()}
                variant = replace_objectives(ep, old_pairs, pairs)
                params, details = {}, {}
                for head, item in variant['heads'].items():
                    starts, geometry = core.initializations(item['x'], ep['req'], item['after'], original[head])
                    params[head], details[head] = core.fit(item['after'], starts['ProSIU'])
                record('E23', full, condition, name, params, reference, details,
                       {'multiplicity': multiplicity, 'matched': matched, 'comparison_count': len(pairs['first'])})


def e25(data, truth, maps, source, assay, seed):
    req = request_for(source, seed)
    base = core.episode(data, truth, maps, source, assay, seed, req=req)
    old = saved_main(assay, seed)
    for fraction in [0., .25, .5, 1.]:
        f = tensors(req['forgotten'])
        f = f[torch.randperm(len(f), generator=rng(852500 + seed), device=DEVICE)]
        restored = f[:round(len(f) * fraction)]
        active = torch.cat([tensors(req['retained']), restored]).sort().values
        pairs = expand_pairs(program_pairs(dict(base['stream'], eps=truth), active, active, seed), active, seed)
        params, ref, details = {}, {}, {}
        for head, item in base['heads'].items():
            objective = core.ExtendedObjective(item['x'], truth, active, pairs, item['cfg'])
            ref[head], _ = core.fit(objective, strict=True, iterations=1000)
            params[head], details[head] = core.fit(objective, old[head])
        condition = f'{fraction:.2f}'
        reference_ep = dict(base, heads={h: dict(v, after=core.ExtendedObjective(v['x'], truth, active, pairs, v['cfg']))
                                         for h, v in base['heads'].items()})
        for name, p in [('Frozen withdrawal model', old), ('Incremental ProSIU', params), ('Retained retraining', ref)]:
            record('E25', reference_ep, condition, name, p, ref, details,
                   {'restored_fraction': fraction, 'restored_count': len(restored)})


def e26(data, truth, maps, source, assay, seed):
    req = request_for(source, seed)
    base = core.episode(data, truth, maps, source, assay, seed, req=req)
    retained = tensors(req['retained'])
    retained = retained[torch.randperm(len(retained), generator=rng(852600 + seed), device=DEVICE)]
    for fraction in [.1, .25, .5, 1.]:
        selected = retained[:round(len(retained) * fraction)].cpu().tolist()
        sparse = dict(req, retained=sorted(selected))
        ep = core.episode(data, truth, maps, source, assay, seed, req=sparse)
        comparison('E26', ep, f'{fraction:.2f}', evaluation_ep=common_evaluation(ep, base),
                   extra={'retained_fraction': fraction, 'used_retained_count': len(selected)})


def e28(data, truth, maps, source, assay, seed):
    anchor_ids = {a['sequence']: i for i, a in enumerate(source['anchors'])}
    row_ids = torch.tensor([[anchor_ids[q] for q in row['q'][:3]] for row in source['rows']], device=DEVICE)
    noise = torch.randn(len(anchor_ids), generator=rng(852800 + seed), device=DEVICE)
    noise[anchor_ids[source['wt']]] = 0.
    for sigma in [0., .05, .1, .2]:
        if completed('E28', assay, seed, f'{sigma:.2f}'):
            continue
        folder = OUTPUT / 'noisy_anchors' / assay / f'seed_{seed}' / f'{sigma:.2f}'
        perturbation = (sigma * data['scale'] * noise[row_ids]).cpu().tolist()
        noisy_source = copy.deepcopy(source)
        for row, delta in zip(noisy_source['rows'], perturbation):
            row['y'][:3] = [v + d for v, d in zip(row['y'][:3], delta)]
        write_json(folder / 'processed' / f'{assay}.json', noisy_source)
        torch.set_default_dtype(torch.float32)
        build_context_features(folder, [assay])
        torch.set_default_dtype(torch.float64)
        if assay == 'amacGFP':
            link = folder / 'features/amacGFP_language.pt'
            if not link.exists():
                link.symlink_to(core.DATA / 'features/amacGFP_language.pt')
        noisy, noisy_truth, noisy_maps, _ = core.dataset(assay, folder)
        shift = truth - noisy_truth
        req = request_for(source, seed)
        stream = observed(noisy, noisy_truth, req)
        ep = core.episode(noisy, truth, noisy_maps, source, assay, seed, req=req, stream_override=stream)
        comparison('E28', ep, f'{sigma:.2f}', extra={'anchor_noise_sd': sigma,
                   'contrast_rmse': shift.square().mean().sqrt().item(), 'intervention': 'controlled anchor perturbation'})


def e30(data, truth, maps, source, assay, seed):
    actual = request_for(source, seed)
    stream = observed(data, truth, actual)
    f, r = tensors(actual['forgotten']), tensors(actual['retained'])
    f = f[torch.randperm(len(f), generator=rng(853000 + seed), device=DEVICE)]
    r = r[torch.randperm(len(r), generator=rng(853001 + seed), device=DEVICE)]
    train = sorted(actual['retained'] + actual['forgotten'])
    for recall, false_positive in itertools.product([.5, .75, 1.], [0., .1, .25]):
        selected = torch.cat([f[:round(len(f) * recall)], r[:round(len(f) * false_positive)]])
        req = make_request(source, seed, selected.cpu().tolist(), [0.] * len(selected), train)
        ep = core.episode(data, truth, maps, source, assay, seed, req=req, stream_override=stream)
        comparison('E30', ep, f'{recall:.2f}_{false_positive:.2f}',
                   extra={'request_recall': recall, 'false_positive_fraction': false_positive})


def e31(data, truth, maps, source, assay, seed):
    base = core.episode(data, truth, maps, source, assay, seed)
    new = data['partitions']['dev_train']
    new = new[torch.randperm(len(new), generator=rng(853100 + seed), device=DEVICE)[:512]]
    for batches in [1, 2, 4]:
        current = saved_main(assay, seed)
        retain = tensors(base['req']['retained'])
        for batch, fresh in enumerate(new.tensor_split(batches)):
            retain = torch.cat([retain, fresh]).sort().values
            stream = dict(base['stream'], eps=base['stream']['eps'].clone())
            stream['eps'][retain] = truth[retain]
            pairs = expand_pairs(program_pairs(stream, retain, retain, seed), retain, seed)
            heads, reference, details = {}, {}, {}
            for head, item in base['heads'].items():
                objective = core.ExtendedObjective(item['x'], stream['eps'], retain, pairs, item['cfg'])
                heads[head] = dict(item, after=objective)
                reference[head], _ = core.fit(objective, iterations=1000, strict=True)
                current[head], details[head] = core.fit(objective, current[head])
            ep = dict(base, heads=heads)
            for method, params in [('Incremental ProSIU', current), ('Retained retraining', reference)]:
                record('E31', ep, f'B{batches}_step{batch + 1}', method, params, reference, details,
                       {'batches': batches, 'batch': batch + 1, 'new_count': len(retain) - len(base['req']['retained'])})


def e34(data, truth, maps, source, assay, seed):
    anchor_dimension = 11 + 3 * min(192, len(data['partitions']['dev_train']))
    identity_dimension = maps['identity'].shape[1]
    for removal in ['Full', 'Anchor descriptors', 'Identity indicators', 'ESM context', 'Sequence contrast']:
        if removal == 'Sequence contrast' and assay != 'amacGFP':
            continue
        changed = {}
        for name, x in maps.items():
            keep = torch.ones(x.shape[1], device=DEVICE, dtype=torch.bool)
            if removal == 'Anchor descriptors':
                keep[1:anchor_dimension] = False
            elif removal == 'Identity indicators':
                keep[anchor_dimension:identity_dimension] = False
            elif removal == 'ESM context' and name == 'context':
                keep[identity_dimension:] = False
            elif removal == 'Sequence contrast' and name == 'language':
                keep[-128:] = False
            changed[name] = x[:, keep]
        for matched in [False, True]:
            if matched:
                projected = {}
                for name, x in changed.items():
                    projection = torch.randn(x.shape[1], 256, generator=rng(853400), device=DEVICE) / 256 ** .5
                    projected[name] = x @ projection
                use = projected
            else:
                use = changed
            ep = core.episode(data, truth, use, source, assay, seed)
            comparison('E34', ep, removal + ('_matched256' if matched else '_native'),
                       methods=['ProSIU'], extra={'removed': removal, 'dimension_matched': matched,
                       'dimensions': {k: v.shape[1] for k, v in use.items()}})


def e38(data, truth, maps, source, assay, seed):
    grids = {'classification': [0., .25, 1., 4.], 'preference': [.25, .5, 1., 2., 4.],
             'calibration': [0., .25, .5, 1.], 'ridge': [.0001, .001, .01, .1],
             'rank_temperature': [0., .05, .25, 1.]}
    default = configuration(assay)
    for field, values in grids.items():
        for value in values:
            cfg = copy.deepcopy(default)
            for head in cfg:
                if field == 'classification' and head == 'preference':
                    continue
                cfg[head][field] = value
            ep = core.episode(data, truth, maps, source, assay, seed, config=cfg)
            comparison('E38', ep, f'{field}_{value:g}', methods=['ProSIU'],
                       extra={'parameter': field, 'value': value, 'configuration': cfg})


FUNCTIONS = {f'E{i}': globals()[f'e{i}'] for i in [18, 19, 20, 21, 23, 25, 26, 28, 30, 31, 34, 38]}


def main(experiments, seeds, assays):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    for assay in assays:
        data, truth, maps, source = core.dataset(assay)
        for seed in seeds:
            for experiment in experiments:
                marker = OUTPUT / experiment / assay / f'seed_{seed}' / 'complete.json'
                if marker.exists():
                    continue
                FUNCTIONS[experiment](data, truth, maps, source, assay, seed)
                write_json(marker, {'experiment': experiment, 'assay': assay, 'seed': seed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=list(FUNCTIONS))
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--assays', nargs='+', default=core.ASSAYS)
    args = parser.parse_args()
    main(args.experiments, args.seeds, args.assays)
