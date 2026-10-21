"""Module replacements and release-path preservation evaluation on CUDA."""
import argparse
import copy
import torch

from followup import core
from followup.run import sparse_request, subgroup_scores, foundations
from prosiu.common import rng, write_json, read_json, configuration
from prosiu.epistatic_program_selection import program_pairs, expand_pairs
from prosiu.mutation_preference_optimization import SelectorObjective
from prosiu.selective_epistasis_forgetting import selective_forgetting
from prosiu.experiment import fit_episode

OUTPUT = core.RESEARCH / 'studies/module_study_20261020'
VARIANTS = ['w_o_EPS', 'w_o_MPO', 'w_o_SEF', 'ProSIU']


def uniform_block(ids, size, seed):
    generator = rng(653400 + seed)
    first = torch.randint(len(ids), (size,), generator=generator, device=core.DEVICE)
    offset = torch.randint(1, len(ids), (size,), generator=generator, device=core.DEVICE)
    return {'first': ids[first], 'second': ids[(first + offset) % len(ids)]}


def objectives(ep, variant):
    train = torch.tensor(sorted(ep['req']['retained'] + ep['req']['forgotten']), device=core.DEVICE)
    retain = torch.tensor(ep['req']['retained'], device=core.DEVICE)
    forgotten = torch.tensor(ep['req']['forgotten'], device=core.DEVICE)
    programs, pairs = {}, {}
    for stage, ids in [('before', train), ('after', retain)]:
        programs[stage] = program_pairs(ep['stream'], ids, ids, ep['seed'])
        block = programs[stage]
        if variant == 'w_o_EPS':
            block = uniform_block(ids, len(block['first']), ep['seed'])
        pairs[stage] = expand_pairs(block, ids, ep['seed'])
    affected_survivors = 0
    if variant == 'w_o_SEF':
        keep = torch.isin(pairs['before']['first'], retain) & torch.isin(pairs['before']['second'], retain)
        old = programs['before']
        affected = torch.isin(old['history'][old['state']], forgotten).any(1)
        affected_survivors = int((keep[:len(affected)] & affected).sum())
        pairs['after'] = {k: v[keep] for k, v in pairs['before'].items()}
    ep = dict(ep, heads={h: dict(item) for h, item in ep['heads'].items()})
    for head, item in ep['heads'].items():
        cfg = copy.deepcopy(item['cfg'])
        if variant == 'w_o_MPO':
            cfg.update(classification=0., preference=0., calibration=1., rank_temperature=1., threshold=0.)
        item['cfg'] = cfg
        for stage, ids in [('before', train), ('after', retain)]:
            item[stage] = SelectorObjective(item['x'], ep['stream']['eps'], ids, pairs[stage], cfg)
    counts = {stage: {'raw': len(p['first']), 'effective': int((ep['stream']['eps'][p['first']] != ep['stream']['eps'][p['second']]).sum())}
              for stage, p in pairs.items()}
    return ep, {'pair_counts': counts, 'surviving_affected_program_histories': affected_survivors}


def ablation(data, truth, maps, source, assay, seed):
    base = core.episode(data, truth, maps, source, assay, seed)
    for variant in VARIANTS:
        ep, composition = objectives(base, variant)
        parameters, diagnostics = {}, {}
        for head, item in ep['heads'].items():
            original, before = item['before'].fit(iterations=600)
            if variant == 'w_o_SEF':
                parameters[head], after = item['after'].fit(original, iterations=600)
            else:
                parameters[head], after = selective_forgetting(item['x'], ep['req'], item['after'], original)
            diagnostics[head] = {'original': before, 'update': after, 'configuration': item['cfg']}
        metrics, _ = core.evaluate(ep, parameters)
        folder = OUTPUT / 'ablation' / assay / f'seed_{seed}' / variant
        record = {'experiment': 'E05', 'assay': assay, 'seed': seed, 'variant': variant,
                  'metrics': metrics, 'diagnostics': diagnostics, 'request': ep['req'], **composition}
        write_json(folder / 'result.json', record)
        torch.save({h: p.cpu() for h, p in parameters.items()}, folder / 'parameters.pt')
        print(f'ablation {assay} seed={seed} {variant}: {metrics}', flush=True)


def preservation(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed, req=sparse_request(source, seed))
    original, reference, _ = foundations(ep)
    folder = OUTPUT / 'references' / assay / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({'original': {h:p.cpu() for h,p in original.items()},
                'reference': {h:p.cpu() for h,p in reference.items()}}, folder / 'parameters.pt')
    for method in ['TARF', 'Entanglement', 'ProSIU']:
        parameters, diagnostics = {}, {}
        if method == 'ProSIU':
            _, checkpoint, diagnostics = fit_episode(ep['stream'], ep['req'], maps, configuration(assay))
            parameters = {h:p['theta'].to(core.DEVICE) for h,p in checkpoint.items()}
        else:
            setting = read_json(core.RESEARCH / 'studies/epistasis_correction/frozen' / method / f'{assay}.json')['configuration']
            for head, item in ep['heads'].items():
                parameters[head], diagnostics[head] = core.update(method, item['after'], item['revoked'], original[head], setting, seed)
                diagnostics[head]['configuration'] = setting
        core.save_result('E03', ep, method, parameters, reference, diagnostics,
                         {'subgroups': subgroup_scores(ep, parameters, original, reference, source), 'request': ep['req']})


def main(seeds):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    core.OUTPUT = OUTPUT
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        for seed in seeds:
            ablation(data, truth, maps, source, assay, seed)
            preservation(data, truth, maps, source, assay, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    main(parser.parse_args().seeds)
