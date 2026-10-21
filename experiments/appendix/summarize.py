"""GPU paired inference and deterministic case selection from complete requests."""
import argparse
import json
from collections import defaultdict

import torch

from .common import *


def confirmation_statistics():
    records = []
    for path in sorted((OUTPUT / 'confirmation/E01').glob('*/seed_*/*/result.json')):
        records.append(read_json(path))
    index = {(r['assay'], r['seed'], r['variant']): r for r in records}
    rows = []
    diffs = []
    for learner in ['final', 'common']:
        for assay in core.ASSAYS:
            for method in core.GENERAL:
                for metric in ['CA', 'RA', 'TA', 'PA']:
                    ours = torch.tensor([index[(assay, s, learner + '_ProSIU')]['metrics'][metric] for s in range(100, 120)], device=DEVICE)
                    other = torch.tensor([index[(assay, s, learner + '_' + method)]['metrics'][metric] for s in range(100, 120)], device=DEVICE)
                    d = ours - other
                    generator = rng(851300)
                    samples = torch.randint(20, (20000, 20), generator=generator, device=DEVICE)
                    boot = d[samples].mean(1)
                    ci = torch.quantile(boot, torch.tensor([.025, .975], device=DEVICE))
                    signs = 2 * torch.randint(2, (100000, 20), generator=generator, device=DEVICE) - 1
                    p = (1 + ((signs * d).mean(1).abs() >= d.mean().abs()).sum()) / 100001
                    rows.append({'learner': learner, 'assay': assay, 'baseline': method, 'metric': metric,
                        'mean_ProSIU': ours.mean().item(), 'sd_ProSIU': ours.std().item(),
                        'mean_baseline': other.mean().item(), 'sd_baseline': other.std().item(),
                        'paired_difference': d.mean().item(), 'ci_lower': ci[0].item(), 'ci_upper': ci[1].item(),
                        'p_raw': p.item(), 'requests': 20, 'seed_differences': d.cpu().tolist()})
                    diffs.append(p)
    p = torch.stack(diffs)
    order = p.argsort()
    adjusted = (p[order] * torch.arange(len(p), 0, -1, device=DEVICE)).cummax(0).values.clamp(max=1)
    for idx, value in zip(order.cpu().tolist(), adjusted.cpu().tolist()):
        rows[idx]['p_holm'] = value
    write_json(OUTPUT / 'E13/results.json', {'experiment': 'E13', 'rows': rows,
        'bootstrap_draws': 20000, 'paired_sign_draws': 100000,
        'family': 'all baseline-by-assay-by-metric comparisons across both learners'})


def cases():
    rows = []
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        candidates = []
        for seed in range(100, 120):
            root = OUTPUT / 'confirmation/E01' / assay / f'seed_{seed}'
            ours = read_json(root / 'final_ProSIU/result.json')['metrics']
            controls = [read_json(root / ('final_' + m) / 'result.json')['metrics'] for m in core.GENERAL]
            difference = sum(ours[k] - max(r[k] for r in controls) for k in ['CA', 'RA', 'TA', 'PA']) / 4
            candidates.append((difference, seed))
        candidates.sort()
        for label, index in [('Lower quartile', 4), ('Median', 9), ('Upper quartile', 14)]:
            difference, seed = candidates[index]
            ep = core.episode(data, truth, maps, source, assay, seed)
            root = OUTPUT / 'confirmation/E01' / assay / f'seed_{seed}'
            parameters = {name: torch.load(root / ('final_' + name) / 'parameters.pt',
                map_location=DEVICE, weights_only=True) for name in ['Original', 'ProSIU', 'Retrain']}
            predictions = {name: core.predictions(ep, p) for name, p in parameters.items()}
            targets = sorted(ep['req']['forgotten'])
            selected = [targets[round((len(targets) - 1) * q)] for q in [.25, .5, .75]]
            for state, target in enumerate(selected):
                raw = source['rows'][target]
                mutations = [source['wt'][p] + str(p + 1) + raw['q'][3][p] for p in raw['positions']]
                proposed = ep['evaluation']['selected_program_candidates'][state]
                actions = {}
                for name, (score, utility) in predictions.items():
                    chosen = proposed[utility[proposed].argmax()].item()
                    actions[name] = {'candidate_id': chosen, 'clean_interaction': truth[chosen].item(),
                                     'utility': utility[chosen].item()}
                rows.append({'assay': assay, 'seed': seed, 'selection': label, 'request_effect_pp': difference,
                    'target_id': target, 'mutations': mutations, 'measurements': raw['y'],
                    'clean_interaction': truth[target].item(), 'observed_interaction': ep['stream']['eps'][target].item(),
                    'predictions': {name: {'sign_score': score[target].item(), 'preference_score': utility[target].item()}
                                    for name, (score, utility) in predictions.items()},
                    'program_state': state, 'selected_actions': actions,
                    'original_metrics': read_json(root / 'final_Original/result.json')['metrics'],
                    'ProSIU_metrics': read_json(root / 'final_ProSIU/result.json')['metrics']})
    write_json(OUTPUT / 'E40/results.json', {'experiment': 'E40', 'rows': rows,
        'selection': 'For each protein, sort 20 independent requests by four-metric mean difference to the best baseline in each metric. Select ranks 5, 10 and 15; target identities at the 25th, 50th and 75th percentiles of each withdrawn identity list.'})


def export():
    records = []
    for experiment in [f'E{i}' for i in range(13, 42)]:
        folder = OUTPUT / experiment
        for path in sorted(folder.rglob('*.json')):
            if path.name not in ['result.json', 'results.json']:
                continue
            value = read_json(path)
            value['relative_path'] = str(path.relative_to(OUTPUT))
            records.append(value)
    confirmation = [read_json(p) for p in sorted((OUTPUT / 'confirmation').glob('E*/*/seed_*/*/result.json'))]
    write_json(OUTPUT / 'exports/all_results.json', {'protocol': read_json(Path(__file__).with_name('protocol.json')),
               'records': records, 'confirmation': confirmation,
               'generated_libraries': [read_json(p) for p in sorted((OUTPUT / 'libraries').glob('*/*.json'))],
               'method_settings': {assay: configuration(assay) for assay in core.ASSAYS},
               'baseline_settings': {method: {assay: read_json(core.RESEARCH / 'studies/epistasis_correction/frozen' /
                    method / f'{assay}.json') for assay in core.ASSAYS}
                    for method in core.GENERAL},
               'encoder_costs': read_json(OUTPUT / 'encoder_costs.json') if (OUTPUT / 'encoder_costs.json').exists() else []})
    counts = defaultdict(int)
    for row in records:
        counts[row['experiment']] += 1
    write_json(OUTPUT / 'exports/status.json', dict(counts))
    print(dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    if not args.export_only:
        confirmation_statistics()
        cases()
    export()
