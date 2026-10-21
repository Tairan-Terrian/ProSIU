"""Reconstruct saved task metrics and check the complete prescribed result inventory."""
from collections import Counter
import math

import torch

from .common import *
from prosiu.metrics import evaluate_scores

EXPECTED = {'E13':1,'E14':15,'E15':15,'E16':15,'E17':15,'E18':240,'E19':360,'E20':300,
            'E21':300,'E22':75,'E23':480,'E24':15,'E25':180,'E26':240,'E27':255,'E28':240,
            'E29':15,'E30':540,'E31':315,'E32':120,'E33':75,'E34':130,'E35':15,'E36':15,
            'E37':15,'E38':315,'E39':15,'E40':1,'E41':15}


def nonfinite(value, path=''):
    found = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite(item, path + '/' + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(nonfinite(item, path + '/' + str(index)))
    return found


def theory_coverage(records):
    rows = {name: [row for result in records if result['experiment'] == name
                   for row in result['rows']] for name in ['E14', 'E15', 'E16', 'E41']}
    ordinary = [r for r in rows['E16'] if r['iteration'] >= 0]
    initial = [r for r in rows['E16'] if r['iteration'] == -1]
    return {
        'certified_states': sum(r['certified_states'] for r in rows['E14']),
        'certified_program_changes': sum(r['certified_violations'] for r in rows['E14']),
        'score_bound_violations': sum(r['score_bound_violations'] for r in rows['E14']),
        'KL_bound_violations': sum(r['KL_bound_violations'] for r in rows['E14']),
        'residual_checkpoints': len(ordinary),
        'residual_bound_covered': sum(r['parameter_distance'] <= r['certificate'] for r in ordinary),
        'initialization_checkpoints': len(initial),
        'initialization_bound_covered': sum(r['parameter_distance'] <= r['theorem_bound'] for r in initial),
        'maximum_protected_constraint_residual': max(r['constraint_residual'] for r in rows['E15']
                                                     if r['kind'] == 'protein' and r['method'] == 'Protected'),
        'minimum_affine_sign_agreement': min(r['sign_agreement'] for r in rows['E41'] if r['kind'] == 'affine'),
        'minimum_precision_sign_agreement': min(r['sign_agreement'] for r in rows['E41'] if r['kind'] == 'precision'),
    }


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    payload = read_json(OUTPUT / 'exports/all_results.json')
    counts = Counter(r['experiment'] for r in payload['records'])
    gaps = {key: {'expected': value, 'actual': counts[key]} for key, value in EXPECTED.items() if counts[key] != value}
    required = {(assay, seed) for assay in core.ASSAYS for seed in range(5)}
    missing_replicates = {}
    for experiment in EXPECTED:
        if experiment in ['E13', 'E40']:
            continue
        present = {(r['assay'], r['seed']) for r in payload['records'] if r['experiment'] == experiment}
        if present != required:
            missing_replicates[experiment] = sorted(required - present)
    invalid = nonfinite(payload)
    max_difference = 0.
    checked = 0
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        for r in payload['records']:
            if r.get('assay') != assay or 'metrics' not in r:
                continue
            saved = torch.load((OUTPUT / r['relative_path']).with_name('predictions.pt'),
                               map_location=DEVICE, weights_only=False)
            test_ids = saved['test_ids'].to(DEVICE) if r['experiment'] == 'E27' else data['partitions']['test']
            metrics, _ = evaluate_scores(saved['score'], truth, saved['request'], test_ids,
                                         saved['evaluation'], saved['utility'])
            if r['experiment'] == 'E18':
                ids = tensors(saved['request']['forgotten'])
                metrics['CA'] = 100 * ((saved['score'][ids] >= 0) == (truth[ids] > 0)).double().mean().item()
            max_difference = max(max_difference, *(abs(metrics[k] - r['metrics'][k]) for k in metrics))
            checked += 1
    result = {'expected_counts': EXPECTED, 'actual_counts': dict(counts), 'inventory_gaps': gaps,
              'nonfinite_fields': invalid, 'independently_recomputed_records': checked,
              'missing_replicates': missing_replicates,
              'theory_coverage': theory_coverage(payload['records']),
              'maximum_metric_difference': max_difference,
              'confirmation_batches': len(list((OUTPUT / 'confirmation').glob('E*/*/seed_*/complete.json')))}
    write_json(OUTPUT / 'exports/verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    assert not gaps
    assert not missing_replicates
    assert not invalid
    assert max_difference < 1e-8
    assert result['confirmation_batches'] == 240


if __name__ == '__main__':
    main()
