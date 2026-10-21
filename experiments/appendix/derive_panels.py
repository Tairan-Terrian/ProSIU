"""Measure restored and newly introduced evidence from saved model predictions."""
import torch
from .common import *


def subset_scores(score, truth, ids):
    labels = truth[ids] > 0
    correct = (score[ids] >= 0) == labels
    return {'count': len(ids), 'positive': labels.sum().item(), 'negative': (~labels).sum().item(),
            'accuracy': 100 * correct.double().mean().item(),
            'correct': correct.sum().item()}


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        references = sorted((OUTPUT / 'E31' / assay).glob('seed_*/*/Retained retraining/result.json'))
        for path in references:
            reference_record = read_json(path)
            seed = reference_record['seed']
            reference = torch.load(path.with_name('predictions.pt'), map_location=DEVICE, weights_only=False)
            ep = core.episode(data, truth, maps, source, assay, seed)
            new = data['partitions']['dev_train']
            new = new[torch.randperm(len(new), generator=rng(853100 + seed), device=DEVICE)[:512]]
            active = torch.cat([tensors(ep['req']['retained']), new[:reference_record['new_count']]]).sort().values
            stream = dict(ep['stream'], eps=ep['stream']['eps'].clone())
            stream['eps'][active] = truth[active]
            pairs = expand_pairs(program_pairs(stream, active, active, seed), active, seed)
            ep['heads'] = {head: dict(item, after=core.ExtendedObjective(
                item['x'], stream['eps'], active, pairs, item['cfg'])) for head, item in ep['heads'].items()}
            record('E31', ep, reference_record['condition'], 'Frozen withdrawal model',
                   saved_main(assay, seed), reference['parameters'], {'update': 'frozen starting model'},
                   {key: reference_record[key] for key in ['batches', 'batch', 'new_count']})
        for experiment in ['E25', 'E31']:
            for path in sorted((OUTPUT / experiment / assay).glob('seed_*/*/*/result.json')):
                result = read_json(path)
                saved = torch.load(path.with_name('predictions.pt'), map_location=DEVICE, weights_only=False)
                seed = result['seed']
                panels = {}
                if experiment == 'E25':
                    f = tensors(saved['request']['forgotten'])
                    order = f[torch.randperm(len(f), generator=rng(852500 + seed), device=DEVICE)]
                    count = result['restored_count']
                    if count:
                        panels['Restored combinations'] = subset_scores(saved['score'], truth, order[:count])
                    if count < len(order):
                        panels['Still withdrawn combinations'] = subset_scores(saved['score'], truth, order[count:])
                else:
                    identities = data['partitions']['dev_train']
                    order = identities[torch.randperm(len(identities), generator=rng(853100 + seed), device=DEVICE)[:512]]
                    count = result['new_count']
                    panels['Newly introduced combinations'] = subset_scores(saved['score'], truth, order[:count])
                    panels['Earlier retained combinations'] = subset_scores(
                        saved['score'], truth, tensors(saved['request']['retained']))
                result['evidence_panels'] = panels
                write_json(path, result)
    print('Saved restored-evidence and continued-learning cohort measurements', flush=True)


if __name__ == '__main__':
    main()
