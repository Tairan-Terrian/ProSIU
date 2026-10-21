"""Counts supporting the prespecified anchor and effect-magnitude strata."""
import torch
from .common import *
from motivation.run import mutation_table

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
for assay in core.ASSAYS:
    source = read_json(core.DATA / 'processed' / f'{assay}.json')
    data, truth = core.load(assay, core.DATA)
    mutations, anchors = mutation_table(source)
    test, dev = data['partitions']['test'], data['partitions']['dev']
    cuts = torch.quantile(truth[dev].abs(), torch.tensor([.25, .5, .75], device=DEVICE))
    bins = torch.bucketize(truth[test].abs(), cuts)
    classes = []
    for band in range(4):
        y = truth[test[bins == band]]
        classes.append([len(y), (y > 0).sum().item(), (y <= 0).sum().item()])
    counts = []
    for seed in range(5):
        permutation = anchors[torch.randperm(len(anchors), generator=rng(852700 + seed), device=DEVICE)]
        valid = torch.isin(mutations, permutation[:round(len(anchors) * .25)]).all(1)
        training = data['partitions']['train']
        small_source = dict(source, rows=[dict(r) for r in source['rows']])
        for i, row in enumerate(small_source['rows']):
            if row['partition'] == 'train' and not valid[i]:
                row['partition'] = 'inactive'
        req = request_for(small_source, seed)
        f = truth[tensors(req['forgotten'])]
        counts.append({'seed': seed, 'training': valid[training].sum().item(),
                       'held_out': valid[test].sum().item(), 'withdrawn': len(f),
                       'positive': (f > 0).sum().item(), 'negative': (f <= 0).sum().item()})
    print(json.dumps({'assay': assay, 'quartile_counts_total_positive_negative': classes,
                      'anchor_common_counts': counts}), flush=True)
