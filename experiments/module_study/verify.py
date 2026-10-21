"""Independent metric reconstruction and complete exports for module interventions."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import torch

from followup import core
from followup.run import sparse_request
from followup.audit import independently_score
from module_study.run import OUTPUT, VARIANTS
from prosiu.common import read_json, write_json


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    records = {name: [read_json(p) | {'source_path': str(p.relative_to(OUTPUT))}
                      for p in sorted((OUTPUT / folder).glob('*/*/*/result.json'))]
               for name, folder in [('ablation', 'ablation'), ('preservation', 'E03')]}
    accepted = read_json(core.ROOT / 'reference/results.json')['runs']
    errors, main_errors, subgroup_errors, baseline_repeat_errors = [], [], [], []
    pair_checks, partition_checks, seed_checks = [], [], []
    for category, variants in [('ablation', VARIANTS), ('preservation', ['TARF', 'Entanglement', 'ProSIU'])]:
        for assay in core.ASSAYS:
            for variant in variants:
                seed_checks.append(sorted(r['seed'] for r in records[category] if r['assay']==assay and r['variant']==variant)==list(range(5)))
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        for seed in range(5):
            main_ep = core.episode(data, truth, maps, source, assay, seed)
            sparse_ep = core.episode(data, truth, maps, source, assay, seed, req=sparse_request(source, seed))
            checkpoint = torch.load(OUTPUT / 'references' / assay / f'seed_{seed}' / 'parameters.pt', map_location=core.DEVICE, weights_only=True)
            ref, _ = core.predictions(sparse_ep, checkpoint['reference'])
            withdrawn = sparse_ep['req']['forgotten']
            mutations = {(p, source['rows'][i]['q'][3][p]) for i in withdrawn for p in source['rows'][i]['positions']}
            positions = {p for p, _ in mutations}
            groups = {'shared_substitution': [], 'shared_position': [], 'remote': []}
            for i in sparse_ep['req']['retained']:
                row = source['rows'][i]
                substitutions = {(p, row['q'][3][p]) for p in row['positions']}
                name = ('shared_substitution' if substitutions & mutations else
                        'shared_position' if set(row['positions']) & positions else 'remote')
                groups[name].append(i)
            complete = next(r for r in records['ablation'] if r['assay']==assay and r['seed']==seed and r['variant']=='ProSIU')
            sampled = next(r for r in records['ablation'] if r['assay']==assay and r['seed']==seed and r['variant']=='w_o_EPS')
            pair_checks.extend(sampled['pair_counts'][stage]['raw']==complete['pair_counts'][stage]['raw'] for stage in ['before', 'after'])
            for category, entries in records.items():
                for row in entries:
                    if row['assay'] != assay or row['seed'] != seed:
                        continue
                    parameters = torch.load((OUTPUT / row['source_path']).parent / 'parameters.pt', map_location=core.DEVICE, weights_only=True)
                    ep = main_ep if category=='ablation' else sparse_ep
                    if row['variant']=='w_o_MPO':
                        ep = dict(ep, heads={h:dict(item, cfg=row['diagnostics'][h]['configuration']) for h,item in ep['heads'].items()})
                    measured = independently_score(ep, parameters)
                    errors.append(max(abs(measured[k]-row['metrics'][k]) for k in measured))
                    if category=='ablation' and row['variant']=='ProSIU':
                        expected = next(r for r in accepted if r['assay']==assay and r['seed']==seed)
                        main_errors.append(max(abs(measured[k]-expected['metrics'][k]) for k in measured))
                    if category=='preservation':
                        score, _ = core.predictions(ep, parameters)
                        for name, indices in groups.items():
                            ids = torch.tensor(indices, device=core.DEVICE)
                            actual = torch.bincount(2*(truth[ids]>0).long()+(score[ids]>=0).long(), minlength=4)
                            reported = row['subgroups'][name]
                            partition_checks.append(actual.cpu().tolist()==reported['confusion'] and len(indices)==reported['n'])
                            rms = (score[ids]-ref[ids]).square().mean().sqrt().item()
                            subgroup_errors.append(abs(rms-reported['prediction_rms_to_retrain']))
                        old = read_json(core.RESEARCH / 'studies/followup_recheck_20261019/E03' / assay / f'seed_{seed}' / row['variant'] / 'result.json')
                        if row['variant'] != 'ProSIU':
                            baseline_repeat_errors.append(max(abs(row['metrics'][k]-old['metrics'][k]) for k in measured))
            print(f'Verified {assay} seed={seed}', flush=True)
    report = {'records': {k:len(v) for k,v in records.items()},
              'all_prescribed_seeds': all(seed_checks), 'matched_raw_pair_counts': all(pair_checks),
              'independent_metric_vectors':len(errors), 'maximum_metric_error':max(errors),
              'maximum_full_model_difference_from_main':max(main_errors),
              'subgroup_memberships_and_confusions_match':all(partition_checks),
              'maximum_subgroup_rmse_error':max(subgroup_errors),
              'maximum_figure_baseline_repeat_difference':max(baseline_repeat_errors)}
    report['passed'] = (report['records']=={'ablation':60,'preservation':45} and all(seed_checks) and all(pair_checks)
                        and max(errors)<1e-10 and max(main_errors)<1e-8 and all(partition_checks)
                        and max(subgroup_errors)<1e-10 and max(baseline_repeat_errors)<1e-10)
    write_json(OUTPUT/'results.json', records)
    write_json(OUTPUT/'verification.json', report)
    with ZipFile(OUTPUT/'measured_results.zip', 'w', ZIP_DEFLATED) as archive:
        for folder in ['ablation','E03','references']:
            for p in sorted((OUTPUT/folder).rglob('*')):
                if p.is_file():
                    archive.write(p, str(p.relative_to(OUTPUT)))
        for p in sorted((core.ROOT/'experiments/module_study').glob('*')):
            if p.is_file():
                archive.write(p, 'code/'+p.name)
        for name in ['results.json','verification.json','gpu0.log','gpu1.log']:
            archive.write(OUTPUT/name, name)
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    main()
