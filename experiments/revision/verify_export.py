"""Re-evaluate saved checkpoints on CUDA and export the complete revision."""
import json
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
import torch
from followup.core import dataset, episode, DEVICE, ASSAYS, ROOT, RESEARCH
from followup.audit import independently_score
from prosiu.common import write_json

OUTPUT=RESEARCH/'studies/followup_revision_20261019'


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    groups={key:[json.loads(p.read_text()) | {'source_path':str(p.relative_to(OUTPUT))}
                for p in sorted((OUTPUT/folder).glob('*/*/*/result.json'))]
            for key,folder in [('ablation','ablation'),('reconstruction','E04')]}
    reference=json.loads((ROOT/'reference/results.json').read_text())['runs']
    errors,release_errors,checks=[],[],[]
    for assay in ASSAYS:
        data,truth,maps,source=dataset(assay)
        for seed in range(5):
            ep=episode(data,truth,maps,source,assay,seed)
            for key,rs in groups.items():
                local=[r for r in rs if r['assay']==assay and r['seed']==seed]
                for r in local:
                    params=torch.load((OUTPUT/r['source_path']).with_name('parameters.pt'),
                                      map_location=DEVICE,weights_only=True)
                    current=ep
                    if key=='ablation' and r['variant']=='w_o_MPO':
                        current=dict(ep,heads={h:dict(v,cfg=r['diagnostics'][h]['configuration']) for h,v in ep['heads'].items()})
                    budgets=[('metrics','converged'),('metrics16','sixteen_steps')] if key=='ablation' else [('metrics',None)]
                    for metric_key,budget in budgets:
                        values=independently_score(current,params[budget] if budget else params)
                        error=max(abs(values[k]-r[metric_key][k]) for k in values)
                        errors.append({'source':r['source_path'],'budget':budget,'maximum_error':error})
                    if key=='ablation' and r['variant']=='ProSIU':
                        accepted=next(t for t in reference if t['assay']==assay and t['seed']==seed)
                        release_errors.append(max(abs(r['metrics'][k]-accepted['metrics'][k]) for k in r['metrics']))
                if key=='reconstruction':
                    counts=[r['comparison_count'] for r in local if r['variant'].endswith('_Matched')]
                    checks.append(len(counts)==4 and len(set(counts))==1)
            print(f'Validated {assay} / seed {seed}',flush=True)
    conditions=Counter((key,r['assay'],r['variant']) for key,rs in groups.items() for r in rs)
    seed_groups={condition:sorted(r['seed'] for r in groups[condition[0]]
                  if r['assay']==condition[1] and r['variant']==condition[2]) for condition in conditions}
    report={'records':{k:len(v) for k,v in groups.items()},'metric_vectors_recomputed':len(errors),
            'max_independent_metric_error':max(e['maximum_error'] for e in errors),
            'max_full_model_difference_from_accepted_release':max(release_errors),
            'all_prescribed_seeds':all(v==list(range(5)) for v in seed_groups.values()),
            'matched_comparison_counts':all(checks),'metric_reconstruction':errors}
    report['passed']=(report['records']=={'ablation':60,'reconstruction':120}
          and report['max_independent_metric_error']<1e-10
          and report['max_full_model_difference_from_accepted_release']<1e-8
          and report['all_prescribed_seeds'] and report['matched_comparison_counts'])
    write_json(OUTPUT/'revision_results.json',groups)
    write_json(OUTPUT/'verification.json',report)
    with ZipFile(OUTPUT/'revision_results.zip','w',ZIP_DEFLATED) as z:
        for folder in ['ablation','E04']:
            for p in sorted((OUTPUT/folder).rglob('*')):
                if p.is_file(): z.write(p,str(p.relative_to(OUTPUT)))
        for p in sorted((ROOT/'experiments/revision').glob('*')):
            if p.is_file(): z.write(p,'code/'+p.name)
        for name in ['revision_results.json','verification.json','run.log']:
            z.write(OUTPUT/name,name)
    print(json.dumps({k:v for k,v in report.items() if k!='metric_reconstruction'},indent=2),flush=True)


if __name__=='__main__':
    main()
