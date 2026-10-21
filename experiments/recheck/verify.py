"""Reconstruct all replication metrics from saved CUDA model parameters."""
import json
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
import torch
from followup.core import ROOT, RESEARCH, DEVICE, ASSAYS, dataset, episode, predictions
from followup.run import sparse_request
from followup.audit import independently_score
from prosiu.common import write_json

OUTPUT=RESEARCH/'studies/followup_recheck_20261019'
OLD_ABLATION=RESEARCH/'studies/followup_revision_20261019/ablation'
OLD_FIGURE=RESEARCH/'studies/followup_main12/E03'


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    records={k:[json.loads(p.read_text()) | {'source_path':str(p.relative_to(OUTPUT))}
                for p in sorted((OUTPUT/folder).glob('*/*/*/result.json'))]
             for k,folder in [('ablation','ablation'),('preservation','E03')]}
    reference=json.loads((ROOT/'reference/results.json').read_text())['runs']
    metric_errors,release_errors,repeat_errors,subgroup_errors=[],[],[],[]
    valid_partitions=[]
    for assay in ASSAYS:
        data,truth,maps,source=dataset(assay)
        for seed in range(5):
            main_ep=episode(data,truth,maps,source,assay,seed)
            sparse_ep=episode(data,truth,maps,source,assay,seed,req=sparse_request(source,seed))
            ref=torch.load(OUTPUT/'E03'/assay/f'seed_{seed}'/'Retrain/parameters.pt',map_location=DEVICE,weights_only=True)
            reference_score,_=predictions(sparse_ep,ref)
            withdrawn=sparse_ep['req']['forgotten']
            mutations={(p,source['rows'][i]['q'][3][p]) for i in withdrawn for p in source['rows'][i]['positions']}
            positions={p for p,_ in mutations}
            groups={'shared_substitution':[],'shared_position':[],'remote':[]}
            for i in sparse_ep['req']['retained']:
                row=source['rows'][i]
                substitutions={(p,row['q'][3][p]) for p in row['positions']}
                group=('shared_substitution' if substitutions & mutations else
                       'shared_position' if set(row['positions']) & positions else 'remote')
                groups[group].append(i)
            for category,entries in records.items():
                for r in entries:
                    if r['assay']!=assay or r['seed']!=seed: continue
                    folder=(OUTPUT/r['source_path']).parent
                    params=torch.load(folder/'parameters.pt',map_location=DEVICE,weights_only=True)
                    if category=='ablation':
                        ep=main_ep
                        if r['variant']=='w_o_MPO':
                            ep=dict(ep,heads={h:dict(v,cfg=r['diagnostics'][h]['configuration']) for h,v in ep['heads'].items()})
                        for field,budget in [('metrics','converged'),('metrics16','sixteen_steps')]:
                            actual=independently_score(ep,params[budget])
                            metric_errors.append(max(abs(actual[k]-r[field][k]) for k in actual))
                        old=json.loads((OLD_ABLATION/assay/f'seed_{seed}'/r['variant']/'result.json').read_text())
                        if r['variant']=='ProSIU':
                            accepted=next(a for a in reference if a['assay']==assay and a['seed']==seed)
                            release_errors.append(max(abs(r['metrics'][k]-accepted['metrics'][k]) for k in r['metrics']))
                    else:
                        actual=independently_score(sparse_ep,params)
                        metric_errors.append(max(abs(actual[k]-r['metrics'][k]) for k in actual))
                        score,_=predictions(sparse_ep,params)
                        for name,indices in groups.items():
                            ids=torch.tensor(indices,device=DEVICE)
                            positive=truth[ids]>0
                            decision=score[ids]>=0
                            confusion=torch.bincount(2*positive.long()+decision.long(),minlength=4)
                            measured=r['subgroups'][name]
                            valid_partitions.append(confusion.cpu().tolist()==measured['confusion'])
                            valid_partitions.append(len(indices)==measured['n'])
                            accuracy=50*(decision[positive].double().mean()+(~decision[~positive]).double().mean()).item()
                            stored=50*(measured['confusion'][0]/measured['negative']+measured['confusion'][3]/measured['positive'])
                            rms=(score[ids]-reference_score[ids]).square().mean().sqrt().item()
                            subgroup_errors.extend([abs(accuracy-stored),abs(rms-measured['prediction_rms_to_retrain'])])
                        old=json.loads((OLD_FIGURE/assay/f'seed_{seed}'/r['variant']/'result.json').read_text())
                    repeat_errors.append({'category':category,'assay':assay,'seed':seed,'variant':r['variant'],
                        'max_metric_difference':max(abs(r['metrics'][k]-old['metrics'][k]) for k in r['metrics'])})
            print(f'Verified {assay} seed={seed}',flush=True)
    counts=Counter((k,r['assay'],r['variant']) for k,rs in records.items() for r in rs)
    report={'record_counts':{k:len(v) for k,v in records.items()},
        'all_conditions_have_five_seeds':all(sorted(r['seed'] for r in records[k] if r['assay']==a and r['variant']==v)==list(range(5)) for k,a,v in counts),
        'metric_vectors_recomputed':len(metric_errors),'max_independent_metric_error':max(metric_errors),
        'max_full_model_difference_from_main':max(release_errors),
        'subgroup_partitions_and_confusions_verified':all(valid_partitions),
        'max_independent_subgroup_error':max(subgroup_errors),
        'max_repeat_metric_difference':max(r['max_metric_difference'] for r in repeat_errors),
        'repeat_comparisons':repeat_errors}
    report['passed']=(report['record_counts']=={'ablation':60,'preservation':90}
        and report['all_conditions_have_five_seeds'] and report['max_independent_metric_error']<1e-10
        and report['max_full_model_difference_from_main']<1e-8
        and report['subgroup_partitions_and_confusions_verified'] and report['max_independent_subgroup_error']<1e-10)
    write_json(OUTPUT/'recheck_results.json',records)
    write_json(OUTPUT/'verification.json',report)
    with ZipFile(OUTPUT/'recheck_results.zip','w',ZIP_DEFLATED) as z:
        for folder in ['ablation','E03']:
            for p in sorted((OUTPUT/folder).rglob('*')):
                if p.is_file(): z.write(p,str(p.relative_to(OUTPUT)))
        for folder in ['recheck','revision']:
            for p in sorted((ROOT/'experiments'/folder).glob('*')):
                if p.is_file(): z.write(p,'code/'+folder+'/'+p.name)
        for name in ['recheck_results.json','verification.json','gpu0.log','gpu1.log']:
            z.write(OUTPUT/name,name)
    print(json.dumps({k:v for k,v in report.items() if k!='repeat_comparisons'},indent=2),flush=True)


if __name__=='__main__':
    main()
