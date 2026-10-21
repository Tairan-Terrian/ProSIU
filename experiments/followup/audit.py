"""Reconstruct measured metrics and account for the fixed evaluation protocol."""
import json
import math
from collections import Counter
import torch

from .core import OUTPUT, DATA, ASSAYS, SEEDS, DEVICE, dataset, episode, predictions, request, configuration
from .run import saved_parameters, sparse_request
from .datasets import EXTERNAL_CONFIG

EXPECTED = dict(zip([f'E{i:02}' for i in range(1,13)],
    [330,60,90,120,120,30,105,150,105,105,540,600]))
checks = []


def check(name, condition, detail):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})


def finite(value):
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,list): return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value,float) else True


def independently_score(ep, parameters):
    score, utility = predictions(ep, parameters)
    result = {}
    for key, ids in [('CA',ep['req']['forgotten']),('RA',ep['req']['retained']),
                     ('TA',ep['data']['partitions']['test'])]:
        labels = ep['truth'][ids]>0
        decisions = score[ids]>=0
        result[key] = 50*(decisions[labels].double().mean()
                         +(~decisions[~labels]).double().mean()).item()
    a,b = ep['evaluation']['first'],ep['evaluation']['second']
    difference = utility[a]-utility[b]
    truth = ep['truth'][a]-ep['truth'][b]
    credit = ((difference>0)==(truth>0)).double()
    credit[difference==0] = .5
    result['PA'] = 100*credit[truth!=0].mean().item()
    return result


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    records = [json.loads(p.read_text()) | {'path':p} for p in sorted(OUTPUT.glob('E*/*/*/*/result.json'))]
    counts = Counter(r['experiment'] for r in records)
    check('fixed experiment counts',dict(counts)==EXPECTED,dict(counts))
    check('finite measured values',all(finite({k:v for k,v in r.items() if k!='path'}) for r in records),len(records))
    check('complete paired seed sets',all(set(r['seed'] for r in records if r['experiment']==e)==set(SEEDS) for e in EXPECTED),SEEDS)
    groups={}
    certificate_violations=[]
    for r in records:
        key=(r['experiment'],r['assay'],r.get('variant',r.get('method',r.get('head'))))
        groups.setdefault(key,[]).append(r['seed'])
        for head,values in r.get('equivalence',{}).items():
            if isinstance(values,dict) and values['parameter_distance']>values['residual_certificate']+0.000001:
                certificate_violations.append([*key,r['seed'],head,values['parameter_distance'],values['residual_certificate']])
    check('all conditions contain each prescribed seed once',all(sorted(v)==SEEDS for v in groups.values()),len(groups))
    check('gradient-residual certificate covers parameter distance',len(certificate_violations)==0,certificate_violations)
    max_metric_error = 0.
    max_rollout_error = 0.
    max_reference_error = 0.
    max_sequential_reference_error = 0.
    query_checks = []
    pools = {}
    for assay in ASSAYS:
        data,truth,maps,source = dataset(assay)
        for seed in SEEDS:
            for learner in ['common','final']:
                ep = episode(data,truth,maps,source,assay,seed,learner)
                for r in records:
                    if r['experiment']=='E01' and r['assay']==assay and r['seed']==seed and r['learner']==learner:
                        params = saved_parameters('E01',assay,seed,r['variant'])
                        metrics = independently_score(ep,params)
                        max_metric_error=max(max_metric_error,max(abs(metrics[k]-r['metrics'][k]) for k in metrics))
                if learner=='final':
                    reference=saved_parameters('E01',assay,seed,'final_Retrain')
                    for r in records:
                        if r['experiment']=='E02' and r['assay']==assay and r['seed']==seed:
                            params=saved_parameters('E02',assay,seed,r['variant'])
                            for h,item in ep['heads'].items():
                                rms=(item['x'][data['partitions']['test']]@(params[h]-reference[h])).square().mean().sqrt().item()
                                max_reference_error=max(max_reference_error,abs(rms-r['equivalence'][h]['H_prediction_rms']))
            local=[r for r in records if r['assay']==assay and r['seed']==seed]
            for r in local:
                if r['experiment'] in ['E02','E04','E05','E08','E09','E12']:
                    evaluation_ep=ep
                    if r['experiment']=='E05' and r['modules']['MPO']==0:
                        evaluation_ep=dict(ep,heads={h:dict(v,cfg=dict(v['cfg'])) for h,v in ep['heads'].items()})
                        evaluation_ep['heads']['classification']['cfg']['threshold']=0.
                    params=saved_parameters(r['experiment'],assay,seed,r['variant'])
                    metrics=independently_score(evaluation_ep,params)
                    max_metric_error=max(max_metric_error,max(abs(metrics[k]-r['metrics'][k]) for k in metrics))
            sparse_ep=episode(data,truth,maps,source,assay,seed,req=sparse_request(source,seed))
            for r in local:
                if r['experiment']=='E03':
                    params=saved_parameters('E03',assay,seed,r['variant'])
                    metrics=independently_score(sparse_ep,params)
                    max_metric_error=max(max_metric_error,max(abs(metrics[k]-r['metrics'][k]) for k in metrics))
            matched=[r['comparison_count'] for r in local if r['experiment']=='E04' and r['variant'].endswith('_Matched')]
            check(f'E04 matched bank size {assay}/{seed}',len(set(matched))==1,matched)
            clean=[r for r in local if r['experiment']=='E04' and r['variant'].split('_')[0]!='RawOnly']
            check(f'E04 endpoint withdrawal {assay}/{seed}',all(r['remaining_endpoint_dependencies']==0 for r in clean),len(clean))
            factorial=[tuple(r['modules'][k] for k in ['EPS','MPO','SEF']) for r in local if r['experiment']=='E05']
            check(f'E05 factorial {assay}/{seed}',len(set(factorial))==8,factorial)
            for r in local:
                if r['experiment']=='E03':
                    groups=[g for k,g in r['subgroups'].items() if k!='single_anchors']
                    valid=all(sum(g['confusion'])==g['n'] and g['positive']>0 and g['negative']>0 for g in groups)
                    valid &= sum(g['n'] for g in groups)==len(r['request']['retained'])
                    check(f'E03 partition {assay}/{seed}/{r["variant"]}',valid,[g['n'] for g in groups])
                if r['experiment']=='E06':
                    for name,trace in r['trajectories'].items():
                        monotone=all(trace[j+1]['seconds']>=trace[j]['seconds'] and trace[j+1]['function_evaluations']>=trace[j]['function_evaluations'] for j in range(len(trace)-1))
                        check(f'E06 accounting {assay}/{seed}/{r["head"]}/{name}',monotone,len(trace))
                if r['experiment'] in ['E07','E08']:
                    roll=r if r['experiment']=='E07' else r['rollout']
                    q=torch.tensor(roll['query_ids'],device=DEVICE)
                    p=torch.tensor(roll['candidate_ids'],device=DEVICE)
                    query_checks.append(bool((q[:,:,None]==p[:,None,:]).any(2).all()))
                    query_checks.append(all(len(set(row))==16 for row in roll['query_ids']))
                    best=truth[q].cummax(1).values
                    regret=truth[p].max(1).values[:,None]-best
                    error=max((best-torch.tensor(roll['per_trajectory_best'],device=DEVICE)).abs().max().item(),
                              (regret-torch.tensor(roll['per_trajectory_regret'],device=DEVICE)).abs().max().item(),
                              (regret.mean(0)-torch.tensor(roll['simple_regret_mean'],device=DEVICE)).abs().max().item())
                    max_rollout_error=max(max_rollout_error,error)
                    key=(assay,seed)
                    if key in pools: query_checks.append(pools[key]==roll['candidate_ids'])
                    else: pools[key]=roll['candidate_ids']
            final=request(source['rows'],seed)
            expected_groups={tuple(source['rows'][i]['positions']) for i in final['forgotten']}
            sequential=[r for r in local if r['experiment']=='E12']
            valid=all(set(map(tuple,r['sequence_group_keys']))==expected_groups and
                      r['trajectory'][-1]['withdrawn_count']==len(final['forgotten']) and
                      len(r['trajectory'])==r['batches'] for r in sequential)
            check(f'E12 identical final withdrawal {assay}/{seed}',valid,len(sequential))
            for r in sequential:
                parameters=saved_parameters('E12',assay,seed,r['variant'])
                for head,item in ep['heads'].items():
                    rms=(item['x'][data['partitions']['test']]@(parameters[head]-reference[head])).square().mean().sqrt().item()
                    max_sequential_reference_error=max(max_sequential_reference_error,abs(rms-r['equivalence'][head]['H_prediction_rms']))
    check('independent retraining-distance reconstruction',max_reference_error<1e-12,max_reference_error)
    check('sequential endpoints share the original retained-only reference',max_sequential_reference_error<1e-8,max_sequential_reference_error)
    check('independent sequential reward reconstruction',max_rollout_error<1e-12,max_rollout_error)
    check('shared pools and unique admissible queries',all(query_checks),len(query_checks))
    for kind in ['Random','PositionPair','SeenConstituents','UnseenConstituent']:
        for assay in ASSAYS:
            source=json.loads((OUTPUT/'extrapolation_data'/kind/'processed'/f'{assay}.json').read_text())
            rows=source['rows']
            original_source=json.loads((DATA/'processed'/f'{assay}.json').read_text())
            development=[i for i,r in enumerate(original_source['rows']) if r['partition'] in ['dev_train','dev']]
            check(f'E11 development quarantine {assay}/{kind}',
                  all(rows[i]['partition']==original_source['rows'][i]['partition'] for i in development),len(development))
            train=[r for r in rows if r['partition']=='train']
            test=[r for r in rows if r['partition']=='test']
            genotype=lambda r:tuple((p,r['q'][3][p]) for p in r['positions'])
            check(f'E11 target separation {assay}/{kind}',set(map(genotype,train)).isdisjoint(set(map(genotype,test))),[len(train),len(test)])
            if kind=='PositionPair':
                check(f'E11 position-pair separation {assay}',set(tuple(r['positions']) for r in train).isdisjoint(set(tuple(r['positions']) for r in test)),len(test))
            if kind in ['SeenConstituents','UnseenConstituent']:
                known={v for r in train for v in genotype(r)}
                valid=all(all(v in known for v in genotype(r)) if kind=='SeenConstituents' else any(v not in known for v in genotype(r)) for r in test)
                check(f'E11 constituent separation {assay}/{kind}',valid,len(test))
    for experiment in ['E10','E11']:
        assay_names=sorted({r['assay'] for r in records if r['experiment']==experiment})
        for name in assay_names:
            if experiment=='E10':
                assay=name
                folder=OUTPUT/'external_data'
                cfg=EXTERNAL_CONFIG
            else:
                assay,kind=name.split('_',1)
                folder=OUTPUT/'extrapolation_data'/kind
                cfg=configuration(assay)
            data,truth,maps,source=dataset(assay,folder)
            for seed in SEEDS:
                ep=episode(data,truth,maps,source,name,seed,config=cfg)
                for r in records:
                    if r['experiment']==experiment and r['assay']==name and r['seed']==seed:
                        params=saved_parameters(experiment,name,seed,r['variant'])
                        if r.get('learner')=='native':
                            from .native_e11 import inputs as native_inputs
                            from legacy_prosiu.correction_model import comparison_features
                            stream,req,settings,metadata=native_inputs(data,truth,source,assay,seed,r['variant'])
                            prediction=stream['prior']+data['x']@params['shared']
                            metrics={}
                            for key,ids in [('CA',req['forgotten']),('RA',req['retained']),('TA',data['partitions']['test'])]:
                                labels=truth[ids]>0
                                decision=prediction[ids]>=0
                                metrics[key]=50*(decision[labels].double().mean()+(~decision[~labels]).double().mean()).item()
                            bank=ep['evaluation']
                            features,offset=comparison_features(stream,bank,torch.tensor(req['retained'],device=DEVICE))
                            prediction=offset+features@params['shared']
                            target=truth[bank['first']]-truth[bank['second']]
                            credit=((prediction>0)==(target>0)).double()
                            credit[prediction==0]=.5
                            metrics['PA']=100*credit[target!=0].mean().item()
                            check(f'E11 native provenance {name}/{seed}/{r["variant"]}',metadata['target_training_labels_in_native_prior']==0,metadata['training_evidence'])
                            check(f'E11 common comparison bank {name}/{seed}/{r["variant"]}',all(bank[k].cpu().tolist()==r['comparison_bank'][k] for k in ['first','second']),len(bank['first']))
                        else:
                            metrics=independently_score(ep,params)
                        max_metric_error=max(max_metric_error,max(abs(metrics[k]-r['metrics'][k]) for k in metrics))
    check('independent four-metric reconstruction of stored primary checkpoints',max_metric_error<1e-9,max_metric_error)
    report={'passed':all(c['passed'] for c in checks),'expected_records':sum(EXPECTED.values()),'counts':dict(counts),'checks':checks}
    folder=OUTPUT/'exports'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
    print(json.dumps([c for c in checks if not c['passed']],indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
