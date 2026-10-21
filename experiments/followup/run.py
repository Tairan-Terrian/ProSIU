"""Ordered execution of the twelve main-paper experiments."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import random

import torch

from .core import *
from .programs import pairs_from_states
from .closed_loop import rollout
METHOD_FILTER=None
REPLACE=False


def foundations(ep):
    before,reference,details={},{},{}
    for head,item in ep['heads'].items():
        before[head],old=fit(item['before'])
        reference[head],ref=fit(item['after'],iterations=1000,strict=True)
        details[head]={'original':old,'reference':ref}
    return before,reference,details


def e01(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            for learner in ['final','common']:
                ep=episode(data,truth,maps,source,assay,seed,learner)
                original,reference,base_details=foundations(ep)
                save_result('E01',ep,learner+'_Original',original,reference,base_details)
                save_result('E01',ep,learner+'_Retrain',reference,reference,base_details)
                for method in (METHOD_FILTER or ['ProSIU','RetainedOpt']+GENERAL):
                    destination=OUTPUT/'E01'/assay/f'seed_{seed}'/(learner+'_'+method)/'result.json'
                    if destination.exists() and not REPLACE:
                        continue
                    parameters,diagnostics={},{}
                    for head,item in ep['heads'].items():
                        start=timed()
                        if method=='ProSIU':
                            starts,geometry=initializations(item['x'],ep['req'],item['after'],original[head])
                            parameters[head],info=fit(item['after'],starts['ProSIU'])
                            info['geometry']=geometry
                        elif method=='RetainedOpt':
                            parameters[head],info=fit(item['after'],original[head])
                        else:
                            setting=read_json(RESEARCH/'studies/epistasis_correction/frozen'/method/f'{assay}.json')['configuration']
                            parameters[head],info=update(method,item['after'],item['revoked'],original[head],setting,seed,learner)
                            info['configuration']=setting
                        info['seconds']=timed()-start
                        diagnostics[head]=info
                    save_result('E01',ep,learner+'_'+method,parameters,reference,diagnostics,
                        {'evaluation':'same direct interaction and pair-ranking outputs within both learner families'})


def saved_parameters(experiment,assay,seed,variant):
    return torch.load(OUTPUT/experiment/assay/f'seed_{seed}'/variant/'parameters.pt',map_location=DEVICE,weights_only=True)


def e02(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            reference=saved_parameters('E01',assay,seed,'final_Retrain')
            for variant in ['Original','ProSIU','RetainedOpt','Retrain']:
                parameters=saved_parameters('E01',assay,seed,'final_'+variant)
                save_result('E02',ep,variant,parameters,reference,{},
                    {'reference':'zero-initialized retained fit with strict residual refinement'})


def sparse_request(source,seed):
    rows=source['rows']
    training=[i for i,r in enumerate(rows) if r['partition']=='train']
    counts={p:sum(p in rows[i]['positions'] for i in training) for p in {p for i in training for p in rows[i]['positions']}}
    positions=sorted(p for p,n in counts.items() if n>=64)
    randomizer=random.Random(75100+seed)
    randomizer.shuffle(positions)
    focal=positions[0]
    around=[i for i in training if focal in rows[i]['positions']]
    # A sparse position request is determined solely by sequence identities.
    selected=around[:max(1,min(64,len(around)))]
    req=request(rows,seed)
    req['forgotten']=selected
    req['retained']=sorted(set(training)-set(selected))
    req['shifts']=[randomizer.choice([-1.,1.]) for _ in selected]
    pairs=[rows[i]['positions'] for i in selected]
    focus={p for pair in pairs for p in pair}
    req['selected_position_pairs']=pairs
    req['adjacent']=[i for i,r in enumerate(rows) if any(p in focus for p in r['positions'])]
    return req


def subgroup_scores(ep,parameters,original,reference,source):
    score,_=predictions(ep,parameters)
    before,_=predictions(ep,original)
    ref,_=predictions(ep,reference)
    rows=source['rows']
    mutations={(p,rows[i]['q'][3][p]) for i in ep['req']['forgotten'] for p in rows[i]['positions']}
    positions={p for p,a in mutations}
    subsets={'shared_substitution':[],'shared_position':[],'remote':[]}
    for i in ep['req']['retained']:
        row=rows[i]
        if any((p,row['q'][3][p]) in mutations for p in row['positions']): key='shared_substitution'
        elif any(p in positions for p in row['positions']): key='shared_position'
        else: key='remote'
        subsets[key].append(i)
    result={}
    for name,indices in subsets.items():
        ids=torch.tensor(indices,device=DEVICE,dtype=torch.long)
        labels=ep['truth'][ids]>0
        confusion=torch.bincount(2*labels.long()+(score[ids]>=0).long(),minlength=4)
        result[name]={'n':len(ids),'positive':int(labels.sum()),'negative':int((~labels).sum()),
            'correct':int(((score[ids]>=0)==labels).sum()),'confusion':confusion.cpu().tolist(),
            'accuracy':100*((score[ids]>=0)==labels).double().mean().item(),
            'prediction_rms_to_retrain':(score[ids]-ref[ids]).square().mean().sqrt().item(),
            'update_rms':(score[ids]-before[ids]).square().mean().sqrt().item()}
    result['single_anchors']={'count':len(mutations),'maximum_absolute_drift':0.,
        'mechanism':'same measured constituent branch before and after update'}
    return result


def e03(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed,req=sparse_request(source,seed))
            original,reference,base=foundations(ep)
            for method in ['Original','ProSIU','RetainedOpt','TARF','Entanglement','Retrain']:
                parameters,details={},{}
                for head,item in ep['heads'].items():
                    if method=='Original': parameters[head]=original[head]
                    elif method=='Retrain': parameters[head]=reference[head]
                    elif method=='RetainedOpt': parameters[head],details[head]=fit(item['after'],original[head])
                    elif method=='ProSIU':
                        starts,geometry=initializations(item['x'],ep['req'],item['after'],original[head])
                        parameters[head],details[head]=fit(item['after'],starts['ProSIU'])
                        details[head]['geometry']=geometry
                    else:
                        setting=read_json(RESEARCH/'studies/epistasis_correction/frozen'/method/f'{assay}.json')['configuration']
                        parameters[head],details[head]=update(method,item['after'],item['revoked'],original[head],setting,seed)
                save_result('E03',ep,method,parameters,reference,details,
                    {'subgroups':subgroup_scores(ep,parameters,original,reference,source),'request':ep['req'],
                     'subgroup_metric':'ordinary accuracy with confusion counts; pooled balanced accuracy computed in report'})


def e04(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            original,reference,base=foundations(ep)
            train=torch.tensor(sorted(ep['req']['retained']+ep['req']['forgotten']),device=DEVICE)
            retain=torch.tensor(ep['req']['retained'],device=DEVICE)
            forget=torch.tensor(ep['req']['forgotten'],device=DEVICE)
            old=program_pairs(ep['stream'],train,train,seed)
            complete=program_pairs(ep['stream'],retain,retain,seed)
            all_pairs=expand_pairs(old,train,seed)
            keep=~(torch.isin(all_pairs['first'],forget)|torch.isin(all_pairs['second'],forget))
            endpoint={k:all_pairs[k][keep] for k in ['first','second']}
            rebuilt=pairs_from_states(ep['stream'],old,retained=retain,seed=seed)
            versions={'RawOnly':all_pairs,'EndpointRemoval':endpoint,
                'StateRebuilt':expand_pairs(rebuilt,retain,seed),'FullClosure':expand_pairs(complete,retain,seed)}
            matched_count=min(len(v['first']) for v in versions.values())
            for matched in [False,True]:
                for name,pairs in versions.items():
                    records=pairs
                    if matched:
                        ids=torch.randperm(len(pairs['first']),generator=rng(75200+seed),device=DEVICE)[:matched_count]
                        records={k:pairs[k][ids] for k in ['first','second']}
                    parameters,details={},{}
                    for head,item in ep['heads'].items():
                        objective=ExtendedObjective(item['x'],ep['stream']['eps'],retain,records,item['cfg'])
                        starts,geometry=initializations(item['x'],ep['req'],objective,original[head])
                        parameters[head],details[head]=fit(objective,starts['ProSIU'])
                    residual_endpoints=int((torch.isin(records['first'],forget)|torch.isin(records['second'],forget)).sum())
                    history_affected=torch.isin(old['history'][old['state']],forget).any(1)
                    live_endpoints=~(torch.isin(old['first'],forget)|torch.isin(old['second'],forget))
                    residual_states=int((history_affected&live_endpoints).sum()) if name in ['RawOnly','EndpointRemoval'] else 0
                    save_result('E04',ep,name+('_Matched' if matched else ''),parameters,reference,details,
                        {'comparison_count':len(records['first']),'matched_count':matched_count,
                         'remaining_endpoint_dependencies':residual_endpoints,
                         'remaining_context_derived_comparisons_in_source_bank':residual_states})


def e05(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            train=torch.tensor(sorted(ep['req']['retained']+ep['req']['forgotten']),device=DEVICE)
            retain=torch.tensor(ep['req']['retained'],device=DEVICE)
            old=program_pairs(ep['stream'],train,train,seed)
            new=program_pairs(ep['stream'],retain,retain,seed)
            for eps,mpo in itertools.product([0,1],repeat=2):
                variant_ep=dict(ep,heads={h:dict(v) for h,v in ep['heads'].items()})
                if eps:
                    old_records,new_records=old,new
                else:
                    old_records=pairs_from_states(ep['stream'],old,kind='heuristic',seed=seed)
                    new_records=pairs_from_states(ep['stream'],new,kind='heuristic',seed=seed)
                original,reference={},{ }
                objectives={}
                for head,item in ep['heads'].items():
                    cfg=copy.deepcopy(item['cfg'])
                    if not mpo:
                        cfg.update(classification=0.,preference=0.,calibration=1.,rank_temperature=1.,threshold=0.)
                    before=ExtendedObjective(item['x'],ep['stream']['eps'],train,expand_pairs(old_records,train,seed),cfg)
                    after=ExtendedObjective(item['x'],ep['stream']['eps'],retain,expand_pairs(new_records,retain,seed),cfg)
                    original[head],_=fit(before)
                    reference[head],_=fit(after,iterations=1000,strict=True)
                    objectives[head]=after
                    variant_ep['heads'][head].update(after=after,cfg=cfg)
                for sef in [0,1]:
                    parameters,details,converged={},{},{}
                    for head,item in ep['heads'].items():
                        objective=objectives[head]
                        start=original[head]
                        if sef:
                            starts,geometry=initializations(item['x'],ep['req'],objective,start)
                            start=starts['ProSIU']
                        parameters[head],details[head]=fit(objective,start,iterations=16)
                        converged[head],_=fit(objective,start)
                    variant=f'EPS{eps}_MPO{mpo}_SEF{sef}'
                    metrics,_=evaluate(variant_ep,converged)
                    save_result('E05',variant_ep,variant,parameters,reference,details,
                        {'modules':{'EPS':eps,'MPO':mpo,'SEF':sef},'budget_iterations':16,
                         'converged_metrics':metrics,'substitution':'same-size heuristic programs; measured-interaction regression; direct retained optimization'})


def e06(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            original=saved_parameters('E01',assay,seed,'final_Original')
            reference=saved_parameters('E01',assay,seed,'final_Retrain')
            for head,item in ep['heads'].items():
                starts,geometry=initializations(item['x'],ep['req'],item['after'],original[head])
                trajectories={}
                for name,start in starts.items():
                    theta=start.clone().requires_grad_(True)
                    optimizer=torch.optim.LBFGS([theta],lr=1.,max_iter=1,history_size=30,
                        tolerance_grad=1e-8,tolerance_change=1e-11,line_search_fn='strong_wolfe')
                    calls=0
                    runtime=geometry['newton_seconds'] if name=='Newton' else (geometry['newton_seconds']+geometry['protection_seconds'] if name in ['ProSIU','Protected'] else 0.)
                    checkpoints=[]
                    for iteration in range(601):
                        if iteration in [0,1,2,4,8,16,32,64,128,256,600]:
                            current=theta.detach()
                            residual=item['after'].gradient(current).norm()
                            row={'iteration':iteration,'function_evaluations':calls,'seconds':runtime,
                                'parameter_distance':(current-reference[head]).norm().item(),
                                'prediction_rms':(item['x']@(current-reference[head])).square().mean().sqrt().item(),
                                'gradient':residual.item(),'certificate':(residual/item['after'].mu).item()}
                            subset=torch.tensor(ep['req']['retained'],device=DEVICE)
                            row['retained_drift']=(item['x'][subset]@(current-original[head])).square().mean().sqrt().item()
                            if head=='classification':
                                for key,ids in [('CA',ep['req']['forgotten']),('RA',ep['req']['retained']),('TA',ep['data']['partitions']['test'])]:
                                    row[key]=balanced_accuracy(item['x'][ids]@current-item['cfg'].get('threshold',0.),(truth[ids]>0).double())
                            else:
                                a,b=ep['evaluation']['first'],ep['evaluation']['second']
                                actual=truth[a]-truth[b]
                                pred=(item['x'][a]-item['x'][b])@current
                                credit=((pred>0)==(actual>0)).double()
                                credit[pred==0]=.5
                                row['PA']=100*credit[actual!=0].mean().item()
                            checkpoints.append(row)
                        if iteration==600: break
                        def closure():
                            nonlocal calls
                            optimizer.zero_grad()
                            loss=item['after'].loss(theta)
                            loss.backward()
                            calls+=1
                            return loss
                        t=timed()
                        optimizer.step(closure)
                        runtime+=timed()-t
                    trajectories[name]=checkpoints
                folder=OUTPUT/'E06'/assay/f'seed_{seed}'/head
                write_json(folder/'result.json',{'experiment':'E06','assay':assay,'seed':seed,'head':head,
                    'geometry':geometry,'trajectories':trajectories})
                print(json.dumps({'experiment':'E06','assay':assay,'seed':seed,'head':head}),flush=True)


def e07(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            for method in ['ProSIU','Original','Retrain','TARF','Entanglement','Heuristic','Random']:
                parameter_method='ProSIU' if method in ['Heuristic','Random'] else method
                parameters=saved_parameters('E01',assay,seed,'final_'+parameter_method)
                policy={'Heuristic':'first_program','Random':'random'}.get(method,'selector')
                kind='heuristic' if method=='Heuristic' else 'language'
                result=rollout(ep,parameters,kind=kind,policy=policy)
                write_json(OUTPUT/'E07'/assay/f'seed_{seed}'/method/'result.json',
                    {'experiment':'E07','assay':assay,'seed':seed,'method':method,**result})
                print(json.dumps({'experiment':'E07','assay':assay,'seed':seed,'method':method,'regret16':result['simple_regret_mean'][-1]}),flush=True)


def e08(assays,seeds):
    from .generate_banks import generate_banks
    generate_banks()
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed)
            train=torch.tensor(sorted(ep['req']['retained']+ep['req']['forgotten']),device=DEVICE)
            retain=torch.tensor(ep['req']['retained'],device=DEVICE)
            old=program_pairs(ep['stream'],train,train,seed)
            new=program_pairs(ep['stream'],retain,retain,seed)
            for kind in ['language','heuristic','random']:
                for bank_seed in range(3):
                    variant_ep=dict(ep,heads={h:dict(v) for h,v in ep['heads'].items()})
                    old_pairs=pairs_from_states(ep['stream'],old,kind,bank_seed,seed=seed)
                    new_pairs=pairs_from_states(ep['stream'],new,kind,bank_seed,seed=seed)
                    parameters,reference,details={},{},{}
                    for head,item in ep['heads'].items():
                        before=ExtendedObjective(item['x'],ep['stream']['eps'],train,expand_pairs(old_pairs,train,seed),item['cfg'])
                        after=ExtendedObjective(item['x'],ep['stream']['eps'],retain,expand_pairs(new_pairs,retain,seed),item['cfg'])
                        original,_=fit(before)
                        starts,_=initializations(item['x'],ep['req'],after,original)
                        parameters[head],details[head]=fit(after,starts['ProSIU'])
                        reference[head],_=fit(after,iterations=1000,strict=True)
                        variant_ep['heads'][head]['after']=after
                    result=rollout(variant_ep,parameters,kind=kind,bank_seed=bank_seed)
                    save_result('E08',variant_ep,kind+f'_{bank_seed}',parameters,reference,details,{'rollout':result})
            parameters=saved_parameters('E01',assay,seed,'final_ProSIU')
            result=rollout(ep,parameters,policy='direct')
            reference=saved_parameters('E01',assay,seed,'final_Retrain')
            save_result('E08',ep,'DirectRanking',parameters,reference,{}, {'rollout':result})


def standard_comparison(experiment,ep,methods,configuration_assay=None):
    original,reference,base=foundations(ep)
    assay=ep['assay'] if configuration_assay is None else configuration_assay
    for method in (METHOD_FILTER or methods):
        destination=OUTPUT/experiment/ep['assay']/f"seed_{ep['seed']}"/method/'result.json'
        if destination.exists() and not REPLACE: continue
        parameters,details={},{ }
        for head,item in ep['heads'].items():
            start=timed()
            if method=='Original': parameters[head]=original[head]
            elif method=='Retrain': parameters[head]=reference[head]
            elif method=='RetainedOpt': parameters[head],details[head]=fit(item['after'],original[head])
            elif method=='ProSIU':
                starts,geometry=initializations(item['x'],ep['req'],item['after'],original[head])
                parameters[head],details[head]=fit(item['after'],starts['ProSIU'])
                details[head]['geometry']=geometry
            else:
                setting=read_json(RESEARCH/'studies/epistasis_correction/frozen'/method/f'{assay}.json')['configuration']
                parameters[head],details[head]=update(method,item['after'],item['revoked'],original[head],setting,ep['seed'])
            details.setdefault(head,{})['seconds']=timed()-start
        save_result(experiment,ep,method,parameters,reference,details,{'foundation_diagnostics':base})


def e09(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            req=request(source['rows'],seed)
            req['shifts']=[0.]*len(req['forgotten'])
            ep=episode(data,truth,maps,source,assay,seed,req=req)
            standard_comparison('E09',ep,['ProSIU','SCRUB','TARF','Entanglement','RetainedOpt','Original','Retrain'])


def e10(assays,seeds):
    from .datasets import prepare_external,EXTERNAL,EXTERNAL_CONFIG
    directory=prepare_external()
    for assay in EXTERNAL:
        data,truth,maps,source=dataset(assay,directory)
        for seed in seeds:
            ep=episode(data,truth,maps,source,assay,seed,config=EXTERNAL_CONFIG)
            standard_comparison('E10',ep,['ProSIU','SCRUB','TARF','Entanglement','RetainedOpt','Original','Retrain'],configuration_assay='cgreGFP')


def e11(assays,seeds):
    from .datasets import prepare_splits
    folders=prepare_splits(assays)
    for assay in assays:
        for kind,folder in folders.items():
            data,truth,maps,source=dataset(assay,folder)
            for seed in seeds:
                ep=episode(data,truth,maps,source,assay+'_'+kind,seed,config=configuration(assay))
                standard_comparison('E11',ep,['ProSIU','TARF','Entanglement','RetainedOpt','Original','Retrain'],configuration_assay=assay)


def e12(assays,seeds):
    for assay in assays:
        data,truth,maps,source=dataset(assay)
        for seed in seeds:
            final_req=request(source['rows'],seed)
            initial_ep=episode(data,truth,maps,source,assay,seed,req=final_req)
            original=saved_parameters('E01',assay,seed,'final_Original')
            stream=initial_ep['stream']
            full_training=sorted(final_req['retained']+final_req['forgotten'])
            shifts=dict(zip(final_req['forgotten'],final_req['shifts']))
            groups={}
            for i in final_req['forgotten']:
                groups.setdefault(tuple(source['rows'][i]['positions']),[]).append(i)
            group_keys=sorted(groups)
            reference_cache={}
            for batches in [1,2,4,8]:
                for order_id in range(1 if batches==1 else 3):
                    names=['ProSIU16','RetainedOpt16','ProSIUConverged','RetainedOptConverged']
                    completed=[OUTPUT/'E12'/assay/f'seed_{seed}'/f'{name}_B{batches}_O{order_id}'/'parameters.pt' for name in names]
                    if not REPLACE and all(path.exists() for path in completed): continue
                    keys=group_keys.copy()
                    if order_id==1: keys.reverse()
                    elif order_id==2: random.Random(78000+seed).shuffle(keys)
                    chunks=[keys[j*len(keys)//batches:(j+1)*len(keys)//batches] for j in range(batches)]
                    states={name:{h:t.clone() for h,t in original.items()} for name in names}
                    traces={name:[] for name in states}
                    cumulative=[]
                    for round_id,chunk in enumerate(chunks):
                        cumulative.extend(i for key in chunk for i in groups[key])
                        req=copy.deepcopy(final_req)
                        req['forgotten']=sorted(cumulative)
                        req['retained']=sorted(set(full_training)-set(cumulative))
                        req['shifts']=[shifts[i] for i in req['forgotten']]
                        req['selected_position_pairs']=keys[:sum(len(c) for c in chunks[:round_id+1])]
                        touched={p for key in req['selected_position_pairs'] for p in key}
                        req['adjacent']=[i for i,r in enumerate(source['rows']) if any(p in touched for p in r['positions'])]
                        ep=episode(data,truth,maps,source,assay,seed,req=req,stream_override=stream)
                        reference_key=tuple(req['retained'])
                        if reference_key not in reference_cache:
                            reference_cache[reference_key]={h:fit(item['after'],iterations=1000,strict=True)[0] for h,item in ep['heads'].items()}
                        reference=reference_cache[reference_key]
                        for name,parameters in states.items():
                            elapsed=0.
                            for head,item in ep['heads'].items():
                                start=timed()
                                theta=parameters[head]
                                if name.startswith('ProSIU'):
                                    starts,_=initializations(item['x'],req,item['after'],theta)
                                    theta=starts['ProSIU']
                                parameters[head],_=fit(item['after'],theta,iterations=16 if name.endswith('16') else 600)
                                elapsed+=timed()-start
                            metrics,details=evaluate(ep,parameters,reference)
                            traces[name].append({'round':round_id+1,'withdrawn_count':len(cumulative),'metrics':metrics,
                                'equivalence':details,'seconds':elapsed})
                    for name,parameters in states.items():
                        save_result('E12',ep,f'{name}_B{batches}_O{order_id}',parameters,reference,{},
                            {'batches':batches,'order':order_id,'trajectory':traces[name],
                             'original_full_offset_stream_preserved':True,'sequence_group_keys':keys})


def main():
    global METHOD_FILTER,REPLACE
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=[f'E{i:02}' for i in range(1,13)])
    parser.add_argument('--assays',nargs='+',default=ASSAYS)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--replace',action='store_true')
    args=parser.parse_args()
    METHOD_FILTER,REPLACE=args.methods,args.replace
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    globals()[args.experiment.lower()](args.assays,args.seeds)


if __name__=='__main__':
    main()
