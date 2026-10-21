"""Trace a fixed avGFP evidence withdrawal and held-out query trajectory."""
import torch

from followup import core
from followup.programs import proposals
from prosiu.common import ASSETS, configuration, read_json, write_json, rng
from prosiu.experiment import fit_episode
from prosiu.epistatic_program_selection import program_pairs, expand_pairs

OUTPUT=core.RESEARCH/'studies/case_study_20261020'


def describe(source, truth, record_id):
    row=source['rows'][record_id]
    sequence=row['q'][3]
    mutations=[{'position':p+1,'wild_type':source['wt'][p],'substitution':sequence[p],
                'label':f'{source["wt"][p]}{p+1}{sequence[p]}',
                'window_start':max(0,p-4)+1,
                'window':sequence[max(0,p-4):p+5]} for p in row['positions']]
    return {'record_id':record_id,'partition':row['partition'],'mutations':mutations,
            'sequence':sequence,'measurements':row['y'],'normalized_interaction':truth[record_id].item(),
            'single_effects':[row['y'][i]-row['y'][0] for i in [1,2]]}


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    assay,seed,trajectory='avGFP',0,0
    data,truth,maps,source=core.dataset(assay)
    ep=core.episode(data,truth,maps,source,assay,seed)
    predicted,checkpoint,diagnostics=fit_episode(ep['stream'],ep['req'],maps,configuration(assay))
    original={h:p['original'].to(core.DEVICE) for h,p in checkpoint.items()}
    parameters={h:p['theta'].to(core.DEVICE) for h,p in checkpoint.items()}
    original_sign,original_utility=core.predictions(ep,original)
    updated_sign,utility=core.predictions(ep,parameters)
    train=torch.tensor(sorted(ep['req']['retained']+ep['req']['forgotten']),device=core.DEVICE)
    retain=torch.tensor(ep['req']['retained'],device=core.DEVICE)
    forgotten=torch.tensor(ep['req']['forgotten'],device=core.DEVICE)
    old=program_pairs(ep['stream'],train,train,seed)
    new=program_pairs(ep['stream'],retain,retain,seed)
    old_pairs=expand_pairs(old,train,seed)
    new_pairs=expand_pairs(new,retain,seed)
    affected_endpoint=torch.isin(old_pairs['first'],forgotten)|torch.isin(old_pairs['second'],forgotten)
    affected_history=torch.isin(old['history'][old['state']],forgotten).any(1)
    dependency_affected=affected_endpoint.clone()
    dependency_affected[:len(affected_history)] |= affected_history
    focus_id=min(ep['req']['forgotten'])
    focus=describe(source,truth,focus_id)
    focus.update({'observed_interaction':ep['stream']['eps'][focus_id].item(),
                  'old_sign_probability':torch.sigmoid(original_sign[focus_id]).item(),
                  'updated_sign_probability':torch.sigmoid(updated_sign[focus_id]).item()})
    stream=dict(ep['stream'],eps=ep['stream']['eps'].clone())
    candidates=ep['data']['partitions']['test']
    generator=rng(76000+seed)
    pools=torch.stack([candidates[torch.randperm(len(candidates),generator=generator,device=core.DEVICE)[:128]] for _ in range(64)])
    histories=torch.stack([retain[torch.randperm(len(retain),generator=generator,device=core.DEVICE)[:16]] for _ in range(64)])
    initial_history=histories[trajectory].cpu().tolist()
    available=torch.ones_like(pools,dtype=torch.bool)
    programs=read_json(ASSETS/'programs.json')
    trace=[]
    original_first=None
    for step in range(4):
        proposed=proposals(stream,pools,histories,'language',0,available=available)
        winning=utility[proposed].argmax(1)
        selected=proposed.gather(1,winning[:,None]).squeeze(1)
        if step==0:
            original_first=int(proposed[trajectory,original_utility[proposed[trajectory]].argmax()])
        selected_id=int(selected[trajectory])
        row=describe(source,truth,selected_id)
        row.update({'query':step+1,'program_index':int(winning[trajectory]),
                    'program_name':programs[int(winning[trajectory])]['name'],
                    'program':programs[int(winning[trajectory])],
                    'model_utility':utility[selected_id].item(),
                    'predicted_positive_probability':torch.sigmoid(updated_sign[selected_id]).item(),
                    'proposed_ids':proposed[trajectory].cpu().tolist(),
                    'history_before':histories[trajectory].cpu().tolist()})
        trace.append(row)
        stream['eps'][selected]=truth[selected]
        histories=torch.cat([histories,selected[:,None]],1)
        available &= pools!=selected[:,None]
    prior=read_json(core.RESEARCH/'studies/followup_main12/E07/avGFP/seed_0/ProSIU/result.json')
    main_ref=next(r for r in read_json(core.ROOT/'reference/results.json')['runs'] if r['assay']==assay and r['seed']==seed)
    metrics,_=core.evaluate(ep,parameters)
    report={'assay':assay,'seed':seed,'trajectory':trajectory,'display_queries':[1,2,3],
            'selection_rule':'fixed avGFP, request seed 0, trajectory 0; smallest withdrawn record identity',
            'source_identifier':source['identifier'],'wild_type':source['wt'],'focus':focus,
            'withdrawn_records':len(forgotten),'retained_records':len(retain),
            'anchor_count':len(source['anchors']),'normalized_anchor_drift':0.,
            'comparison_counts':{'before':len(old_pairs['first']),'rebuilt':len(new_pairs['first']),
                                 'affected_endpoints':int(affected_endpoint.sum()),
                                 'affected_including_history':int(dependency_affected.sum())},
            'initial_history':initial_history,'candidate_pool':pools[trajectory].cpu().tolist(),
            'old_model_first_choice_on_common_state':describe(source,truth,original_first),
            'queries':trace,'metrics':metrics,'diagnostics':diagnostics,
            'request':ep['req'],'verification':{
                'trace_matches_existing_episode': [r['record_id'] for r in trace]==prior['query_ids'][trajectory][:4],
                'full_model_main_metric_difference':max(abs(metrics[k]-main_ref['metrics'][k]) for k in metrics),
                'all_queries_held_out':all(r['partition']=='test' for r in trace),
                'unique_queries':len({r['record_id'] for r in trace})==len(trace)}}
    write_json(OUTPUT/'case.json',report)
    torch.save(checkpoint,OUTPUT/'checkpoint.pt')
    print({'verification':report['verification'],'focus':[m['label'] for m in focus['mutations']],
           'queries':[[m['label'] for m in r['mutations']] for r in trace]},flush=True)


if __name__=='__main__':
    main()
