"""Measured-pool sequential decisions with per-trajectory observation histories."""
import torch
from prosiu.common import DEVICE,rng
from .programs import proposals
from .core import predictions


def rollout(ep,parameters,kind='language',bank_seed=0,policy='selector',episodes=64,budget=16):
    data=dict(ep['stream'],eps=ep['stream']['eps'].clone())
    candidate_ids=ep['data']['partitions']['test']
    retained=torch.tensor(ep['req']['retained'],device=DEVICE)
    generator=rng(76000+ep['seed'])
    pools=torch.stack([candidate_ids[torch.randperm(len(candidate_ids),generator=generator,device=DEVICE)[:128]] for _ in range(episodes)])
    histories=torch.stack([retained[torch.randperm(len(retained),generator=generator,device=DEVICE)[:16]] for _ in range(episodes)])
    available=torch.ones_like(pools,dtype=torch.bool)
    _,utility=predictions(ep,parameters)
    query_ids,values,unique_proposals=[],[],[]
    for step in range(budget):
        proposed=proposals(data,pools,histories,kind,bank_seed,available=available)
        unique_proposals.append(torch.tensor([len(ids.unique()) for ids in proposed],device=DEVICE))
        if policy=='random':
            priorities=torch.rand(pools.shape,generator=generator,device=DEVICE).masked_fill(~available,-torch.inf)
            selected=pools.gather(1,priorities.argmax(1)[:,None]).squeeze(1)
        elif policy=='direct':
            priorities=utility[pools].masked_fill(~available,-torch.inf)
            selected=pools.gather(1,priorities.argmax(1)[:,None]).squeeze(1)
        elif policy=='first_program':
            selected=proposed[:,0]
        else:
            selected=proposed.gather(1,utility[proposed].argmax(1)[:,None]).squeeze(1)
        query_ids.append(selected)
        values.append(ep['truth'][selected])
        # Only selected measurements are released to the observation history.
        data['eps'][selected]=ep['truth'][selected]
        histories=torch.cat([histories,selected[:,None]],1)
        available &= pools!=selected[:,None]
    queries=torch.stack(query_ids,1)
    measured=torch.stack(values,1)
    best=measured.cummax(1).values
    pool_truth=ep['truth'][pools]
    maximum=pool_truth.max(1).values
    threshold=pool_truth.topk(13,dim=1).values[:,-1]
    regret=maximum[:,None]-best
    ranks=(pool_truth[:,:,None]<=best[:,None,:]).double().mean(1)*100
    hit=(best>=threshold[:,None]).double()*100
    return {'budget':list(range(1,budget+1)),
        'best_mean':best.mean(0).cpu().tolist(),'simple_regret_mean':regret.mean(0).cpu().tolist(),
        'best_percentile_mean':ranks.mean(0).cpu().tolist(),'top_decile_hit_percent':hit.mean(0).cpu().tolist(),
        'best_curve_area':best.mean().item(),'proposal_unique_mean':torch.stack(unique_proposals,1).double().mean().item(),
        'query_ids':queries.cpu().tolist(),'candidate_ids':pools.cpu().tolist(),
        'per_trajectory_best':best.cpu().tolist(),'per_trajectory_regret':regret.cpu().tolist(),
        'episodes':episodes,'initial_history_size':16,'policy':policy,'bank':kind,'bank_seed':bank_seed}
