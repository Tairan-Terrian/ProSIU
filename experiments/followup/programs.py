"""Numerical program execution on explicitly observed evidence states."""
from pathlib import Path
import torch
from prosiu.common import ASSETS,DEVICE,read_json,rng


def bank(kind='language',seed=0):
    if kind=='language':
        source=ASSETS/'programs.json' if seed==0 else Path('/media/yjz/htr/ProSIU/studies/followup_main12/program_banks')/f'bank_{seed}'/'programs.json'
        rows=read_json(source)
        weights=torch.tensor([r['weights'] for r in rows],device=DEVICE,dtype=torch.float64)
        ranks=torch.tensor([r['rank'] for r in rows],device=DEVICE)
    elif kind=='heuristic':
        axes=torch.eye(5,device=DEVICE,dtype=torch.float64)
        weights=torch.cat([axes,-axes,axes[:3]+axes[1:4],axes[:3]-axes[1:4]])
        ranks=(torch.arange(16,device=DEVICE)+seed)%8
        weights=weights.roll(seed,dims=0)
    else:
        generator=rng(74100+seed)
        weights=torch.randn(16,5,generator=generator,device=DEVICE,dtype=torch.float64)
        weights=weights/weights.norm(dim=1,keepdim=True)
        ranks=torch.randint(8,(16,),generator=generator,device=DEVICE)
    return weights,ranks


def proposals(data,pools,histories,kind='language',bank_seed=0,valid_history=None,available=None):
    weights,ranks=bank(kind,bank_seed)
    similarity=torch.exp(-torch.cdist(data['x'][pools],data['x'][histories]).square()/8)
    if valid_history is not None:
        similarity*=valid_history[:,None,:]
    density=similarity.sum(-1)
    local=(similarity@data['eps'][histories].unsqueeze(-1)).squeeze(-1)/(1+density)
    effects=(data['y'][:,1:3]-data['y'][:,:1])/data['scale']
    operands=torch.stack([data['offset'][pools],local,1/(1+density),
        -(effects[:,0]-effects[:,1]).abs()[pools],
        ((data['positions'][:,0]-data['positions'][:,1]).abs().double()/(data['positions'].max()+1))[pools]],-1)
    priority=operands@weights.T
    if available is not None:
        priority=priority.masked_fill(~available[:,:,None],-torch.inf)
    order=priority.argsort(1,descending=True)
    return pools.gather(1,order.gather(1,ranks[None,None].expand(len(pools),1,-1)).squeeze(1))


def pairs_from_states(data,records,kind='language',bank_seed=0,retained=None,seed=0):
    pools,histories=records['candidate_pools'],records['history']
    valid_history=available=None
    if retained is not None:
        valid_history=torch.isin(histories,retained)
        available=torch.isin(pools,retained)
    selected=proposals(data,pools,histories,kind,bank_seed,valid_history,available)
    indices=torch.combinations(torch.arange(16,device=DEVICE),r=2)
    first=selected[:,indices[:,0]].flatten()
    second=selected[:,indices[:,1]].flatten()
    state=torch.arange(len(pools),device=DEVICE)[:,None].expand(-1,len(indices)).flatten()
    keep=first!=second
    first,second,state=first[keep],second[keep],state[keep]
    chosen=torch.randperm(len(first),generator=rng(650300+seed),device=DEVICE)[:2048]
    return {'first':first[chosen],'second':second[chosen],'state':state[chosen],
        'history':histories,'candidate_pools':pools,'selected_program_candidates':selected}
