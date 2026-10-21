"""One-module ablation of the frozen ProSIU release, using CUDA throughout."""
import argparse
import copy
from pathlib import Path
import torch

from followup.core import dataset, episode, initializations, evaluate, fit
from followup.core import ROOT, RESEARCH, DEVICE, ASSAYS, ExtendedObjective
from prosiu.common import write_json
from prosiu.epistatic_program_selection import program_pairs, expand_pairs
from prosiu.mutation_preference_optimization import SelectorObjective
from prosiu.selective_epistasis_forgetting import selective_forgetting

OUTPUT = RESEARCH/'studies/followup_revision_20261019/ablation'
VARIANTS = ['ProSIU','w_o_EPS','w_o_MPO','w_o_SEF']


def objectives(ep, variant):
    train = torch.tensor(sorted(ep['req']['retained']+ep['req']['forgotten']),device=DEVICE)
    retained = torch.tensor(ep['req']['retained'],device=DEVICE)
    pairs = {}
    for name, ids in [('before',train),('after',retained)]:
        programs = program_pairs(ep['stream'],ids,ids,ep['seed'])
        if variant == 'w_o_EPS':
            # EPS supplies program-derived pairs. The original uniform pairs remain.
            programs = {'first':ids[:0], 'second':ids[:0]}
        pairs[name] = expand_pairs(programs,ids,ep['seed'])
    altered = dict(ep,heads={h:dict(v) for h,v in ep['heads'].items()})
    counts = {name:len(value['first']) for name,value in pairs.items()}
    for head,item in altered['heads'].items():
        cfg = copy.deepcopy(item['cfg'])
        if variant == 'w_o_MPO':
            # Replace the sign/preference objective with measured-interaction regression.
            cfg.update(classification=0.,preference=0.,calibration=1.,rank_temperature=1.,threshold=0.)
        item['cfg'] = cfg
        item['before'] = SelectorObjective(item['x'],ep['stream']['eps'],train,pairs['before'],cfg)
        item['after'] = SelectorObjective(item['x'],ep['stream']['eps'],retained,pairs['after'],cfg)
    return altered,counts


def run(assays,seeds):
    for assay in assays:
        data,truth,maps,source = dataset(assay)
        for seed in seeds:
            base = episode(data,truth,maps,source,assay,seed)
            for variant in VARIANTS:
                ep,counts = objectives(base,variant)
                parameters,short,diagnostics = {},{},{}
                for head,item in ep['heads'].items():
                    original,old = item['before'].fit(iterations=600)
                    if variant == 'w_o_SEF':
                        start = original
                        parameters[head],details = item['after'].fit(start,iterations=600)
                        geometry = {}
                    else:
                        parameters[head],details = selective_forgetting(item['x'],ep['req'],item['after'],original)
                        starts,geometry = initializations(item['x'],ep['req'],item['after'],original)
                        start = starts['ProSIU']
                    short[head],short_info = item['after'].fit(start,iterations=16)
                    diagnostics[head] = {'before':old,'converged':details,'sixteen_steps':short_info,
                                         'geometry':geometry,'configuration':item['cfg']}
                final_metrics,_ = evaluate(ep,parameters)
                early_metrics,_ = evaluate(ep,short)
                folder = OUTPUT/assay/f'seed_{seed}'/variant
                record = {'experiment':'E05','assay':assay,'seed':seed,'variant':variant,
                    'metrics':final_metrics,'metrics16':early_metrics,'pair_counts':counts,
                    'diagnostics':diagnostics,'request':ep['req'],
                    'frozen_release':'src/prosiu; full model calls selective_forgetting directly'}
                write_json(folder/'result.json',record)
                torch.save({'converged':{h:p.cpu() for h,p in parameters.items()},
                            'sixteen_steps':{h:p.cpu() for h,p in short.items()}},folder/'parameters.pt')
                print(f'{assay} seed={seed} {variant}: {final_metrics}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--assays',nargs='+',default=ASSAYS)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    run(args.assays,args.seeds)
