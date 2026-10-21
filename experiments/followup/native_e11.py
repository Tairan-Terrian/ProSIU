"""Published protein predictors with the accepted retained-evidence adapter."""
import json
import torch

from .core import (OUTPUT, RESEARCH, ASSAYS, SEEDS, DEVICE, dataset, request, observed,
                   program_pairs, fit, timed, write_json, read_json)
from legacy_prosiu.correction_model import build_problem, comparison_features
from legacy_prosiu.accuracy_baselines import update

METHODS=['Kermut','Metalic','ProteinTTT']
STUDY=RESEARCH/'studies/epistasis_correction'


def inputs(data,truth,source,assay,seed,method):
    original=read_json(STUDY/'data'/f'{assay}.json')
    lookup={r['q'][3]:i for i,r in enumerate(original['rows'])}
    ids=torch.tensor([lookup[r['q'][3]] for r in source['rows']],device=DEVICE)
    native=torch.load(STUDY/'priors'/method/f'{assay}_{seed}.pt',map_location=DEVICE,weights_only=False)
    req=request(source['rows'],seed)
    stream=observed(data,truth,req)
    stream['prior']=(native['prediction'][ids].double()-data['anchor'])/data['scale']
    return stream,req,read_json(STUDY/'frozen'/method/f'{assay}.json'),native['metadata']


def scores(stream,truth,req,theta):
    prediction=stream['prior']+stream['x']@theta
    metrics={}
    for name,ids in [('CA',req['forgotten']),('RA',req['retained']),('TA',stream['partitions']['test'])]:
        labels=truth[ids]>0
        decisions=prediction[ids]>=0
        metrics[name]=50*(decisions[labels].double().mean()+(~decisions[~labels]).double().mean()).item()
    retain=torch.tensor(req['retained'],device=DEVICE)
    bank=program_pairs(stream,stream['partitions']['test'],retain,10000+req['seed'])
    features,offset=comparison_features(stream,bank,retain)
    ordering=offset+features@theta
    target=truth[bank['first']]-truth[bank['second']]
    credit=((ordering>0)==(target>0)).double()
    credit[ordering==0]=.5
    metrics['PA']=100*credit[target!=0].mean().item()
    return metrics,bank


def main():
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    for assay in ASSAYS:
        for kind in ['Random','PositionPair','SeenConstituents','UnseenConstituent']:
            data,truth,maps,source=dataset(assay,OUTPUT/'extrapolation_data'/kind)
            name=assay+'_'+kind
            for seed in SEEDS:
                for method in METHODS:
                    stream,req,settings,metadata=inputs(data,truth,source,assay,seed,method)
                    before,after,revoked=build_problem(stream,req,settings['task'],seed)
                    original,old_info=fit(before)
                    reference,ref_info=fit(after,iterations=1000,strict=True)
                    start=timed()
                    theta,details=update(method,after,revoked,original,settings['configuration'],seed)
                    seconds=timed()-start
                    metrics,bank=scores(stream,truth,req,theta)
                    residual=after.gradient(theta).norm().item()
                    equivalence={'parameter_distance':(theta-reference).norm().item(),
                        'residual_certificate':residual/after.mu,'gradient':residual,
                        'objective_gap':(after.loss(theta)-after.loss(reference)).item()}
                    for tag,ids in [('F',req['forgotten']),('R',req['retained']),('H',data['partitions']['test'])]:
                        equivalence[tag+'_prediction_rms']=(data['x'][ids]@(theta-reference)).square().mean().sqrt().item()
                    record={'experiment':'E11','assay':name,'seed':seed,'variant':method,
                        'learner':'native','metrics':metrics,'equivalence':{'shared':equivalence},
                        'configuration':settings,'native_metadata':metadata,
                        'diagnostics':{'original':old_info,'reference':ref_info,'adapter':details,'seconds':seconds},
                        'comparison_bank':{k:bank[k].cpu().tolist() for k in ['first','second']},
                        'adaptation':'unchanged constituent-only native predictor; refitted retained-evidence residual adapter'}
                    folder=OUTPUT/'E11'/name/f'seed_{seed}'/method
                    write_json(folder/'result.json',record)
                    torch.save({'shared':theta.cpu()},folder/'parameters.pt')
                    torch.save({'original':original.cpu(),'reference':reference.cpu()},folder/'reference_parameters.pt')
                    print(json.dumps({'experiment':'E11','assay':name,'seed':seed,'variant':method,'metrics':metrics}),flush=True)


if __name__=='__main__': main()
