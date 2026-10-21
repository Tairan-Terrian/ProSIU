"""Shared CUDA objectives, evaluation, and experiment records.

Published baseline adapters retain the attribution in ProSIU/references and
ProSIU/prosiu/baseline_defaults.py. The released ProSIU implementation is reused.
"""
import copy
import importlib.util
import json
import math
from functools import cached_property
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from prosiu.common import DEVICE, configuration, read_json, write_json, rng
from prosiu.data import load, observed, request
from prosiu.features import build_selector_features
from prosiu.epistatic_program_selection import program_pairs, expand_pairs
from prosiu.mutation_preference_optimization import SelectorObjective
from prosiu.metrics import evaluate_scores, balanced_accuracy

ROOT = Path('/media/yjz/htr/ProSIU-main')
RESEARCH = Path('/media/yjz/htr/ProSIU')
OUTPUT = RESEARCH / 'studies/followup_main12'
DATA = ROOT / 'data'
ASSAYS = ['amacGFP', 'cgreGFP', 'avGFP']
SEEDS = list(range(5))
GENERAL = ['SCRUB', 'SalUn', 'SFR-on', 'NegMerge', 'SatImp', 'TARF', 'Entanglement']

spec = importlib.util.spec_from_file_location('legacy_prosiu', RESEARCH/'prosiu/__init__.py',
    submodule_search_locations=[str(RESEARCH/'prosiu')])
legacy = importlib.util.module_from_spec(spec)
sys.modules['legacy_prosiu'] = legacy
spec.loader.exec_module(legacy)
from legacy_prosiu.correction_model import build_problem, Likelihood as LegacyLikelihood
from legacy_prosiu.accuracy_baselines import update as legacy_update
from legacy_prosiu.baselines import derivative, negative_merge, sample_fisher
from legacy_prosiu.neutral_baselines import entanglement


def timed():
    torch.cuda.synchronize()
    return time.perf_counter()


def fit(objective, start=None, iterations=600, strict=False):
    dimension = objective.a.shape[1]
    theta = objective.a.new_zeros(dimension) if start is None else start.detach().clone()
    theta.requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta], lr=1., max_iter=iterations,
        history_size=30, tolerance_grad=1e-8, tolerance_change=1e-11,
        line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        optimizer.zero_grad()
        value = objective.loss(theta)
        value.backward()
        calls += 1
        return value
    optimizer.step(closure)
    result = theta.detach()
    newton_steps = 0
    if strict:
        for _ in range(8):
            gradient = objective.gradient(result)
            if gradient.norm().item() <= 1e-9:
                break
            direction = torch.linalg.solve(objective.hessian(result), gradient)
            step = 1.
            loss = objective.loss(result)
            while objective.loss(result-step*direction) > loss + 1e-13:
                step *= .5
            result -= step*direction
            newton_steps += 1
    return result, {'gradient': objective.gradient(result).norm().item(),
        'loss': objective.loss(result).item(), 'iterations': optimizer.state[theta]['n_iter'],
        'function_evaluations': calls, 'reference_newton_steps': newton_steps}


def initializations(x, req, objective, original):
    start = timed()
    gradient = objective.gradient(original)
    newton = torch.linalg.solve(objective.hessian(original), -gradient)
    after_newton = timed()
    ids = sorted(set(req['retained']) & set(req['adjacent']))
    protected, target = x[ids], x[req['forgotten']]
    metric = .01*torch.eye(x.shape[1], device=DEVICE, dtype=x.dtype)
    if len(protected):
        metric = metric + protected.T@protected/len(protected)
    inverse = torch.linalg.solve(metric, target.T)
    constrained = inverse @ (torch.linalg.pinv(target@inverse, rtol=1e-9, hermitian=True)@(target@newton))
    after_protection = timed()
    return {'Original':original, 'Zero':torch.zeros_like(original),
        'Newton':original+newton, 'Protected':original+constrained,
        'ProSIU':original+.75*newton+.25*constrained}, {
        'newton_seconds':after_newton-start, 'protection_seconds':after_protection-after_newton,
        'constraint_residual':(target@(constrained-newton)).norm().item(),
        'newton_cost':(newton@metric@newton).item(),
        'protected_cost':(constrained@metric@constrained).item(), 'protected_count':len(ids)}


class ExtendedObjective(SelectorObjective):
    def __init__(self, x, eps, ids, records, cfg):
        super().__init__(x, eps, ids, records, cfg)
        self.n_measurements = len(ids)
    @cached_property
    def lipschitz(self):
        x=self.x
        bound = (self.mu*torch.eye(x.shape[1],device=DEVICE,dtype=x.dtype)
            + self.a.T@(self.weights[:,None]*self.a)/4
            + self.calibration/len(self.x)*(self.x.T@self.x))
        return torch.linalg.eigvalsh(bound)[-1].item()

    def preference_losses(self, theta):
        logits = self.a@theta
        return self.p*F.softplus(-logits,threshold=40)+(1-self.p)*F.softplus(logits,threshold=40)

    def calibration_losses(self,theta):
        return torch.sqrt(1+(self.x@theta-self.calibration_target).square())-1


class TaskLikelihood:
    """Active binary channels and the normalized pseudo-Huber measurement density."""
    def __init__(self, objective):
        active = objective.weights > 0
        self.a, self.p = objective.a[active], objective.p[active]
        if objective.calibration > 0:
            self.x, self.y = objective.x, objective.calibration_target
        else:
            self.x = objective.x[:0]
            self.y = objective.calibration_target[:0]
        weights = torch.cat([self.a.new_full((len(self.x),), objective.calibration/len(objective.x)),objective.weights[active]])
        self.weights = weights/weights.sum()
        self.features = torch.cat([self.x,self.a])
        self.n = len(weights)
        self.measurement_grid=torch.linspace(-32.,32.,513,device=DEVICE,dtype=self.x.dtype)
        self.measurement_energy=torch.sqrt(1+self.measurement_grid.square())
        density=torch.exp(-self.measurement_energy)
        self.measurement_quadrature=density/density.sum()
        bound = self.x.T@(self.weights[:len(self.x),None]*self.x)
        bound += self.a.T@(self.weights[len(self.x):,None]*self.a)/4
        self.lipschitz = torch.linalg.eigvalsh(bound)[-1].item()

    def losses(self,theta):
        residual = self.x@theta-self.y
        normalization = math.log(2*math.e*0.6019072301972346)
        measurement = torch.sqrt(1+residual.square())-1+normalization
        logits = self.a@theta
        return torch.cat([measurement,self.p*F.softplus(-logits,threshold=40)+(1-self.p)*F.softplus(logits,threshold=40)])

    def mean_loss(self,theta):
        return self.weights@self.losses(theta)

    def conditional_loss(self,theta,indices):
        return self.weights[indices]@self.losses(theta)[indices]/self.weights[indices].sum()

    def teacher_kl(self,theta,teacher):
        shift=self.x@(teacher-theta)
        measurement=(torch.sqrt(1+(shift[:,None]+self.measurement_grid).square())
            -self.measurement_energy)@self.measurement_quadrature
        logits, reference = self.a@theta,self.a@teacher
        probability = reference.sigmoid()
        binary = probability*(F.softplus(-logits)-F.softplus(-reference))+(1-probability)*(F.softplus(logits)-F.softplus(reference))
        return self.weights@torch.cat([measurement,binary])

    def randomized_loss(self,theta,generator):
        order = torch.randperm(len(self.y),generator=generator,device=DEVICE)
        measurement = torch.sqrt(1+(self.x@theta-self.y[order]).square())-1
        labels = 1-(self.p >= .5).double()
        logits = self.a@theta
        return self.weights@torch.cat([measurement,F.softplus((1-2*labels)*logits,threshold=40)])

    def preference_residual(self,theta):
        return (self.a@theta).sigmoid()-self.p

    def fisher(self,theta):
        residual=self.x@theta-self.y
        measurement=((residual/torch.sqrt(1+residual.square()))[:,None]*self.x).square()
        binary=(self.preference_residual(theta)[:,None]*self.a).square()
        return (self.weights[:,None]*torch.cat([measurement,binary])).sum(0)


def update(method,objective,revoked,original,setting,seed,learner='final'):
    if learner=='common':
        return legacy_update(method,objective,revoked,original,setting,seed,likelihood_type=LegacyLikelihood)
    forget,retain = TaskLikelihood(revoked),TaskLikelihood(objective)
    if method=='NegMerge':
        members=[{'steps':n,'learning_rate_factor':eta} for n in [40,160,640] for eta in [.5,1.]]
        vector,details=negative_merge(forget,original,objective,members)
        return original-setting['negation_scale']*vector,details
    parameters=setting['algorithm']
    strength=setting['forget_strength']
    eta=setting['learning_rate_factor']/(objective.lipschitz+strength*forget.lipschitz)
    if method=='SCRUB':
        eta=setting['learning_rate_factor']/(parameters['retain_weight']*objective.lipschitz
            +parameters['retain_kl_weight']*retain.lipschitz+strength*forget.lipschitz)
    if method=='Entanglement':
        return entanglement(forget,retain,original,setting,eta)
    theta=original.clone()
    generator=rng(seed+6701)
    if method=='SalUn':
        theta.requires_grad_(True)
        saliency=derivative(forget.mean_loss(theta),theta).abs()
        selected=saliency.argsort(descending=True)[:int(len(theta)*parameters['salient_fraction'])]
        mask=torch.zeros_like(theta)
        mask[selected]=1
        theta=theta.detach()
    if method=='SFR-on':
        mask=forget.fisher(theta)>=parameters['fisher_threshold']*retain.fisher(theta)
        velocity=torch.zeros_like(theta)
    for iteration in range(setting['steps']):
        theta.requires_grad_(True)
        if method=='SCRUB':
            if iteration<setting['steps']*parameters['max_fraction']:
                ascent=derivative(-strength*forget.teacher_kl(theta,original),theta)
                theta=(theta-eta*ascent).detach().requires_grad_(True)
            loss=parameters['retain_weight']*objective.loss(theta)+parameters['retain_kl_weight']*retain.teacher_kl(theta,original)
        elif method=='SalUn':
            loss=objective.loss(theta)+strength*forget.randomized_loss(theta,generator)
        elif method=='SFR-on':
            if iteration%parameters['frequency']==0:
                losses=forget.losses(theta)
                weights=torch.softmax(forget.weights.log()-parameters['loss_exponent']*losses.detach().log(),dim=0)
                adaptive=-strength*(1+math.cos(math.pi*iteration/setting['steps']))/2*(weights*losses).sum()
                fast=derivative(adaptive,theta)*mask
                fast*=torch.clamp(parameters['clip_norm']/fast.norm(),max=1)
                velocity=parameters['momentum']*velocity+fast
                theta=(theta-eta*velocity).detach().requires_grad_(True)
            loss=objective.loss(theta)
        elif method=='SatImp':
            logp=-forget.losses(theta)
            p=logp.exp().detach()
            weights=p.pow(parameters['beta1'])*(1-p).pow(parameters['beta2'])
            loss=parameters['retain_weight']*objective.loss(theta)+strength*(forget.weights@(weights*logp))
        else:
            losses=objective.preference_losses(theta)
            fraction=parameters['retain_fraction']
            selected=[]
            for begin,end in [(0,objective.n_measurements),(objective.n_measurements,len(losses))]:
                size=int((end-begin)*fraction)
                selected.append(begin+losses[begin:end].detach().argsort()[:size])
            indices=torch.cat(selected)
            calibration=objective.calibration_losses(theta)
            mi=calibration.detach().argsort()[:int(len(calibration)*fraction)]
            annealing=max(1-iteration/(setting['steps']*parameters['annealing_fraction']),0.)
            loss=(objective.weights[indices]@losses[indices]/fraction+objective.calibration*calibration[mi].mean()
                +objective.mu*theta.square().sum()/2-strength*annealing*forget.mean_loss(theta))
        gradient=derivative(loss,theta)
        if method=='SalUn': gradient*=mask
        if method=='SFR-on':
            velocity=parameters['momentum']*velocity+gradient
            gradient=velocity
        theta=(theta-eta*gradient).detach()
    return theta,{'learning_rate':eta,'steps':setting['steps'],'gradient':objective.gradient(theta).norm().item()}


def dataset(assay,data_dir=DATA):
    data,truth=load(assay,data_dir)
    maps=build_selector_features(assay,data,data_dir)
    source=read_json(Path(data_dir)/'processed'/f'{assay}.json')
    return data,truth,maps,source


def episode(data,truth,maps,source,assay,seed,learner='final',req=None,config=None,stream_override=None):
    req=request(source['rows'],seed) if req is None else req
    stream=observed(data,truth,req) if stream_override is None else dict(stream_override)
    stream['prior']=torch.zeros_like(truth)
    train=torch.tensor(sorted(req['retained']+req['forgotten']),device=DEVICE)
    retained=torch.tensor(req['retained'],device=DEVICE)
    forgotten=torch.tensor(req['forgotten'],device=DEVICE)
    evaluation=program_pairs(stream,data['partitions']['test'],retained,10000+seed)
    heads={}
    if learner=='common':
        task=read_json(RESEARCH/'studies/epistasis_correction/frozen/SCRUB'/f'{assay}.json')['task']
        before,after,revoked=build_problem(stream,req,task,seed)
        heads['shared']={'x':data['x'],'before':before,'after':after,'revoked':revoked,'cfg':task}
    else:
        config=configuration(assay) if config is None else config
        old_programs=program_pairs(stream,train,train,seed)
        new_programs=program_pairs(stream,retained,retained,seed)
        old_pairs=expand_pairs(old_programs,train,seed)
        new_pairs=expand_pairs(new_programs,retained,seed)
        affected=(torch.isin(old_pairs['first'],forgotten)|torch.isin(old_pairs['second'],forgotten))
        context_affected=torch.isin(old_programs['history'][old_programs['state']],forgotten).any(1)
        affected[:len(context_affected)] |= context_affected
        revoked_pairs={k:old_pairs[k][affected] for k in ['first','second']}
        for head,cfg in config.items():
            x=maps[cfg['features']]
            heads[head]={'x':x,'before':ExtendedObjective(x,stream['eps'],train,old_pairs,cfg),
                'after':ExtendedObjective(x,stream['eps'],retained,new_pairs,cfg),
                'revoked':ExtendedObjective(x,stream['eps'],forgotten,revoked_pairs,cfg),'cfg':cfg}
    return {'data':data,'truth':truth,'stream':stream,'req':req,'heads':heads,'evaluation':evaluation,
        'assay':assay,'seed':seed,'learner':learner}


def predictions(ep,parameters):
    if ep['learner']=='common':
        score=ep['data']['x']@parameters['shared']
        return score,score
    heads=ep['heads']
    score=heads['classification']['x']@parameters['classification']-heads['classification']['cfg'].get('threshold',0.)
    utility=heads['preference']['x']@parameters['preference']
    return score,utility


def evaluate(ep,parameters,references=None):
    score,utility=predictions(ep,parameters)
    metrics,_=evaluate_scores(score,ep['truth'],ep['req'],ep['data']['partitions']['test'],ep['evaluation'],utility)
    details={}
    if references is not None:
        reference_score,reference_utility=predictions(ep,references)
        for head,item in ep['heads'].items():
            theta,ref=parameters[head],references[head]
            residual=item['after'].gradient(theta).norm()
            details[head]={'parameter_distance':(theta-ref).norm().item(),'residual_certificate':(residual/item['after'].mu).item(),
                'gradient':residual.item(),'objective_gap':(item['after'].loss(theta)-item['after'].loss(ref)).item()}
            for name,ids in [('F',torch.tensor(ep['req']['forgotten'],device=DEVICE)),('R',torch.tensor(ep['req']['retained'],device=DEVICE)),('H',ep['data']['partitions']['test'])]:
                delta=item['x'][ids]@(theta-ref)
                details[head][name+'_prediction_rms']=delta.square().mean().sqrt().item()
        details['sign_disagreement']=100*((score>=0)!=(reference_score>=0)).double().mean().item()
        a,b=ep['evaluation']['first'],ep['evaluation']['second']
        details['preference_disagreement']=100*((utility[a]>utility[b])!=(reference_utility[a]>reference_utility[b])).double().mean().item()
    return metrics,details


def save_result(experiment,ep,variant,parameters,references,diagnostics,extra=None):
    metrics,details=evaluate(ep,parameters,references)
    folder=OUTPUT/experiment/ep['assay']/f"seed_{ep['seed']}"/variant
    record={'experiment':experiment,'assay':ep['assay'],'seed':ep['seed'],'variant':variant,
        'learner':ep['learner'],'metrics':metrics,'equivalence':details,'diagnostics':diagnostics}
    if extra: record.update(extra)
    write_json(folder/'result.json',record)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.cpu() for k,v in parameters.items()},folder/'parameters.pt')
    print(json.dumps({'experiment':experiment,'assay':ep['assay'],'seed':ep['seed'],'variant':variant,'metrics':metrics}),flush=True)
    return record
