"""Measured optimization, geometry, preservation and numerical diagnostics."""
import argparse
import copy
import itertools

import torch

from .common import *
from followup.programs import proposals
from followup.run import sparse_request
from motivation.run import mutation_table
from prosiu.metrics import balanced_accuracy
from prosiu.features import build_selector_features


def trajectory(objective, start, x, reference, checkpoints=(0, 1, 2, 4, 8, 16, 32, 64)):
    theta = start.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta], lr=1., max_iter=1, history_size=30,
                                  tolerance_grad=1e-8, tolerance_change=1e-11, line_search_fn='strong_wolfe')
    calls, elapsed, diagnostic_elapsed, rows, parameters = 0, 0., 0., [], []
    def snapshot(budget, actual):
        nonlocal diagnostic_elapsed
        begin = core.timed()
        current = theta.detach().clone()
        residual = objective.gradient(current).norm()
        parameter_distance = (current - reference).norm().item()
        prediction_rms = (x @ (current - reference)).square().mean().sqrt().item()
        diagnostic_elapsed += core.timed() - begin
        rows.append({'iteration': budget, 'optimizer_iterations': actual, 'seconds': elapsed,
                     'evaluation_seconds': diagnostic_elapsed,
                     'function_evaluations': calls, 'parameter_distance': parameter_distance,
                     'prediction_rms': prediction_rms,
                     'certificate': (residual / objective.mu).item(), 'gradient': residual.item()})
        parameters.append(current)
        return residual.item()
    for iteration in range(max(checkpoints) + 1):
        if iteration in checkpoints:
            residual = snapshot(iteration, iteration)
            if residual <= 1e-8:
                for budget in checkpoints:
                    if budget > iteration:
                        snapshot(budget, iteration)
                break
        if iteration == max(checkpoints):
            break
        def closure():
            nonlocal calls
            optimizer.zero_grad()
            loss = objective.loss(theta)
            loss.backward()
            calls += 1
            return loss
        begin = core.timed()
        previous = theta.detach().clone()
        optimizer.step(closure)
        elapsed += core.timed() - begin
        if torch.equal(previous, theta.detach()):
            for budget in checkpoints:
                if budget > iteration:
                    snapshot(budget, iteration + 1)
            break
    return rows, parameters


def e14_e16(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed)
    original, reference = saved_main(assay, seed, 'Original'), saved_main(assay, seed, 'Retrain')
    decisions, certificates = [], []
    selected = ep['evaluation']['selected_program_candidates']
    for head, item in ep['heads'].items():
        obj, x = item['after'], item['x']
        starts, geometry = core.initializations(x, ep['req'], obj, original[head])
        gradient, hessian = obj.gradient(original[head]), obj.hessian(original[head])
        delta = starts['Newton'] - original[head]
        rho = .25 * (obj.weights * obj.a.norm(dim=1).pow(3)).sum()
        rho += obj.calibration * obj.x.norm(dim=1).pow(3).mean()
        for name in ['Original', 'Newton', 'ProSIU']:
            points, parameters = trajectory(obj, starts[name], x, reference[head])
            for point, theta in zip(points, parameters):
                point = dict(point, head=head, method=name)
                certificates.append(point)
                if head == 'preference':
                    features = x[selected]
                    ref_scores, scores = features @ reference[head], features @ theta
                    error = (theta - reference[head]).norm()
                    bound = features.norm(dim=2).amax(1) * error
                    score_error = (scores - ref_scores).abs().amax(1)
                    chosen = scores.argmax(1)
                    regret = ref_scores.max(1).values - ref_scores.gather(1, chosen[:, None]).squeeze(1)
                    margin = ref_scores.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
                    certified = margin > 2 * bound
                    agreement = scores.argmax(1) == ref_scores.argmax(1)
                    for temperature in [.25, 1., 4.]:
                        log_u = torch.log_softmax(scores / temperature, dim=1)
                        log_r = torch.log_softmax(ref_scores / temperature, dim=1)
                        kl = (log_u.exp() * (log_u - log_r)).sum(1).clamp_min(0.)
                        tv = (log_u.exp() - log_r.exp()).abs().sum(1) / 2
                        delta_score = bound / temperature
                        decisions.append({'method': name, 'iteration': point['iteration'], 'temperature': temperature,
                            'score_error': score_error.mean().item(), 'score_bound': bound.mean().item(),
                            'regret': regret.mean().item(), 'regret_bound': (2 * bound).mean().item(),
                            'KL': kl.mean().item(), 'KL_bound': (.5 * delta_score.square()).mean().item(),
                            'TV': tv.mean().item(), 'TV_bound': (delta_score / 2).clamp(max=1).mean().item(),
                            'certified_states': certified.sum().item(), 'state_count': len(selected),
                            'switches': (~agreement).sum().item(),
                            'certified_violations': (certified & ~agreement).sum().item(),
                            'score_bound_violations': (score_error > bound + 1e-10).sum().item(),
                            'KL_bound_violations': (kl > .5 * delta_score.square() + 1e-10).sum().item()})
            e = starts[name] - original[head] - delta
            linear = (gradient + hessian @ delta).norm() / obj.mu
            protection = (hessian @ e).norm() / obj.mu
            remainder = rho * (delta + e).square().sum() / (2 * obj.mu)
            certificates.append({'head': head, 'method': name, 'iteration': -1,
                'parameter_distance': (starts[name] - reference[head]).norm().item(),
                'linear_solve_term': linear.item(), 'protection_term': protection.item(),
                'curvature_term': remainder.item(), 'theorem_bound': (linear + protection + remainder).item(),
                'rho': rho.item(), 'mu': obj.mu})
    write_diagnostic('E14', assay, seed, decisions)
    write_diagnostic('E16', assay, seed, certificates)


def e15(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed, req=sparse_request(source, seed))
    original, _, _ = foundations(ep)
    rows = []
    for head, item in ep['heads'].items():
        x, obj = item['x'], item['after']
        starts, _ = core.initializations(x, ep['req'], obj, original[head])
        v = x[ep['req']['forgotten']]
        ids = sorted(set(ep['req']['retained']) & set(ep['req']['adjacent']))
        j = x[ids]
        metric = .01 * torch.eye(x.shape[1], device=DEVICE) + j.T @ j / len(j)
        dn, dc = starts['Newton'] - original[head], starts['Protected'] - original[head]
        b = v @ dn
        pinv = torch.linalg.pinv(v, rtol=1e-9)
        for name, d in [('Newton', dn), ('Protected', dc)]:
            rows.append({'kind': 'protein', 'head': head, 'method': name,
                         'constraint_residual': (v @ d - b).norm().item(),
                         'collateral_cost': (d @ metric @ d).item(),
                         'retained_rms': (j @ d).square().mean().sqrt().item(), 'target_rows': len(v)})
        for index in range(8):
            perturbation = torch.randn(len(dc), generator=rng(851500 + seed * 10 + index), device=DEVICE)
            perturbation = perturbation - pinv @ (v @ perturbation)
            d = dc + perturbation / perturbation.norm() * dc.norm() / 2
            rows.append({'kind': 'protein', 'head': head, 'method': f'Feasible perturbation {index + 1}',
                         'constraint_residual': (v @ d - b).norm().item(), 'collateral_cost': (d @ metric @ d).item(),
                         'retained_rms': (j @ d).square().mean().sqrt().item(), 'target_rows': len(v)})
    for rank in [0, 8, 16, 24, 32]:
        for target_dimension in [4, 8, 16]:
            generator = rng(851600 + seed + rank + target_dimension)
            q, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator, device=DEVICE))
            j = q[:, :rank].T
            p = torch.eye(32, device=DEVICE) - j.T @ j
            v = torch.randn(target_dimension, 32, generator=generator, device=DEVICE)
            vp = v @ p
            u, singular, vh = torch.linalg.svd(vp, full_matrices=False)
            active = singular > 1e-8
            inverse = (vh[active].T / singular[active]) @ u[:, active].T
            b = torch.randn(target_dimension, generator=generator, device=DEVICE)
            d = p @ inverse @ b
            residual = (v @ d - b).norm()
            rows.append({'kind': 'controlled linear', 'protected_rank': rank, 'target_dimension': target_dimension,
                         'attainable_rank': active.sum().item(), 'constraint_residual': residual.item(),
                         'protected_residual': (j @ d).norm().item(), 'feasible': residual.item() < 1e-7,
                         'smallest_active_singular': singular[active].min().item() if active.any() else 0.})
    write_diagnostic('E15', assay, seed, rows)


def efficiency_start(item, req, original, method):
    begin = core.timed()
    if method == 'Direct retained fit':
        return original, 0., 0.
    if method == 'Retraining':
        return torch.zeros_like(original), 0., 0.
    obj, x = item['after'], item['x']
    delta = torch.linalg.solve(obj.hessian(original), -obj.gradient(original))
    newton_seconds = core.timed() - begin
    if method == 'Newton':
        return original + delta, newton_seconds, 0.
    begin = core.timed()
    adjacent = sorted(set(req['retained']) & set(req['adjacent']))
    j, v = x[adjacent], x[req['forgotten']]
    metric = .01 * torch.eye(x.shape[1], device=DEVICE) + j.T @ j / len(j)
    inverse = torch.linalg.solve(metric, v.T)
    protected = inverse @ (torch.linalg.pinv(v @ inverse, rtol=1e-9, hermitian=True) @ (v @ delta))
    start = original + .75 * delta + .25 * protected
    return start, newton_seconds, core.timed() - begin


def e17(data, truth, maps, source, assay, seed):
    rows = []
    default = request_for(source, seed)
    train = tensors(sorted(default['retained'] + default['forgotten']))
    order = train[torch.randperm(len(train), generator=rng(851700 + seed), device=DEVICE)]
    axes = [('Training records', n, .2, None) for n in [512, 1024, 2048, 4096]]
    axes += [('Withdrawal fraction', 2048, f, None) for f in [.05, .2, .4]]
    axes += [('Feature dimension', 2048, .2, d) for d in [128, 256, 512]]
    for axis, size, fraction, dimension in axes:
        selected = order[:size]
        f = selected[:round(len(selected) * fraction)]
        req = make_request(source, seed, f.cpu().tolist(), [1.] * len(f), selected.cpu().tolist())
        use = maps
        if dimension is not None:
            use = {k: x @ (torch.randn(x.shape[1], dimension, device=DEVICE, generator=rng(851701)) / dimension ** .5)
                   for k, x in maps.items()}
        torch.cuda.reset_peak_memory_stats()
        begin = core.timed()
        ep = core.episode(data, truth, use, source, assay, seed, req=req)
        reconstruction = core.timed() - begin
        reconstruction_peak = torch.cuda.max_memory_allocated()
        original, reference, _ = foundations(ep)
        for head, item in ep['heads'].items():
            for name in ['Direct retained fit', 'Newton', 'ProSIU', 'Retraining']:
                torch.cuda.reset_peak_memory_stats()
                start, newton_seconds, protection_seconds = efficiency_start(item, req, original[head], name)
                points, params = trajectory(item['after'], start, item['x'], reference[head],
                                             checkpoints=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 600))
                overhead = newton_seconds + protection_seconds
                for threshold in [.01, .001, .0001]:
                    successful = [p for p in points if p['prediction_rms'] <= threshold]
                    point = successful[0] if successful else points[-1]
                    rows.append({'axis': axis, 'training_records': len(selected), 'withdrawal_fraction': fraction,
                        'dimension': item['x'].shape[1], 'requested_dimension': dimension, 'head': head, 'method': name,
                        'threshold': threshold, 'reached': bool(successful), 'error': point['prediction_rms'],
                        'seconds': point['seconds'] + point['evaluation_seconds'] + overhead + reconstruction,
                        'optimization_seconds': point['seconds'], 'evaluation_seconds': point['evaluation_seconds'],
                        'reconstruction_seconds': reconstruction, 'geometry_seconds': overhead,
                        'newton_seconds': newton_seconds, 'protection_seconds': protection_seconds,
                        'peak_memory_mib': max(reconstruction_peak, torch.cuda.max_memory_allocated()) / 1024 ** 2,
                        'function_evaluations': point['function_evaluations'],
                        'iterations': point['optimizer_iterations'], 'budget': point['iteration']})
    write_diagnostic('E17', assay, seed, rows)


def e22(data, truth, maps, source, assay, seed):
    req = sparse_request(source, seed)
    ep = core.episode(data, truth, maps, source, assay, seed, req=req)
    original, reference, _ = foundations(ep)
    comparison('E22', ep, 'Sparse request', original=original, reference=reference)
    retained, forgotten = tensors(req['retained']), tensors(req['forgotten'])
    mutation, _ = mutation_table(source)
    exact = torch.isin(mutation[retained], mutation[forgotten].unique()).any(1)
    positions = torch.isin(data['positions'][retained], data['positions'][forgotten].unique()).sum(1)
    distance = torch.cdist(data['x'][retained], data['x'][forgotten]).min(1).values
    quantiles = torch.quantile(distance, torch.tensor([1 / 3, 2 / 3], device=DEVICE))
    strata = {'Exact substitution': exact, 'Shared position': (positions > 0) & ~exact,
              'Remote': positions == 0}
    strata.update({f'Similarity {label}': mask for label, mask in [
        ('high', distance <= quantiles[0]), ('medium', (distance > quantiles[0]) & (distance <= quantiles[1])),
        ('low', distance > quantiles[1]) ]})
    rows = []
    for method in METHODS:
        saved = torch.load(output_file('E22', assay, seed, 'Sparse request', method).with_name('predictions.pt'),
                           map_location=DEVICE, weights_only=False)
        x = ep['heads']['classification']['x']
        reference_score, _ = core.predictions(ep, reference)
        for label, mask in strata.items():
            ids = retained[mask]
            correct = (saved['score'][ids] >= 0) == (truth[ids] > 0)
            rows.append({'method': method, 'stratum': label, 'count': len(ids),
                         'correct': correct.sum().item(), 'positive_count': (truth[ids] > 0).sum().item(),
                         'accuracy': correct.double().mean().item() * 100,
                         'prediction_rmse': (saved['score'][ids] - reference_score[ids]).square().mean().sqrt().item()})
    write_diagnostic('E22', assay, seed, rows)


def e24(data, truth, maps, source, assay, seed):
    rows = []
    for fraction in [.01, .05, .1, .2]:
        req = request_for(source, seed, fraction)
        stream = observed(data, truth, req)
        train, retained, forgotten = tensors(sorted(req['retained'] + req['forgotten'])), tensors(req['retained']), tensors(req['forgotten'])
        bank = program_pairs(stream, train, train, seed)
        pools, histories = bank['candidate_pools'], bank['history']
        original = proposals(stream, pools, histories)
        affected = torch.isin(pools, forgotten).any(1) | torch.isin(histories, forgotten).any(1)
        valid_history, available = torch.isin(histories, retained), torch.isin(pools, retained)
        repetitions = []
        for repeat in range(5):
            torch.cuda.reset_peak_memory_stats()
            begin = core.timed()
            full = proposals(stream, pools, histories, valid_history=valid_history, available=available)
            full_seconds = core.timed() - begin
            full_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
            begin = core.timed()
            local = original.clone()
            local[affected] = proposals(stream, pools[affected], histories[affected],
                valid_history=valid_history[affected], available=available[affected])
            local_seconds = core.timed() - begin
            repetitions.append({'fraction': fraction, 'repeat': repeat, 'states': len(pools),
                         'affected_states': affected.sum().item(), 'identical_outputs': (local == full).sum().item(),
                         'total_outputs': local.numel(), 'full_seconds': full_seconds, 'local_seconds': local_seconds,
                         'full_memory_mib': full_memory, 'local_memory_mib': torch.cuda.max_memory_allocated() / 1024 ** 2})
        pairs = torch.combinations(torch.arange(full.shape[1], device=DEVICE), r=2)
        rebuilt, fitted = {}, {}
        for label, selected in [('full', full), ('local', local)]:
            first, second = selected[:, pairs[:, 0]].flatten(), selected[:, pairs[:, 1]].flatten()
            distinct = first != second
            first, second = first[distinct], second[distinct]
            choose = torch.randperm(len(first), generator=rng(852400 + seed), device=DEVICE)[:2048]
            rebuilt[label] = {'first': first[choose], 'second': second[choose]}
            records = expand_pairs(rebuilt[label], retained, seed)
            fitted[label] = {}
            for head, cfg in configuration(assay).items():
                x = maps[cfg['features']]
                objective = core.ExtendedObjective(x, stream['eps'], retained, records, cfg)
                theta, _ = core.fit(objective)
                fitted[label][head] = x @ theta
        equal_records = all(torch.equal(rebuilt['full'][k], rebuilt['local'][k]) for k in ['first', 'second'])
        prediction_rms = max((fitted['full'][h] - fitted['local'][h]).square().mean().sqrt().item()
                             for h in fitted['full'])
        for row in repetitions:
            row.update(identical_supervision=equal_records, prediction_rmse=prediction_rms)
        rows.extend(repetitions)
        folder = OUTPUT / 'E24' / assay / f'seed_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({'full': full.cpu(), 'local': local.cpu(), 'affected': affected.cpu()}, folder / f'replay_{fraction}.pt')
    write_diagnostic('E24', assay, seed, rows, {'operator': 'fixed-bank counterfactual replay'})


def e27_e29(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed)
    mutations, anchors = mutation_table(source)
    generator = rng(852700 + seed)
    permutation = anchors[torch.randperm(len(anchors), generator=generator, device=DEVICE)]
    eligibility = {fraction: torch.isin(mutations, permutation[:round(len(anchors) * fraction)]).all(1)
                   for fraction in [.25, .5, .75, 1.]}
    common = eligibility[.25]
    coverage, strata = [], []
    test, dev = data['partitions']['test'], data['partitions']['dev']
    cuts = torch.quantile(truth[dev].abs(), torch.tensor([.25, .5, .75], device=DEVICE))
    for method in METHODS:
        params = saved_main(assay, seed, method)
        score, utility = core.predictions(ep, params)
        for fraction, valid in eligibility.items():
            common_ids = test[common[test]]
            coverage.append({'method': method, 'fraction': fraction, 'anchor_count': round(len(anchors) * fraction),
                             'eligible_test': valid[test].sum().item(), 'total_test': len(test),
                             'common_test': len(common_ids), 'coverage_percent': valid[test].double().mean().item() * 100,
                             'common_accuracy': ((score[common_ids] >= 0) == (truth[common_ids] > 0)).double().mean().item() * 100})
        thresholds = torch.linspace(-2., 2., 81, device=DEVICE)
        positive, negative = truth[dev] > 0, truth[dev] <= 0
        decisions = score[dev, None] >= thresholds
        dev_accuracy = .5 * (decisions[positive].double().mean(0) + (~decisions[negative]).double().mean(0))
        chosen_threshold = thresholds[dev_accuracy.argmax()]
        for strategy, threshold in [('Fixed', 0.), ('Development threshold', chosen_threshold)]:
            for band in range(4):
                bins = torch.bucketize(truth[test].abs(), cuts)
                ids = test[bins == band]
                predicted, labels = score[ids] >= threshold, truth[ids] > 0
                probability = torch.sigmoid(score[ids])
                strata.append({'method': method, 'threshold_policy': strategy, 'threshold': float(threshold),
                    'band': band + 1, 'count': len(ids), 'positive': labels.sum().item(),
                    'true_positive': (predicted & labels).sum().item(),
                    'negative': (~labels).sum().item(), 'true_negative': (~predicted & ~labels).sum().item(),
                    'accuracy': (predicted == labels).double().mean().item() * 100,
                    'brier': (probability - labels.double()).square().mean().item(),
                    'mean_absolute_interaction': truth[ids].abs().mean().item()})
    write_diagnostic('E27', assay, seed, coverage, {'interpretation': 'valid-anchor eligibility on a common frozen model'})
    write_diagnostic('E29', assay, seed, strata, {'development_absolute_effect_cutoffs': cuts.cpu().tolist()})
    # The second panel refits after restricting the active evidence library.
    # Frozen feature coordinates and measurement normalization stay fixed.
    restricted_source = dict(source, rows=[dict(r) for r in source['rows']])
    for index, row in enumerate(restricted_source['rows']):
        if row['partition'] == 'train' and not common[index]:
            row['partition'] = 'inactive'
    shared_request = request_for(restricted_source, seed)
    common_test = test[common[test]]
    active_data = dict(data, partitions=dict(data['partitions'], test=common_test))
    shared_episode = core.episode(active_data, truth, maps, source, assay, seed, req=shared_request)
    train = data['partitions']['train']
    for fraction, valid in eligibility.items():
        training = train[valid[train]].cpu().tolist()
        req = make_request(source, seed, shared_request['forgotten'], shared_request['shifts'], training)
        active_episode = core.episode(active_data, truth, maps, source, assay, seed, req=req)
        comparison('E27', active_episode, f'{fraction:.2f}',
                   evaluation_ep=common_evaluation(active_episode, shared_episode),
                   extra={'anchor_fraction': fraction, 'eligible_training': len(training),
                          'eligible_test': valid[test].sum().item(), 'common_test': len(common_test),
                          'feature_policy': 'fixed precomputed coordinates; active evidence restricted by anchor coverage'})


def e39(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed, req=sparse_request(source, seed))
    original, reference, _ = foundations(ep)
    rows = []
    for head, item in ep['heads'].items():
        x, obj = item['x'], item['after']
        adjacent = sorted(set(ep['req']['retained']) & set(ep['req']['adjacent']))
        retained = tensors(ep['req']['retained'])
        random_ids = retained[torch.randperm(len(retained), generator=rng(853900 + seed), device=DEVICE)[:len(adjacent)]]
        dn = torch.linalg.solve(obj.hessian(original[head]), -obj.gradient(original[head]))
        v = x[ep['req']['forgotten']]
        reporting_metric = .01 * torch.eye(x.shape[1], device=DEVICE) + x[adjacent].T @ x[adjacent] / len(adjacent)
        for name, ids in [('Identity', None), ('Adjacent', tensors(adjacent)), ('All retained', retained), ('Random retained', random_ids)]:
            metric = .01 * torch.eye(x.shape[1], device=DEVICE)
            if ids is not None:
                metric += x[ids].T @ x[ids] / len(ids)
            inverse = torch.linalg.solve(metric, v.T)
            dc = inverse @ (torch.linalg.pinv(v @ inverse, rtol=1e-9, hermitian=True) @ (v @ dn))
            for omega in [0., .25, .5, .75, 1.]:
                d = (1 - omega) * dn + omega * dc
                points, _ = trajectory(obj, original[head] + d, x, reference[head], checkpoints=(0, 1, 4, 16, 64))
                for point in points:
                    rows.append(dict(point, head=head, metric=name, omega=omega,
                                     initial_protected_cost=(d @ reporting_metric @ d).item(),
                                     constraint_residual=(v @ (d - dn)).norm().item()))
    write_diagnostic('E39', assay, seed, rows)


def e41(data, truth, maps, source, assay, seed):
    ep = core.episode(data, truth, maps, source, assay, seed)
    params = saved_main(assay, seed)
    rows = []
    y = torch.tensor([r['y'] for r in source['rows']], device=DEVICE)
    base = (y[:, 3] - y[:, 1] - y[:, 2] + y[:, 0]) / data['scale']
    for a, b in [(.1, 0.), (1., 10.), (10., -5.), (100., 100.)]:
        transformed = a * y + b
        contrast = (transformed[:, 3] - transformed[:, 1] - transformed[:, 2] + transformed[:, 0]) / (a * data['scale'])
        transformed_data = dict(data, y=a * data['y'] + b, scale=a * data['scale'],
                                anchor=a * data['anchor'] + b)
        transformed_maps = build_selector_features(assay, transformed_data, core.DATA)
        feature_error = max((transformed_maps[name] - x).abs().max().item() for name, x in maps.items())
        prediction_error = max(((transformed_maps[item['cfg']['features']] - item['x']) @ params[head]).abs().max().item()
                               for head, item in ep['heads'].items())
        rows.append({'kind': 'affine', 'scale': a, 'offset': b, 'contrast_max_error': (contrast - base).abs().max().item(),
                     'feature_max_error': feature_error, 'prediction_max_error': prediction_error,
                     'sign_agreement': ((contrast > 0) == (base > 0)).double().mean().item() * 100})
    for head, item in ep['heads'].items():
        obj = item['after']
        ref = params[head]
        theta = ref.float().clone().requires_grad_(True)
        a, p, w = obj.a.float(), obj.p.float(), obj.weights.float()
        x, target = obj.x.float(), obj.calibration_target.float()
        optimizer = torch.optim.LBFGS([theta], max_iter=600, history_size=30, tolerance_grad=1e-6,
                                     tolerance_change=1e-8, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            logits = a @ theta
            loss = obj.mu * theta.square().sum() / 2
            loss += w @ (p * torch.nn.functional.softplus(-logits) + (1 - p) * torch.nn.functional.softplus(logits))
            loss += obj.calibration * (torch.sqrt(1 + (x @ theta - target).square()) - 1).mean()
            loss.backward()
            return loss
        optimizer.step(closure)
        converted = theta.detach().double()
        error = item['x'] @ (converted - ref)
        rows.append({'kind': 'precision', 'head': head, 'prediction_rmse': error.square().mean().sqrt().item(),
                     'parameter_distance': (converted - ref).norm().item(),
                     'gradient_fp64': obj.gradient(ref).norm().item(), 'gradient_fp32': obj.gradient(converted).norm().item(),
                     'sign_agreement': ((item['x'] @ converted >= 0) == (item['x'] @ ref >= 0)).double().mean().item() * 100})
    write_diagnostic('E41', assay, seed, rows)


FUNCTIONS = {'theory': e14_e16, 'E15': e15, 'E17': e17, 'E22': e22, 'E24': e24,
             'strata': e27_e29, 'E39': e39, 'E41': e41}


def main(experiments, seeds, assays):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    for assay in assays:
        data, truth, maps, source = core.dataset(assay)
        for seed in seeds:
            for experiment in experiments:
                marker = OUTPUT / 'completion' / experiment / assay / f'{seed}.json'
                if marker.exists():
                    continue
                FUNCTIONS[experiment](data, truth, maps, source, assay, seed)
                write_json(marker, {'experiment': experiment, 'assay': assay, 'seed': seed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=list(FUNCTIONS))
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--assays', nargs='+', default=core.ASSAYS)
    args = parser.parse_args()
    main(args.experiments, args.seeds, args.assays)
