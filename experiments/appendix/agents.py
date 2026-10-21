"""Fixed-library, prompt and evidence-budget analyses of protein program selection."""
import argparse
import itertools
import json

import torch

from .common import *
from prosiu.common import ASSETS


def bank_tensor(programs):
    return (torch.tensor([p['weights'] for p in programs], device=DEVICE),
            torch.tensor([p['rank'] for p in programs], device=DEVICE))


def propose(stream, pools, histories, available, programs):
    weights, ranks = bank_tensor(programs)
    similarity = torch.exp(-torch.cdist(stream['x'][pools], stream['x'][histories]).square() / 8)
    density = similarity.sum(-1)
    local = (similarity @ stream['eps'][histories, None]).squeeze(-1) / (1 + density)
    effects = (stream['y'][:, 1:3] - stream['y'][:, :1]) / stream['scale']
    operands = torch.stack([stream['offset'][pools], local, 1 / (1 + density),
        -(effects[:, 0] - effects[:, 1]).abs()[pools],
        ((stream['positions'][:, 0] - stream['positions'][:, 1]).abs().double() /
         (stream['positions'].max() + 1))[pools]], -1)
    priority = (operands @ weights.T).masked_fill(~available[:, :, None], -torch.inf)
    ordered = priority.argsort(1, descending=True)
    return pools.gather(1, ordered.gather(1, ranks[None, None].expand(len(pools), 1, -1)).squeeze(1))


def rollout(ep, params, programs, history_size=16, pool_size=64, budget=16, episodes=64, direct=False):
    generator = rng(853700 + ep['seed'])
    test, retained = ep['data']['partitions']['test'], tensors(ep['req']['retained'])
    pools = torch.stack([test[torch.randperm(len(test), generator=generator, device=DEVICE)[:pool_size]] for _ in range(episodes)])
    histories = torch.stack([retained[torch.randperm(len(retained), generator=generator, device=DEVICE)[:history_size]] for _ in range(episodes)])
    stream = dict(ep['stream'], eps=ep['stream']['eps'].clone())
    available = torch.ones_like(pools, dtype=torch.bool)
    _, utility = core.predictions(ep, params)
    measured, queries, diversity, times = [], [], [], []
    preference_correct, preference_count, selection_entropy = [], [], []
    for step in range(budget):
        begin = core.timed()
        proposed = propose(stream, pools, histories, available, programs)
        if direct:
            chosen = pools.gather(1, utility[pools].masked_fill(~available, -torch.inf).argmax(1)[:, None]).squeeze(1)
        else:
            chosen = proposed.gather(1, utility[proposed].argmax(1)[:, None]).squeeze(1)
        times.append(core.timed() - begin)
        left, right = torch.triu_indices(len(programs), len(programs), offset=1, device=DEVICE)
        truth_difference = ep['truth'][proposed[:, left]] - ep['truth'][proposed[:, right]]
        score_difference = utility[proposed[:, left]] - utility[proposed[:, right]]
        active = truth_difference != 0
        concordance = (1 + score_difference.sign() * truth_difference.sign()) / 2
        preference_correct.append(concordance[active].sum())
        preference_count.append(active.sum())
        selected_program = utility[proposed].argmax(1)
        frequencies = torch.bincount(selected_program, minlength=len(programs)).double() / episodes
        selection_entropy.append(-torch.special.xlogy(frequencies, frequencies).sum())
        diversity.append(torch.tensor([len(x.unique()) for x in proposed], device=DEVICE))
        queries.append(chosen)
        measured.append(ep['truth'][chosen])
        stream['eps'][chosen] = ep['truth'][chosen]
        histories = torch.cat([histories, chosen[:, None]], 1)
        available &= pools != chosen[:, None]
    measured = torch.stack(measured, 1)
    best = measured.cummax(1).values
    pool_truth = ep['truth'][pools]
    regret = pool_truth.max(1).values[:, None] - best
    threshold = pool_truth.topk(math.ceil(.1 * pool_size), dim=1).values[:, -1]
    percentile = (pool_truth[:, :, None] <= best[:, None]).double().mean(1) * 100
    cumulative_correct = torch.stack(preference_correct).cumsum(0)
    cumulative_count = torch.stack(preference_count).cumsum(0)
    return {'history_size': history_size, 'pool_size': pool_size, 'programs': len(programs),
            'unique_programs': len({(tuple(p['weights']), p['rank']) for p in programs}),
            'steps': list(range(1, budget + 1)), 'best': best.mean(0).cpu().tolist(),
            'regret': regret.mean(0).cpu().tolist(), 'percentile': percentile.mean(0).cpu().tolist(),
            'hit': (best >= threshold[:, None]).double().mean(0).mul(100).cpu().tolist(),
            'best_curve_area': best.mean().item(), 'regret_area': regret.mean().item(),
            'unique_proposals': torch.stack(diversity).double().mean().item(),
            'proposal_preference_accuracy': (100 * cumulative_correct / cumulative_count).cpu().tolist(),
            'preference_comparisons': cumulative_count.cpu().tolist(),
            'selection_entropy': torch.stack(selection_entropy).cpu().tolist(),
            'decision_seconds': sum(times) / len(times), 'query_ids': torch.stack(queries, 1).cpu().tolist(),
            'candidate_ids': pools.cpu().tolist(), 'episodes': episodes}


@torch.inference_mode()
def generate_libraries(model_labels=('Qwen2.5-3B-Instruct', 'Llama-3.2-3B-Instruct')):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    names = ['additive', 'interaction', 'novelty', 'balance', 'separation']
    program_ids = [f'P{i:02}' for i in range(1, 17)]
    weights_schema = {'type': 'object', 'required': names,
        'properties': {name: {'type': 'number', 'enum': [-3., -2., -1., -.5, 0., .5, 1., 2., 3.]}
                       for name in names}}
    program_schema = {'type': 'object', 'required': ['name', 'weights', 'rank'],
        'properties': {'name': {'type': 'string', 'maxLength': 48}, 'weights': weights_schema,
                       'rank': {'type': 'integer', 'enum': list(range(8))}}}
    schema = {'type': 'object', 'required': program_ids,
              'properties': {name: program_schema for name in program_ids}}
    base = (
        'Generate 16 distinct numerical candidate-selection programs for a measured protein double-mutation pool. '
        'Each program scores candidates using a weighted sum of five standardized operands in this exact order: '
        'additive single-mutant effect, local interaction estimate from observed combinations, novelty relative '
        'to the observed set, constituent-effect balance (larger means closer effects), and positional separation. '
        'The program returns the candidate with zero-based descending score rank 0 to 7. '
        'Only the observed history supplies measurements. Return a JSON object with entries P01 through P16. '
        'Each entry contains name, weights, and rank. The weights object contains exactly five named numbers: '
        'additive, interaction, novelty, balance, separation. ')
    prompts = [base + 'Cover complementary exploration and exploitation strategies.',
               base + 'Favor a diverse portfolio of interpretable evidence-based numerical selection rules.',
               base + 'Construct a compact search portfolio balancing observed interaction evidence and unexplored candidates.']
    for model_label in model_labels:
        conditions = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)] if model_label.startswith('Qwen') else [(0, 0)]
        pending = [(p, s) for p, s in conditions if not (OUTPUT / 'libraries' / model_label / f'P{p}_S{s}.json').exists()]
        if not pending:
            continue
        model_path = '/media/yjz/htr/llm_model/' + model_label
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(DEVICE).eval()
        for prompt_id, seed in pending:
            torch.manual_seed(853600 + seed)
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompts[prompt_id]}],
                add_generation_prompt=True, return_tensors='pt', return_dict=True).to(DEVICE)
            tokens = model.generate(**inputs, max_new_tokens=6000, do_sample=True, temperature=.7, top_p=.9,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema)))
            response = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            decoded = json.loads(response)
            programs = [dict(name=decoded[key]['name'],
                             weights=[decoded[key]['weights'][name] for name in names],
                             rank=decoded[key]['rank']) for key in program_ids]
            write_json(OUTPUT / 'libraries' / model_label / f'P{prompt_id}_S{seed}.json',
                       {'model': model_label, 'prompt_id': prompt_id, 'seed': seed, 'prompt': prompts[prompt_id],
                        'response': response, 'programs': programs, 'constrained_decoding': True})
            print('generated', model_label, prompt_id, seed, flush=True)
        del model
        torch.cuda.empty_cache()


def main(experiments, seeds):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    if 'E36' in experiments:
        generate_libraries()
    default = read_json(ASSETS / 'programs.json')
    second = read_json(MAIN / 'program_banks/bank_1/programs.json')
    third = read_json(MAIN / 'program_banks/bank_2/programs.json')
    for assay in core.ASSAYS:
        data, truth, maps, source = core.dataset(assay)
        for seed in seeds:
            ep = core.episode(data, truth, maps, source, assay, seed)
            params = saved_main(assay, seed)
            for experiment in experiments:
                path = OUTPUT / experiment / assay / f'seed_{seed}' / 'result.json'
                if path.exists():
                    continue
                rows = []
                if experiment == 'E35':
                    for n, kind in itertools.product([4, 8, 16, 32], ['Raw', 'Deduplicated', 'Independent library']):
                        programs = (second + third)[:n] if kind == 'Independent library' else (default + second)[:n]
                        if kind == 'Deduplicated':
                            unique = {}
                            for p in programs:
                                unique.setdefault((tuple(p['weights']), p['rank']), p)
                            programs = list(unique.values())
                        result = rollout(ep, params, programs)
                        rows.append(dict(result, requested_programs=n, library=kind))
                elif experiment == 'E36':
                    for library in sorted((OUTPUT / 'libraries').glob('*/*.json')):
                        spec = read_json(library)
                        result = rollout(ep, params, spec['programs'])
                        rows.append(dict(result, model=spec['model'], prompt_id=spec['prompt_id'],
                                         generation_seed=spec['seed'], executable=len(spec['programs']), generated=len(spec['programs'])))
                else:
                    for history, pool in itertools.product([4, 8, 16, 32], [32, 64, 128]):
                        for method in ['TARF', 'Entanglement', 'ProSIU']:
                            parameters = saved_main(assay, seed, method)
                            result = rollout(ep, parameters, default, history_size=history, pool_size=pool)
                            rows.append(dict(result, method=method))
                write_diagnostic(experiment, assay, seed, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=['E35', 'E36', 'E37'])
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    args = parser.parse_args()
    main(args.experiments, args.seeds)
