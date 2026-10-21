"""Encoder and head-structure controls with fixed downstream protocols."""
import argparse
import copy
import shutil

import torch
from transformers import AutoTokenizer

from .common import *
from prosiu.features import build_context_features, build_language_features


def prepare_encoder():
    costs = []
    encoder650 = Path(__file__).resolve().parents[2] / '.cache/models/esm650'
    AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D').save_pretrained(encoder650)
    for name, model, model_source in [
            ('encoder35', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t12_35M_UR50D'),
            ('encoder650', 'facebook/esm2_t33_650M_UR50D', str(encoder650))]:
        folder = OUTPUT / name
        (folder / 'processed').mkdir(parents=True, exist_ok=True)
        (folder / 'features').mkdir(parents=True, exist_ok=True)
        for assay in core.ASSAYS:
            shutil.copyfile(core.DATA / 'processed' / f'{assay}.json', folder / 'processed' / f'{assay}.json')
        shutil.copyfile(core.DATA / 'features/amacGFP_language.pt', folder / 'features/amacGFP_language.pt')
        torch.set_default_dtype(torch.float32)
        torch.cuda.reset_peak_memory_stats()
        begin = core.timed()
        build_context_features(folder, core.ASSAYS, model_name=model_source)
        elapsed = core.timed() - begin
        torch.set_default_dtype(torch.float64)
        for assay in core.ASSAYS:
            path = folder / 'features' / f'{assay}.pt'
            features = torch.load(path, map_location='cpu', weights_only=False)
            features['encoder'] = model
            torch.save(features, path)
        metadata = {'model': model, 'projection': 48, 'landmarks': 256, 'frozen': True,
                    'feature_build_seconds': elapsed, 'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                    'assays': core.ASSAYS, 'feature_kind': 'Context', 'weights': 'cached pretrained weights'}
        write_json(folder / 'encoder.json', metadata)
        costs.append(metadata)
    folder = OUTPUT / 'language650_cost'
    (folder / 'processed').mkdir(parents=True, exist_ok=True)
    (folder / 'features').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(core.DATA / 'processed/amacGFP.json', folder / 'processed/amacGFP.json')
    torch.set_default_dtype(torch.float32)
    torch.cuda.reset_peak_memory_stats()
    begin = core.timed()
    build_language_features(folder, model_name=str(encoder650))
    elapsed = core.timed() - begin
    torch.set_default_dtype(torch.float64)
    costs.append({'model': 'facebook/esm2_t33_650M_UR50D', 'feature_kind': 'Sequence contrast',
                  'feature_build_seconds': elapsed, 'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
                  'assays': ['amacGFP'], 'weights': 'cached pretrained weights', 'frozen': True})
    write_json(OUTPUT / 'encoder_costs.json', costs)


def e32(assay, seeds):
    for label, folder in [('ESM-2 35M', core.DATA), ('ESM-2 650M', OUTPUT / 'encoder650')]:
        data, truth, maps, source = core.dataset(assay, folder)
        config = copy.deepcopy(configuration(assay))
        for head in config:
            config[head]['features'] = 'context'
        for seed in seeds:
            ep = core.episode(data, truth, maps, source, assay, seed, config=config)
            comparison('E32', ep, label, extra={'encoder': label, 'matched_features': 'context'})


def e33(data, truth, maps, source, assay, seed):
    defaults = configuration(assay)
    for variant in ['Shared head', 'Two context heads', 'Selected two heads',
                    'Shared head 512', 'Two heads 256']:
        cfg = copy.deepcopy(defaults)
        use = dict(maps)
        if variant in ['Shared head', 'Shared head 512']:
            joint = dict(cfg['classification'], features='context', classification=1., preference=1.,
                         calibration=cfg['preference'].get('calibration', 0.), rank_temperature=0.)
            cfg = {'classification': dict(joint), 'preference': dict(joint)}
        elif variant in ['Two context heads', 'Two heads 256']:
            for head in cfg:
                cfg[head]['features'] = 'context'
        if variant in ['Shared head 512', 'Two heads 256']:
            dimension = 512 if variant == 'Shared head 512' else 256
            use = {k: x @ (torch.randn(x.shape[1], dimension, device=DEVICE, generator=rng(853300)) / dimension ** .5)
                   for k, x in maps.items()}
        ep = core.episode(data, truth, use, source, assay, seed, config=cfg)
        extra = {'variant': variant, 'parameters': sum(v['x'].shape[1] for v in ep['heads'].values()) //
                 (2 if variant.startswith('Shared') else 1)}
        if variant.startswith('Shared'):
            if completed('E33', assay, seed, variant, ['ProSIU']):
                continue
            item = ep['heads']['classification']
            original, _ = core.fit(item['before'])
            reference, _ = core.fit(item['after'], strict=True, iterations=1000)
            starts, _ = core.initializations(item['x'], ep['req'], item['after'], original)
            theta, info = core.fit(item['after'], starts['ProSIU'])
            record('E33', ep, variant, 'ProSIU', {h: theta for h in cfg}, {h: reference for h in cfg},
                   {'shared_head': info}, extra)
        else:
            comparison('E33', ep, variant, methods=['ProSIU'], extra=extra)
        path = output_file('E33', assay, seed, variant, 'ProSIU')
        result = read_json(path)
        saved = torch.load(path.with_name('predictions.pt'), map_location=DEVICE, weights_only=False)
        geometry = {}
        for head, item in ep['heads'].items():
            obj, theta = item['after'], saved['parameters'][head]
            n = obj.n_measurements
            error = torch.sigmoid(obj.a @ theta) - obj.p
            sign_gradient = obj.a[:n].T @ error[:n] / n
            preference_gradient = obj.a[n:].T @ error[n:] / len(error[n:])
            cosine = torch.nn.functional.cosine_similarity(sign_gradient, preference_gradient, dim=0)
            geometry[head] = {'task_gradient_cosine': cosine.item(),
                              'sign_gradient_norm': sign_gradient.norm().item(),
                              'preference_gradient_norm': preference_gradient.norm().item()}
        result['head_geometry'] = geometry
        write_json(path, result)


def main(seeds):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    if not (OUTPUT / 'encoder_costs.json').exists():
        prepare_encoder()
    for assay in core.ASSAYS:
        e32(assay, seeds)
        data, truth, maps, source = core.dataset(assay)
        for seed in seeds:
            e33(data, truth, maps, source, assay, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    main(parser.parse_args().seeds)
