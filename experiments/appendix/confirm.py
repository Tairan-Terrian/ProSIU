"""Independent requests for the existing matched-learner, closure, and trajectory studies."""
import argparse
import torch
from followup import core, run


def main(seeds):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    destination = core.RESEARCH / 'studies/appendix_20261020/confirmation'
    core.OUTPUT = run.OUTPUT = destination
    for assay in core.ASSAYS:
        for seed in seeds:
            for experiment in ['E01', 'E02', 'E04', 'E06']:
                marker = destination / experiment / assay / f'seed_{seed}' / 'complete.json'
                if marker.exists():
                    continue
                getattr(run, experiment.lower())([assay], [seed])
                core.write_json(marker, {'assay': assay, 'seed': seed, 'experiment': experiment})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    main(parser.parse_args().seeds)
