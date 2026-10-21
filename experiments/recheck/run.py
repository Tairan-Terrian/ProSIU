"""Repeat the fixed ablation and related-combination protocols on CUDA."""
import argparse
import torch
from followup import core
from followup import run as followup
from revision import ablation

OUTPUT=core.RESEARCH/'studies/followup_recheck_20261019'


def main(seeds):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    ablation.OUTPUT=OUTPUT/'ablation'
    core.OUTPUT=OUTPUT
    followup.OUTPUT=OUTPUT
    ablation.run(core.ASSAYS,seeds)
    followup.e03(core.ASSAYS,seeds)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    main(p.parse_args().seeds)
