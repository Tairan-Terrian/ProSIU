"""Repeat the frozen supervision-reconstruction experiment in a separate folder."""
import torch
from followup import core,run

destination=core.RESEARCH/'studies/followup_revision_20261019'
core.OUTPUT=destination
run.OUTPUT=destination
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
run.e04(core.ASSAYS,list(range(5)))
