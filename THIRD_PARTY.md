# Sources and third-party terms

The root MIT license covers the ProSIU-authored implementation. Model weights and dataset archives are downloaded from their original providers and retain the providers' terms.

| Resource | Role | Upstream terms and attribution |
|---|---|---|
| [ProteinGym v1.3](https://github.com/OATML-Markslab/ProteinGym) | Published substitution-assay tables | Upstream repository: MIT. See `licenses/ProteinGym-MIT.txt` and cite ProteinGym, NeurIPS 2023. |
| [Orthologous GFP Fitness Peaks](https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks) | Original wild-type measurements | Upstream repository: GPL-3.0. See `licenses/GFP-GPL-3.0.txt`. The downloader obtains the original CSVs directly. |
| [ESM-2](https://github.com/facebookresearch/esm) | Frozen 35M and 650M protein encoders | MIT. See `licenses/ESM-MIT.txt`. The implementation uses the public Transformers interface. |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Generator of the frozen program library | Qwen Research License, including its non-commercial scope and documentation attribution requirement. See `licenses/Qwen-Research.txt` and `NOTICE`. **Built with Qwen.** |
| [Transformers](https://github.com/huggingface/transformers) | Loading public encoders and optional program generation | Installed dependency, Apache-2.0. |
| [PyTorch](https://github.com/pytorch/pytorch) | GPU computation and optimization | Installed dependency, BSD-style license. |
| [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) | Optional constrained program generation | Installed optional dependency, MIT. |

The release contains ProSIU's numerical implementation and fixed generated programs. Dataset preprocessing is implemented for the three published GFP assays. The feature builder calls the frozen public ESM-2 encoders directly. The repository's source distribution includes upstream notices and keeps downloaded data and weights in ignored directories.

## Method and data references

- Somermeyer et al. **Heterogeneity of the GFP fitness landscape and data-driven protein design.** eLife, 2022. https://elifesciences.org/articles/75842
- Sarkisyan et al. **Local fitness landscape of the green fluorescent protein.** Nature, 2016. https://www.nature.com/articles/nature17995
- Notin et al. **ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design.** NeurIPS Datasets and Benchmarks, 2023. https://github.com/OATML-Markslab/ProteinGym
- Lin et al. **Evolutionary-scale prediction of atomic-level protein structure with a language model.** Science, 2023. https://github.com/facebookresearch/esm
- Rafailov et al. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model.** NeurIPS, 2023. https://arxiv.org/abs/2305.18290
- Qwen Team. **Qwen2.5.** https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

The preference implementation uses pairwise logistic objectives and a calibrated numerical selector. The current benchmark trains two task heads while retaining its frozen language-program bank. This release describes that implemented computation directly.
