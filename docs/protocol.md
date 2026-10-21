# Benchmark and implementation

## Measured contrasts

For a background sequence and two substitutions, each target has four measured sequences:

$$q=(x_0,x_a,x_b,x_{ab}),\qquad e_q=y_{ab}-y_a-y_b+y_0.$$

The constituent branch is fixed at $a_q=y_a+y_b-y_0$. The task selector learns the normalized interaction $\epsilon_q=e_q/\sigma$, where $\sigma$ is the sample standard deviation of the valid WT and single-mutation measurements. amacGFP and cgreGFP brightness is converted to base-10 logarithms. avGFP uses its published log-brightness values. The original measured WT value is used for each protein.

The three ProteinGym v1.3 assay identifiers are:

| Protein | Assay |
|---|---|
| amacGFP | `Q8WTC7_9CNID_Somermeyer_2022` |
| cgreGFP | `D7PM05_CLYGR_Somermeyer_2022` |
| avGFP | `GFP_AEQVI_Sarkisyan_2016` |

Targets are measured double mutants whose two constituent singles are also measured. The preprocessing sorts mutation tuples, applies Python random seed 650017, and uses the fixed 40%/20%/20%/20% identity split with caps 4096/2048/1024/2048. The resulting counts are:

| Protein | Formal training | Development training | Development evaluation | Formal test | Valid anchors |
|---|---:|---:|---:|---:|---:|
| amacGFP | 4096 | 2048 | 1024 | 2048 | 1202 |
| cgreGFP | 3806 | 1904 | 1024 | 1904 | 1170 |
| avGFP | 4096 | 2048 | 1024 | 2048 | 1085 |

Target identities are disjoint between these partitions. Permanently valid single-mutation measurements are available as anchors. All three proteins belong to the GFP family.

## Controlled evidence correction

Training combinations are grouped by unordered position pair. Request seed `650100 + seed` shuffles these groups and selects whole groups until they cover at least 20% of the formal training targets. Each selected group receives a random offset of either $+\sigma$ or $-\sigma$. These are simulated measurement offsets introduced by this benchmark.

The original selector is fitted on the offset-contaminated training stream. A withdrawal request removes the selected records and their derived preference supervision. The retained stream contains the other training records. The original clean withdrawn measurements are used by evaluation to compute CA. The training and unlearning functions receive observed training values and retained values. Formal test measurements enter the metric calculation after fitting.

The final configurations were selected using development-training requests 9100 and 9101 and the development-evaluation partition. The release bundles the selected configurations. Formal runs use seeds 0–4.

## Frozen representations

The feature construction keeps the published single-mutation branch fixed and uses these maps:

1. **Context:** ESM-2 35M WT residue states and substitution-embedding differences are projected into 48 dimensions using GPU seed 650027. Symmetric pair descriptors and measured single-effect descriptors produce a 256-landmark Nyström map. Its moments and landmarks use formal-training inputs independently of the combination labels.
2. **Anchor and identity:** radial features of the sorted transformed constituent effects use widths 0.2, 0.7, and 2.0, with 192 fixed development-training landmarks and seed 651007. Explicit mutation and position indicators supplement these features.
3. **Language:** amacGFP's preference head additionally uses frozen ESM-2 650M mean sequence embeddings and their four-sequence interaction contrast. Each map uses 128 principal directions fitted to development-training representations, followed by the released scaling. Model fitting uses combination labels only through its training objective.

The final feature choices are `context` for all three classification heads, `language` for amacGFP's preference head, `context` for cgreGFP's preference head, and `identity` for avGFP's preference head. The inputs, dimensions, projection seeds, batching, and normalization are fixed by the release.

## Epistatic Program Selection

The agent uses the 16 entries in `assets/programs.json`. Each program ranks a 64-candidate pool using five operands: measured additive effect, local interaction estimate, novelty, constituent balance, and sequence separation. It selects the candidate at its specified rank.

For each request, 128 observation states each contain 16 measured context records. Program comparisons are sampled with seed `650300 + request_seed`. The program-pair bank contains at most 2048 comparisons. Training also adds 8192 sampled record pairs, with identical-endpoint pairs removed, using seed `651100 + request_seed`.

After withdrawal, both the program states and the additional comparison bank are rebuilt on retained identities. Formal PA candidates use the original fixed context representation and program-selection rule with seed `10000 + request_seed`. Changing the learned selector changes candidate scores while preserving this comparison definition.

The language model contributes the frozen program library. The learned state updated in this benchmark consists of the numerical selector heads. Pretrained encoders and the library stay fixed throughout each withdrawal episode.

## Mutation Preference Optimization

Let $x_i$ be the fixed representation, $z_i=\mathbf{1}[\epsilon_i>0]$, and $v_{ij}=x_i-x_j$. Each head minimizes a strongly convex objective of the form

$$
L(\theta)=\frac{\lambda}{2}\|\theta\|^2
+c\sum_{i\in S}w_i\,\ell(z_i,x_i^\top\theta)
+\frac{\alpha}{|P|}\sum_{(i,j)\in P}\ell(p_{ij},v_{ij}^\top\theta)
+\frac{\beta}{|S|}\sum_{i\in S}\left(\sqrt{1+(x_i^\top\theta-t_i)^2}-1\right),
$$

$$\ell(p,u)=p\log(1+e^{-u})+(1-p)\log(1+e^u),\qquad
w_i=\frac{1}{2\,|\{k\in S:z_k=z_i\}|}.$$

For a positive ranking temperature $T$, $p_{ij}=\operatorname{sigmoid}((\epsilon_i-\epsilon_j)/T)$ and $t_i=\epsilon_i/T$. For the hard-preference setting, $p_{ij}=\mathbf{1}[\epsilon_i>\epsilon_j]$ and $t_i=\epsilon_i$. Exact measurement ties are excluded from the preference loss.

The classification and preference heads have separate configurations and parameters. The classification head may also include preference supervision. Classification thresholds are fixed by development selection. PA uses the preference-head score difference.

Original and retained-reference fits use FP64 L-BFGS with history size 30, strong-Wolfe line search, a 600-iteration cap, gradient tolerance 0.00000001, and objective-change tolerance 0.00000000001. The original model uses the observed training stream. The reference fit starts from zero on the retained stream and serves as a convergence diagnostic.

## Selective Epistasis Forgetting

For the retained objective at the original parameters, compute

$$g_R=\nabla L_R(\theta_D),\qquad H_R=\nabla^2 L_R(\theta_D),\qquad d_N=-H_R^{-1}g_R.$$

Let $V$ contain the withdrawn feature rows. The protection set contains retained combinations sharing a position with a withdrawn group. With its feature matrix $X_P$,

$$M=\frac{X_P^\top X_P}{|P|}+0.01I,\qquad
d_C=M^{-1}V^\top(VM^{-1}V^\top)^\dagger Vd_N.$$

The initialization is $\theta_D+0.75d_N+0.25d_C$. The retained objective is then solved using the same FP64 L-BFGS settings. The pseudoinverse uses relative tolerance 0.000000001. Both heads undergo this update.

When the retained neighborhood is empty, its empirical second-moment contribution is zero and the protection metric is $M=0.01I$.

The checkpoint stores original, unlearned, and retained-reference parameters. Diagnostics report the protection constraint residual, retained gradient, parameter distance, objective gap, and maximum prediction difference to the retained-reference fit.

## Metrics and reference comparison

CA, RA, and TA use the same balanced-accuracy definition:

$$100\times\frac{\mathrm{TPR}+\mathrm{TNR}}2.$$

The clean interaction target is positive when $\epsilon>0$. The classification score predicts positive when its threshold-adjusted value is at least zero.

PA compares the preference-head ordering of the two selected candidates with the ordering of their clean measured interactions. A predicted tie receives half credit. A measured tie is excluded. All metrics retain their natural percentage scale.

The summary uses every preset seed and the sample standard deviation. `evaluate` reconstructs metrics from saved predictions. `verify` compares the complete 15-run reproduction with the reference values and records every difference. The release uses a 0.1-percentage-point per-run comparison tolerance.
