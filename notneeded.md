## 5. Approach 3: PrimeShield v2

**Notebook:** `t1.ipynb`

### 5.1 Method Overview

PrimeShield v2 was designed to address an earlier failure mode in which cosine similarity remained too high. The revised method directly repels the anonymized embedding from the original embedding and uses a stronger optimization procedure.

The core ideas are:

- use a structured perturbation initialization,
- focus perturbations using a saliency face mask,
- add a prime-residue embedding target,
- optimize with MI-FGSM momentum,
- adapt the privacy and utility norms.

### 5.2 Architecture

<p align="center">
  <img src="assets/primeshield_architecture.svg" width="92%" alt="PrimeShield v2 architecture">
</p>

**Figure 14.** PrimeShield v2 architecture. The method combines structured initialization, saliency masking, prime-residue targets, adaptive norms, and MI-FGSM optimization.

### 5.3 Mathematical Formulation

PrimeShield v2 computes an auxiliary prime-residue target from the original embedding:

$$r_i = \lfloor |z_i|\, S \rfloor \bmod q_{i \bmod K}$$

$$z_i^{\star} = 2\,\frac{r_i}{q_i} - 1$$

The anonymized image is:

$$A = \mathrm{clip}(I + P)$$

The method uses a privacy margin in embedding space:

$$\mathcal{L}_{\mathrm{margin}} = \max\!\left(0,\; \alpha - \|z_A - z_I\|_{p_1}\right)$$

It also uses direct cosine repulsion:

$$\mathcal{L}_{\mathrm{cos}} = \left(1 + \cos(z_I, z_A)\right)^2$$

The utility loss keeps the image close to the original:

$$\mathcal{L}_{\mathrm{util}} = \|A - I\|_{p_2}$$

The total objective combines privacy, utility, smoothness, and prime-structure regularization:

$$\mathcal{L} = \lambda_1\,\mathcal{L}_{\mathrm{util}} + \lambda_2\,\mathcal{L}_{\mathrm{margin}} + \lambda_3\,\mathcal{L}_{\mathrm{cos}} + \lambda_4\,\mathcal{L}_{\mathrm{totient}} + \lambda_5\,\mathrm{TV}(P)$$

The optimizer uses momentum iterative FGSM (MI-FGSM):

$$g_{t+1} = \mu\, g_t + \frac{\nabla_P \mathcal{L}}{\|\nabla_P \mathcal{L}\|_1}$$

$$P_{t+1} = P_t + \eta \cdot \mathrm{sign}(g_{t+1})$$

### 5.4 Implementation Details

The notebook evaluates PrimeShield v2 on CelebA-HQ, LFW, and VGGFace2-style generalization. It also includes ablations to show which components improve privacy.

| Component | Purpose |
|---|---|
| Saliency face mask | Protect visually important facial regions |
| Golden spiral initialization | Provide structured perturbation start |
| Prime-residue target | Add a deterministic identity-displacement signal |
| Cosine repulsion | Directly push $z_A$ away from $z_I$ |
| MI-FGSM momentum | Stabilize and strengthen optimization |
| Adaptive $p_1, p_2$ | Balance embedding privacy and image utility |

### 5.5 Visualizations

<p align="center">
  <img src="assets/primeshield_cell23_out00.png" width="88%" alt="PrimeShield components">
</p>

**Figure 15.** PrimeShield v2 components: input image, saliency face mask, spiral initialization, and quick anonymized output.

<p align="center">
  <img src="assets/primeshield_cell29_out00.png" width="82%" alt="PrimeShield qualitative results">
</p>

**Figure 16.** Qualitative results showing original images, masks, anonymized images, and amplified perturbations.

<p align="center">
  <img src="assets/primeshield_cell31_out00.png" width="86%" alt="PrimeShield convergence">
</p>

**Figure 17.** Convergence curves. Cosine similarity drops while SSIM remains high.

<p align="center">
  <img src="assets/primeshield_cell37_out00.png" width="86%" alt="PrimeShield ablation">
</p>

**Figure 18.** Ablation results showing the effect of optimization and masking components.

### 5.6 Results

CelebA-HQ white-box results:

| Metric | PrimeShield v2 |
|---|---:|
| Images | 200 |
| Mean SSIM ↑ | 0.9421 |
| Median SSIM ↑ | 0.9419 |
| Mean cosine similarity ↓ | −0.1820 |
| Privacy rate, cosine $< 0.25$ | 95.0 % |
| Mean L2 distance | 1.5288 |
| Learned $p_1$ for privacy | $1.807 \pm 0.067$ |
| Learned $p_2$ for utility | $1.989 \pm 0.000$ |

LFW verification at FPR $= 0.001$:

| Method | FaceNet/CASIA TPR ↓ | ArcFace/VGGFace2 TPR ↓ | SSIM ↑ |
|---|---:|---:|---:|
| PrIdentity paper | 0.0190 | 0.0020 | 0.8422 |
| PrimeShield v1 | 0.0680 | 0.0090 | 0.9342 |
| **PrimeShield v2** | **0.0600** | **0.0040** | **0.9542** |

VGGFace2 generalization:

| Metric | Result |
|---|---:|
| White-box cosine ↓ | −0.4630 |
| White-box privacy | 99 % |
| Black-box cosine ↓ | 0.6966 |
| Black-box privacy | 0 % |
| SSIM ↑ | 0.9459 |

### 5.7 Interpretation

PrimeShield v2 gives the highest visual quality among the recorded approaches and remains close to PrIdentity on white-box ArcFace TPR. Its main limitation is black-box generalization: the VGGFace2 experiment shows that the anonymized image can still remain similar under a held-out model. The method is effective for white-box privacy and utility, but it needs a stronger ensemble strategy for transfer.

---
