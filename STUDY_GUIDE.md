# Lab 2 — KDE, Bayes Classifier & PCA: Solution Notes and Study Guide

Everything here was implemented in `Unsolved/`; the original unsolved files are still inside `lab_2_new.zip`.
All four notebooks run top-to-bottom with **zero errors** and all tests pass.

---

## 0. How to run it

```bash
cd ~/Lab2
source .venv/bin/activate          # venv created for this lab (packages in requirements.txt)
cd Unsolved
jupyter notebook                   # or open the .ipynb files in VS Code and pick the .venv kernel
```

> ⚠️ **scikit-learn is pinned to 1.7.2.** With the newest version (1.9) the *given* QDA test
> (`BayesClassifier.ipynb`, section 4, "1.3") fails. Newer sklearn computes QDA covariances with the
> biased ÷n estimator; our class (and the lab's tests) use the unbiased ÷(n−1) one. The ratio
> was exactly n/(n−1). This isn't a bug in our code, so I pinned the version instead of changing the lab's code.
> Worth knowing in case the TA asks.

Suggested order (same as the README): KDE → Bayes → PCA → Microbiologist.

---

## 1. What I implemented, TODO by TODO

### 1.1 `KDEstimator.py` (no for-loops allowed)

| TODO | Code | Why |
|---|---|---|
| 0 | `self.bump = bump; self.bandwidth = bandwidth` | store hyperparameters (was already there) |
| 1 | `bandwidth in ['Silverman','Scott']` (if str) **or** `isinstance(bandwidth,(int,float))` | input validation; the `isinstance(str)` guard avoids comparing numbers to strings |
| 2 | `self.M, self.N = x_train.shape; self.x_train = x_train` | M = #points, N = #features. KDE is a *lazy learner*: fitting = storing the data |
| 3 | `avg_σ = np.mean(np.std(x_train, axis=0))` | σ̂ = (σ₁+…+σ_N)/N, one std per **column** (`axis=0`) |
| 4 | Silverman `avg_σ * (4/(M*(N+2)))**(1/(N+4))`, Scott `avg_σ * M**(-1/(N+4))` | formulas from the notebook |
| 5 | `scale = sqrt((2π)^N)`, numerator `exp(-0.5 * einsum('ij,ij->i', x, x))` | `einsum('ij,ij->i')` = row-wise dot product xᵀx for all rows at once → shape (m,) |
| 6 | `np.all(np.abs(x) <= 0.5, axis=-1).astype(float)` | 1 only if **every** coordinate is in [−½, ½] |
| 7 | `(1 / h**N) * self.g(x / h)` | ϕ(x) = g(x/h)/hᴺ. The 1/hᴺ keeps the integral = 1 after stretching by h |
| 8 | `scale * np.sum(self.ϕ(x - xₘ))` | **broadcasting**: x (N,) − xₘ (M,N) → (M,N); one bump per training point; average |
| 9 | `np.apply_along_axis(self.P, -1, x_val)` | `axis=-1` = the last axis: each row for a 2D input, the whole vector for a 1D input. That's how "the same line" handles both cases |

Small extra: inside `g`, if `self.N` is `None` (g called before `fit`), I fall back to `x.shape[-1]`.

### 1.2 `BayesClassifier.py`

| TODO | Code | Why |
|---|---|---|
| 1 | `self.kde_list[k].fit(x_train_given_class)` | one KDE per class, fitted **only** on that class's points, so it estimates P(x \| C=k) |
| 2 | `P = self.kde_list[k].transform(x)` | evaluate the class-k density at x |
| 3 | `estimated_params = self.estimate_params(x_train_given_class)` | user-supplied parameter estimator (e.g. MLE of μ, Σ) |
| 4 | `self.density_params_list.append(estimated_params)` | index k ↔ `self.labels[k]` |
| 5 | `P = self.density(x, self.density_params_list[k])` | user-supplied density with class-k params |
| 6, 7 | `raise NotImplementedError(...)` | GMM comes in a later lab |

Small extra: `fit_custom_density` resets `self.density_params_list = []` first. Without that, calling `fit` twice would leave stale parameters at the front of the list.
`deepcopy` (already in the template) matters: without it, all classes would share **one** KDE object and the last `fit` would overwrite the others.

### 1.3 `PCA.py`

| TODO | Code |
|---|---|
| 1 | `μ = mean(x, axis=0)`, `σ = std(x, axis=0)` |
| 2 | `z = (x − μ) / σ` |
| 3 | `Σ = np.cov(z, rowvar=False)` (columns = variables) |
| 4 | `λs, U = np.linalg.eig(Σ)`, where **columns** of U are eigenvectors |
| 5 | `inds = np.argsort(λs)[::-1]`, `λs = λs[inds]`, `U = U[:, inds]` (reorder the **columns**, not the rows!) |
| 6 | `A = U[:, :L].T`, shape (L, n) |
| 7 | `Z = z @ A.T`, so Z (m×L) = X (m×n) · Aᵀ (n×L) |
| 8 | `X = (Z @ A) * σ + μ`, the inverse transform followed by de-standardization |

**Extra requirements (done and tested in the last cell of PCA.ipynb):**
- `PCA(new_dim, solver='svd')`: SVD of `Z/√(m−1)`; `λ = S²`, `U = Vᵀ.T` (already sorted descending).
- `new_dim` as a float in (0,1): `L = searchsorted(cumsum(λ/Σλ), new_dim) + 1`. On iris at 95% this keeps 2 components, the same as sklearn.
- Also stores `self.λs`, `self.explained_variance_ratio`, `self.L`.

### 1.4 Notebook TODOs
- **KDEstimator.ipynb:** both bandwidths for the test equal `0.7937005259840998`; filled `plot_3d` (min/max → `linspace(…,100)` → `meshgrid` → list comprehension of `kde.transform` → `plot_surface(x1,x2,z[i])`); bandwidths, bumps; TODO 9 uses Rect with `h = √12 · h_Silverman` (see §3).
  - TODO 7 says "10.9", but the plot title says "BW=10.0". I used 10.0 (almost certainly a typo).
- **PCA.ipynb:** `np.random.multivariate_normal(μ, Σ, 100)`, `PCA(new_dim=2)`, `PCA(new_dim=1)`.
- **Microbiologist.ipynb:** `KDEstimator(bump='Gauss', bandwidth=0.5)`, `BayesClassifier(mode='KDE', kde_config=kde_config)`; the CSV was already provided with the `Serial` header.

All written answers are in the notebooks' "Answer goes here" cells. §3 summarises them.

---

## 2. Results

**Microbes dataset (10 features, 10 classes, 24,421 train / 6,106 val):**

| Model | Val accuracy |
|---|---|
| Gaussian Naive Bayes | 26.0% |
| QDA | 34.1% |
| LDA | 37.4% |
| KDE Gauss, Scott (h≈2) | 79.8% |
| KDE Gauss, Silverman (h≈2) | 83.4% |
| **KDE Gauss, h = 0.5 (required)** | **97.7%** |
| KDE Gauss, h = 0.3 | 98.1% (no NaN rows) |
| KDE Gauss, h = 0.2 | 98.4% (3 NaN rows) |
| KDE Gauss, h = 0.1 | 98.5% (25 NaN rows) |
| KDE Rect, h = 0.5 / 1.0 | 97.4% / 97.3% (~145 NaN rows) |
| (reference) 1-Nearest-Neighbour | 98.5% |

Timing: QDA ≈ 0.6 s, KDE ≈ 5–7 s (fit + predict).

### 🔎 Two important things I discovered (good discussion material)
1. **Data leakage / duplicates:** 84% of the rows in `microbes.csv` are exact duplicates, and **97.6% of validation rows appear verbatim in the training set**. A KDE with a small h becomes a nearest-neighbour lookup, which is why "smaller h is always better" here and why 1-NN also gets 98.5%. On deduplicated data the picture would change (larger optimal h, lower accuracy).
2. **NaN probabilities:** with a small h (or a Rect bump), a query point can get density **exactly 0 for every class**: Gaussian underflow (`exp(-huge)` = 0.0), or a point outside every box for Rect. Then `predict_proba_x` computes 0/0 = NaN, and `argmax` silently returns index 0. The standard fix is to work with **log-densities and the log-sum-exp trick**.

---

## 3. Answers to the lab questions (cheat sheet)

**KDE: why larger h means smoother.** P(x) is an average of M bumps of width h. A small h gives narrow spikes on each data point (high variance, overfitting). A large h gives wide overlapping bumps that average out detail (high bias, oversmoothing; at h = 10 the blobs merge into one hill). h is the **bias–variance knob**.

**KDE: best h of {0.1, 0.5, 1, 2, 10}.** h ≈ 1–2: the three blob modes are visible and smooth. Silverman's rule gives h ≈ 1.97 for this data.

**KDE: which densities integrate to 1?** All of them. ∫ϕ = ∫g(x/h)/hᴺ dx = ∫g(u)du = 1 (substituting u = x/h, dx = hᴺdu), and an average of densities is a density.

**Silverman vs Scott.** Silverman = Scott × (4/(N+2))^(1/(N+4)). That factor is 1.06 for N=1, **exactly 1 for N=2** (so the 2D plots are identical), and 0.92 for N=10. Both scale like σ̂·M^(−1/(N+4)) because both come from minimizing AMISE under a Gaussian reference density.

**Why is Rect more discontinuous?** The box function jumps from 1 to 0 at its edge, so a sum of boxes is a piecewise-constant staircase. A Gaussian is infinitely differentiable. With the same h, the box also has a smaller spread (its std is h/√12).

**Which estimator is equivalent to the Rect bump?** A **histogram** with a moving window, also called the naive/Parzen-window estimator: P(x) = count of points in the cube of side h around x / (M·hᴺ).

**TODO 9 (smooth Rect, close to Gauss).** A box of width h has variance h²/12 per dimension; the Gaussian bump has variance h². Matching the variances gives h_rect = √12·h_gauss ≈ 3.46 × 1.97 ≈ 6.84.

**PCA: why is cov(Z) diagonal?** cov(XAᵀ) = AΣAᵀ = UᵀΣU = Uᵀ(UΛUᵀ)U = Λ. Σ is symmetric, so it has orthonormal eigenvectors. The new features are **uncorrelated**, and their variances are the eigenvalues.

**PCA: is separability lost?** 2 components: no. It's an invertible rotation (after standardizing), so the classes are still linearly separable. 1 component: projecting onto PC1 = [1,1]/√2 creates a slight overlap near 0, and the best threshold reaches 97%, not 100%. **PCA is unsupervised**: maximum variance ≠ maximum class separation (that's what LDA/Fisher does).

**PCA: orthogonal matrix.** UᵀU = I means a rotation (det +1) or a reflection (det −1). It preserves distances and angles, so plot 2 is plot 1 rigidly rotated so that PC1 lies on the x-axis.

**PCA: 1-component plot.** Each point is the orthogonal projection of x onto the PC1 line, z₁ = k₁ᵀx. PC1 is the line that minimizes the squared perpendicular (reconstruction) error.

**Microbes Q1: why GNB/LDA/QDA fail.** The class-conditional densities are not Gaussian: skewness reaches +9.8 (FilledArea) and −8.2 (EulerNumber), features are nearly discrete (EulerNumber has 79 unique values), and classes are multimodal. That's a mis-specified model, so its bias is large. KDE is non-parametric, plus the duplicate leakage above. GNB is worst because it also assumes independence, while features are correlated (|corr(Extrema, BoundingBox1)| = 0.97). **LDA > QDA** because QDA estimates 10 covariances with 55 parameters each (some classes have only ~490 points) on non-Gaussian data, so it has higher variance. LDA's single pooled covariance acts as a regularizer.

**Microbes Q2: complexity.** Gaussian inference costs O(K·N²) per point and is **independent of M** (≈1,000 ops). KDE costs O(M·N) per point (≈244,000 ops) and must store all the data, about 250× more work, and the gap grows with M. KDE fitting is cheap (it only stores data); Gaussian fitting is O(M·N²).

**Microbes Q3: why Silverman doesn't help.** h ≈ 1.94–2.07, about 4× the 0.5 we used, which oversmooths (83.4%). The reasons:
- The rule assumes a **normal reference density**.
- σ̂ is inflated by heavy tails; the robust variant uses min(σ, IQR/1.34).
- A single isotropic h is used for features whose stds range from 0.86 to 6.6.
- It optimizes density error (MISE), not classification accuracy.

---

## 4. What to study (in priority order)

### A. Bayes classifier foundations (from Lab 1)
- Bayes' rule: P(C|x) = P(x|C)P(C) / P(x); the MAP decision rule argmax_k P(x|C=k)P(C=k).
- Why the denominator P(x) can be dropped for `predict`, but we normalize in `predict_proba_x`.
- QDA vs LDA vs GNB: what each assumes about Σ_k (separate / shared / diagonal), and the resulting decision boundaries (quadratic / linear / axis-aligned quadratic).
- Parameter counts: QDA K·N(N+1)/2, LDA N(N+1)/2, GNB K·N for the covariance parts.
- MLE of μ, Σ; biased (÷n) vs unbiased (÷(n−1)) covariance and `np.cov(bias=...)`.

### B. Kernel density estimation
- Parametric vs non-parametric estimation; "lazy" learners.
- Histogram → Parzen window → smooth kernels (Gaussian, Epanechnikov, box).
- Why ϕ(x) = g(x/h)/hᴺ still integrates to 1 (change of variables).
- **Bandwidth = bias–variance trade-off**; MISE/AMISE; why h shrinks like M^(−1/(N+4)).
- Silverman's and Scott's rules, their Gaussian assumption, and the robust IQR version.
- The curse of dimensionality for KDE (needs exponentially more data as N grows).
- KDE classifier ≈ soft k-NN / Parzen classifier; complexity O(M·N) per query.
- Numerical issues: underflow, and log-density with **log-sum-exp**.

### C. PCA
- Covariance matrix, and why standardization matters (features on different scales dominate otherwise).
- Spectral theorem: a symmetric Σ has real eigenvalues and orthonormal eigenvectors, Σ = UΛUᵀ.
- Two derivations of PCA: **maximize projected variance** (Lagrange multipliers give the eigenvector problem) and **minimize reconstruction error**.
- Eigenvalues = variances of the PCs; explained variance ratio; scree plot; choosing L.
- Orthogonal matrices = rotations/reflections; projection; reconstruction X ≈ ZA.
- **SVD**: A = UΣVᵀ; the right singular vectors of X are the eigenvectors of XᵀX; σᵢ² = λᵢ·(m−1). Why SVD is numerically better (it never forms XᵀX, which squares the condition number).
- Eigenvector **sign ambiguity** (why the tests compare `np.abs`).
- PCA vs LDA (unsupervised vs supervised dimensionality reduction).

### D. NumPy skills used in this lab
- Broadcasting `(N,) − (M,N) → (M,N)`.
- `np.einsum('ij,ij->i', x, x)` (row-wise dot products); also know `np.sum(x*x, axis=1)` as an equivalent.
- `np.apply_along_axis`, `axis=-1`.
- `np.argsort(...)[::-1]`, fancy indexing of columns `U[:, inds]`.
- `np.meshgrid`, `np.c_[...]`, `ravel`/`reshape` for surface plots.
- `np.cov(rowvar=False)`, `np.linalg.eig` vs `eigh` vs `svd`, `np.cumsum`, `np.searchsorted`.
- `copy.deepcopy` vs a shared reference.

### E. ML practice lessons from the Microbiologist notebook
- Always check for **duplicates and train/test leakage** before trusting a high score.
- Check distributional assumptions (histograms, skewness) before choosing a parametric model.
- Hyperparameter tuning belongs on a validation set separate from the final test set (we tuned on `x_val`, which is optimistic).

---

## 5. Recommended materials
- Your lecture slides on **Bayes classifiers / density estimation / PCA** (the notebooks reference "proven in the lecture").
- *Pattern Recognition and Machine Learning*, Bishop: §2.5 (Non-parametric methods: histograms, KDE, k-NN) and §12.1 (PCA: max-variance and min-error formulations).
- *The Elements of Statistical Learning*, Hastie et al.: §6.6 (Kernel density estimation & classification), §4.3 (LDA/QDA), §14.5 (Principal components).
- Wikipedia: "Kernel density estimation" (especially *A rule-of-thumb bandwidth estimator*), "Principal component analysis", "Singular value decomposition", "Orthogonal matrix", "LogSumExp".
- StackExchange (linked in the notebook): *Kernel bandwidth: Scott's vs. Silverman's rules*; also *Relationship between SVD and PCA* on stats.stackexchange.
- 3Blue1Brown, *Essence of Linear Algebra*: eigenvectors/eigenvalues and change of basis (good for intuition on rotation by U).
- StatQuest: "PCA step-by-step" and "Naive Bayes" videos.
- NumPy docs: `einsum`, broadcasting, `apply_along_axis`, `linalg.eig`/`svd`.
- scikit-learn docs: `KernelDensity`, `PCA`, `QuadraticDiscriminantAnalysis`, as references to compare with.

---

## 6. Likely discussion questions (practice these out loud)
1. Walk through `P(self, x)` in KDE: what are the shapes at every step?
2. Why do we need `deepcopy` in `fit_kde`?
3. Why does ϕ divide by hᴺ and not by h?
4. What happens to KDE as M → ∞ and h → 0 (with M·hᴺ → ∞)? (It converges to the true density.)
5. How would KDE behave with N = 100 features? (Curse of dimensionality.)
6. Why sort eigenvectors by **columns**? What breaks if you sort rows?
7. Why is the inverse transform exact when L = n, and approximate otherwise? What is the reconstruction error? (The sum of the discarded eigenvalues.)
8. Why standardize before PCA? When might you *not* want to?
9. Eig vs SVD: which is more stable, and why?
10. Why did the QDA test fail on new scikit-learn? (Biased vs unbiased covariance.)
11. Is 97.7% on the microbes data a trustworthy estimate? (No: duplicates and leakage.)
12. How would you avoid the NaN probabilities? (Log-space and log-sum-exp.)
