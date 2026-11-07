# Pairwise ranking of candidate jokes: model and derivations

This page is the precise reference for the math behind `judge/`. Every formula comes with
where it comes from. Scope is the current plan: **all-pairs comparisons + a Bradley–Terry fit,
then sort.** Adaptive pair selection and Borda scores are deliberately left out until
experiments show they are needed.

## 1. Setup and notation

For one prompt there are $n$ candidate jokes, indexed $i = 1, \dots, n$. The judge is asked
$T$ questions. Question $t$ involves an unordered pair $\lbrace a_t, b_t \rbrace$; a fair coin
decides which joke is shown first, $f_t$, and which second, $s_t$. The judge's answer is
recorded as the winner.

Counts derived from the log:

```math
w_{ij} = \#\lbrace t : i \text{ beat } j \rbrace, \qquad n_{ij} = w_{ij} + w_{ji}, \qquad W_i = \sum_{j \ne i} w_{ij}. \qquad\qquad (1.1)
```

$w_{ij}$ is how often $i$ beat $j$, $n_{ij}$ how often they met, $W_i$ the total wins of $i$.
With all-pairs asked once, $n_{ij} = 1$ for every $i \ne j$.

$\sigma(x) = 1 / (1 + e^{-x})$ is the logistic function. Two facts used throughout:

```math
\sigma(-x) = 1 - \sigma(x), \qquad \sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr). \qquad\qquad (1.2)
```

The second follows from differentiating: $\sigma'(x) = e^{-x}/(1+e^{-x})^2 = \sigma(x) \cdot \tfrac{e^{-x}}{1+e^{-x}}$.

## 2. The model

**Bradley–Terry (Bradley & Terry, 1952).** Each joke has a real-valued score $\theta_i$, and
answers to different questions are independent with

```math
P(i \text{ beats } j) = p_{ij} = \sigma(\theta_i - \theta_j) = \frac{\pi_i}{\pi_i + \pi_j}, \qquad \pi_i = e^{\theta_i}. \qquad\qquad (2.1)
```

The two forms agree: $\sigma(\theta_i - \theta_j) = \frac{1}{1 + e^{\theta_j - \theta_i}} = \frac{e^{\theta_i}}{e^{\theta_i} + e^{\theta_j}}$.

**Where the model comes from.** Suppose on each question the judge perceives joke $i$'s
funniness as $u_i = \theta_i + \varepsilon_i$, with noise $\varepsilon_i$ i.i.d. standard
Gumbel, and picks the larger. The difference of two independent standard Gumbel variables is
standard logistic, so

```math
P(u_i \gt  u_j) = P(\varepsilon_j - \varepsilon_i \lt  \theta_i - \theta_j) = \sigma(\theta_i - \theta_j). \qquad\qquad (2.2)
```

So Bradley–Terry is the "noisy perception, pick the max" model with one specific noise shape.

**Assumptions this commits us to.**

- Answers are independent given $\theta$ (no judge memory or drift).
- No ties: every question has a winner.
- Stochastic transitivity: if $p_{ij} \gt  \tfrac12$ and $p_{jk} \gt  \tfrac12$ then $p_{ik} \gt  \tfrac12$.
  A judge with genuinely cyclic preferences (A > B > C > A on average) cannot be fit exactly.
- Position does not matter. This one is false for LLM judges; the algorithm (Section 7) randomizes which joke is shown first.

**Only differences are identified.** Adding the same constant $c$ to every $\theta_i$ leaves
every $p_{ij}$ unchanged, so the data can never pin down the overall level of $\theta$.
Section 4 fixes this.

## 3. The likelihood

The probability of the observed answers is $\prod_{i \ne j} p_{ij}^{\,w_{ij}}$ (each win of
$i$ over $j$ contributes a factor $p_{ij}$). Taking logs and using
$\log p_{ij} = \log \pi_i - \log(\pi_i + \pi_j)$:

```math
\ell(\theta) = \sum_{i \ne j} w_{ij} \log p_{ij} = \sum_i W_i\, \theta_i - \sum_{i \lt  j} n_{ij} \log\bigl(e^{\theta_i} + e^{\theta_j}\bigr). \qquad\qquad (3.1)
```

The first sum collects $\log \pi_i = \theta_i$ once per win; the second collects
$\log(\pi_i + \pi_j)$ once per game between $i$ and $j$, whoever won.

**Gradient.** Differentiating, with $\partial_{\theta_i} \log(e^{\theta_i} + e^{\theta_j}) = p_{ij}$:

```math
\frac{\partial \ell}{\partial \theta_i} = W_i - \sum_{j \ne i} n_{ij}\, p_{ij}. \qquad\qquad (3.2)
```

Actual wins minus expected wins. The maximum likelihood estimate (MLE) sets this to zero for
every joke: **each joke's expected wins equal its actual wins.**

**Concavity.** Differentiating again (Section 6 does it in full), the Hessian is
$-L$, where $L$ is a weighted graph Laplacian, which is positive semidefinite. So $\ell$ is
concave and any stationary point is a global maximum.

**When the MLE fails to exist (Zermelo, 1929; Ford, 1957).** A finite maximizer exists, and is
unique up to the shift in Section 2, if and only if for every split of the jokes into two
non-empty groups, some joke in each group has beaten some joke in the other. Equivalently, the
directed "who beat whom" graph is strongly connected. If one joke lost every game, the
likelihood keeps increasing as its $\theta \to -\infty$, and no finite MLE exists. With 24
jokes and one game per pair this happens easily: a joke that loses all 23 games is plausible.

## 4. The prior: one dummy opponent

**The problem.** You fit the scores by maximizing the log-likelihood of the observed games:

```math
\hat\theta = \arg\max_\theta\ \ell(\theta), \qquad \ell(\theta) = \sum_i W_i\,\theta_i - \sum_{i\lt j} n_{ij}\log\bigl(e^{\theta_i} + e^{\theta_j}\bigr). \qquad\qquad (4.1)
```

This can fail in two ways.

- *No finite answer.* If joke $i$ lost every game, each of its terms
  $\log\sigma(\theta_j - \theta_i)$ keeps increasing as $\theta_i \to -\infty$. The likelihood
  has no maximum, only a supremum at $-\infty$. The same happens at $+\infty$ for a joke that
  won every game. With 24 jokes and one game per pair, this is likely to happen.
- *No unique answer.* $\ell$ depends only on differences $\theta_i - \theta_j$, so $\theta$ and
  $\theta + c\mathbf{1}$ fit equally well. $\ell$ is concave but not strictly concave: it is
  flat along the direction $\mathbf{1}$.

This is the same situation ridge regression addresses. When $X^\top X$ is singular, least
squares has infinitely many solutions; with near-singularity, the estimate blows up.

**Why add the penalty.**

```math
\ell_\alpha(\theta) = \ell(\theta) + \alpha\sum_i \bigl[\log\sigma(\theta_i) + \log\sigma(-\theta_i)\bigr] \qquad\qquad (4.2)
```

It makes the problem well-posed. Each penalty term has second derivative
$-2\sigma(\theta_i)(1-\sigma(\theta_i)) \lt  0$ and tends to $-\infty$ as $|\theta_i| \to \infty$.
So $\ell_\alpha$ is strictly concave and goes to $-\infty$ in every direction: a maximizer
always exists, is unique, and is finite.

**Stationarity condition.** The prior term's derivative is
$\alpha(1 - \sigma(\theta_i)) - \alpha\,\sigma(\theta_i) = \alpha - 2\alpha\,\sigma(\theta_i)$.
Writing $p_{i0} = \sigma(\theta_i)$ (the probability of beating the dummy), the maximizer
satisfies, for every $i$:

```math
W_i + \alpha = \sum_{j \ne i} n_{ij}\, p_{ij} + 2\alpha\, p_{i0}. \qquad\qquad (4.3)
```

Same rule as before, with the pseudo-games counted: actual wins (real + virtual) equal
expected wins (against real jokes + against the dummy).

**Centering.** The reported score is $\tilde\theta_i = \hat\theta_i - \bar{\hat\theta}$. This
changes no difference $\theta_i - \theta_j$, hence no $p_{ij}$ and no ranking. It is a
display convention: "above 0" means "better than this prompt's average joke." Anything that
involves the dummy (Section 6) must use the uncentered $\hat\theta$.

## 5. Computing the estimate: the MM algorithm (Hunter, 2004)

**Goal.** Find the $\theta$ that satisfies the stationarity condition (4.3) for every joke.
This section shows why (4.3) cannot be solved directly, and derives the update the code uses.

**Rewrite the objective in $\pi$.** Substitute $\theta_i = \log \pi_i$ into (4.2). The data
part is (3.1). For the penalty part, $\sigma(\theta_i) = \frac{\pi_i}{\pi_i + 1}$ and
$\sigma(-\theta_i) = \frac{1}{\pi_i + 1}$, so
$\log\sigma(\theta_i) + \log\sigma(-\theta_i) = \log \pi_i - 2\log(\pi_i + 1)$. Together:

```math
\ell_\alpha(\pi) = \sum_i (W_i + \alpha)\log \pi_i \;-\; \sum_{i\lt j} n_{ij}\log(\pi_i + \pi_j) \;-\; 2\alpha\sum_i \log(\pi_i + 1) \qquad\qquad (5.1)
```

**Why it cannot be solved directly.** Setting $\partial \ell_\alpha / \partial \pi_i = 0$ in (5.1):

```math
\frac{W_i + \alpha}{\pi_i} = \sum_{j \ne i} \frac{n_{ij}}{\pi_i + \pi_j} + \frac{2\alpha}{\pi_i + 1} \qquad\qquad (5.2)
```

This is (4.3) written in $\pi$. Every $\pi_j$ appears inside the denominators on the right, so
the $n$ equations are coupled and nonlinear. The culprit is the term $-\log(\pi_i + \pi_j)$ in
(5.1): a log of a sum mixes two unknowns.

**The idea.** Replace each $-\log(\text{sum})$ in (5.1) by something linear in $\pi$ that lies
below it and touches it at the current guess $\pi^{(k)}$. Then the problem splits into one easy
equation per joke.

**The inequality.** $\log$ is concave, so it lies below its tangent line at any $y \gt  0$:
$\log x \le \log y + \frac{x - y}{y}$ for all $x \gt  0$. Negating both sides:

```math
-\log x \;\ge\; -\log y - \frac{x}{y} + 1, \qquad \text{equality iff } x = y \qquad\qquad (5.3)
```

**Apply (5.3) to the pair terms**, with $x = \pi_i + \pi_j$ (unknown) and
$y = \pi_i^{(k)} + \pi_j^{(k)}$ (current guess, a known number):

```math
-\log(\pi_i + \pi_j) \;\ge\; -\log\bigl(\pi_i^{(k)} + \pi_j^{(k)}\bigr) - \frac{\pi_i + \pi_j}{\pi_i^{(k)} + \pi_j^{(k)}} + 1 \qquad\qquad (5.4)
```

**Apply (5.3) to the penalty terms**, with $x = \pi_i + 1$ and $y = \pi_i^{(k)} + 1$:

```math
-\log(\pi_i + 1) \;\ge\; -\log\bigl(\pi_i^{(k)} + 1\bigr) - \frac{\pi_i + 1}{\pi_i^{(k)} + 1} + 1 \qquad\qquad (5.5)
```

**Plug both into (5.1).** Multiply (5.4) by $n_{ij} \ge 0$ and sum over pairs, multiply (5.5) by
$2\alpha \gt  0$ and sum over jokes, and substitute into (5.1). The first sum of (5.1) is kept
as is. Every term that does not contain the unknown $\pi$ is collected into a number
$C^{(k)}$:

```math
\ell_\alpha(\pi) \;\ge\; Q(\pi \mid \pi^{(k)}) + C^{(k)}, \qquad \text{equality at } \pi = \pi^{(k)} \qquad\qquad (5.6)
```

where

```math
Q(\pi \mid \pi^{(k)}) = \sum_i (W_i + \alpha)\log \pi_i \;-\; \sum_{i\lt j} n_{ij}\,\frac{\pi_i + \pi_j}{\pi_i^{(k)} + \pi_j^{(k)}} \;-\; 2\alpha\sum_i \frac{\pi_i}{\pi_i^{(k)} + 1} \qquad\qquad (5.7)
```

Equality in (5.6) at $\pi = \pi^{(k)}$ holds because (5.4) and (5.5) are equalities when
$x = y$.

**Regroup (5.7) by joke.** In the middle sum, the pair $\lbrace i, j \rbrace$ contributes
$\pi_i$ to joke $i$ and $\pi_j$ to joke $j$, so it can be rewritten as a sum over jokes:

```math
Q(\pi \mid \pi^{(k)}) = \sum_i \Bigl[(W_i + \alpha)\log \pi_i - \pi_i\, D_i^{(k)}\Bigr], \qquad D_i^{(k)} = \sum_{j \ne i} \frac{n_{ij}}{\pi_i^{(k)} + \pi_j^{(k)}} + \frac{2\alpha}{\pi_i^{(k)} + 1} \qquad\qquad (5.8)
```

$D_i^{(k)}$ is a known number: it uses only the current guess.

**Maximize (5.8).** Each bracket depends on one $\pi_i$ only and is concave in it. Setting its
derivative $\frac{W_i + \alpha}{\pi_i} - D_i^{(k)}$ to zero:

```math
\pi_i^{(k+1)} = \frac{W_i + \alpha}{D_i^{(k)}} \qquad\qquad (5.9)
```

This is the update. Compare with (5.2): it is the same equation, with the right side of (5.2)
evaluated at the current guess instead of the unknown. At convergence,
$\pi^{(k+1)} = \pi^{(k)}$, and (5.9) becomes (5.2) exactly.

**Why each step improves the fit.**

```math
\ell_\alpha(\pi^{(k+1)}) \;\ge\; Q(\pi^{(k+1)} \mid \pi^{(k)}) + C^{(k)} \;\ge\; Q(\pi^{(k)} \mid \pi^{(k)}) + C^{(k)} \;=\; \ell_\alpha(\pi^{(k)}) \qquad\qquad (5.10)
```

The first $\ge$ is (5.6). The second $\ge$ holds because $\pi^{(k+1)}$ maximizes (5.8). The $=$
is the equality case of (5.6). Since (4.2) has a unique maximizer (Section 4), the iterates
converge to it (Hunter, 2004).

**In the code.** Since (5.8) splits by joke, (5.9) updates all jokes at once from the old
values:

```python
N = W + W.T                                   # n_ij
wins = W.sum(axis=1) + alpha                  # W_i + alpha, numerator of (5.9)
D = (N / (pi[:, None] + pi[None, :])).sum(axis=1) + 2 * alpha / (pi + 1.0)   # D_i^(k), (5.8)
pi = wins / D                                 # (5.9)
```

**Stopping.** The code stops when $\max_i |\log \pi_i^{(k+1)} - \log \pi_i^{(k)}| \lt  \text{tol}$.
Small steps do not strictly guarantee closeness to the optimum. A direct check is how far
(4.3) is from holding:

```math
r_i = W_i + \alpha - \sum_{j \ne i} n_{ij}\, p_{ij} - 2\alpha\, p_{i0} \qquad\qquad (5.11)
```

which is exactly 0 at the optimum. Reporting $\max_i |r_i|$ after the fit costs one line.

## 6. Uncertainty

**Hessian.** The Hessian is the matrix of second derivatives of the objective (4.2),
$H_{im} = \partial^2 \ell_\alpha / \partial\theta_i\,\partial\theta_m$. It measures how sharply the
objective curves around the optimum, which is what the uncertainty below is built from. It is
obtained in three steps: write the gradient, find the derivatives of the probabilities in it,
then differentiate the gradient once more.

*The gradient.* The data part's derivative is (3.2). The penalty part's derivative, using
$\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$ from (1.2), is
$\alpha(1 - \sigma(\theta_i)) - \alpha\,\sigma(\theta_i) = \alpha - 2\alpha\,\sigma(\theta_i)$.
Adding them, with $p_{i0} = \sigma(\theta_i)$:

```math
g_i = \frac{\partial \ell_\alpha}{\partial \theta_i} = W_i + \alpha - \sum_{j \ne i} n_{ij}\, p_{ij} - 2\alpha\, p_{i0} \qquad\qquad (6.1)
```

Setting $g_i = 0$ gives the stationarity condition (4.3).

*Derivatives of the probabilities.* $p_{ij} = \sigma(\theta_i - \theta_j)$ depends only on
$\theta_i$ and $\theta_j$. By the chain rule and $\sigma' = \sigma(1 - \sigma)$ from (1.2), the
inner derivative of $\theta_i - \theta_j$ is $+1$ with respect to $\theta_i$ and $-1$ with
respect to $\theta_j$:

```math
\frac{\partial p_{ij}}{\partial \theta_i} = p_{ij}(1 - p_{ij}), \qquad \frac{\partial p_{ij}}{\partial \theta_j} = -\,p_{ij}(1 - p_{ij}), \qquad \frac{\partial p_{ij}}{\partial \theta_m} = 0 \ \ (m \ne i, j) \qquad\qquad (6.2)
```

Likewise $p_{i0} = \sigma(\theta_i)$ depends only on $\theta_i$:

```math
\frac{\partial p_{i0}}{\partial \theta_i} = p_{i0}(1 - p_{i0}), \qquad \frac{\partial p_{i0}}{\partial \theta_m} = 0 \ \ (m \ne i) \qquad\qquad (6.3)
```

*Diagonal entries: differentiate (6.1) with respect to $\theta_i$.* Go term by term. $W_i$ and
$\alpha$ are constants, so their derivative is 0. Each $-n_{ij}\,p_{ij}$ gives
$-n_{ij}\,p_{ij}(1 - p_{ij})$ by the first part of (6.2). The term $-2\alpha\,p_{i0}$ gives
$-2\alpha\,p_{i0}(1 - p_{i0})$ by (6.3). Adding:

```math
H_{ii} = \frac{\partial g_i}{\partial \theta_i} = -\sum_{j \ne i} n_{ij}\, p_{ij}(1 - p_{ij}) - 2\alpha\, p_{i0}(1 - p_{i0}) \qquad\qquad (6.4)
```

*Off-diagonal entries: differentiate (6.1) with respect to $\theta_j$, $j \ne i$.* In the sum,
only the term with this particular $j$ contains $\theta_j$; all others have derivative 0 by
the third part of (6.2). That one term gives $-n_{ij} \cdot \bigl(-p_{ij}(1 - p_{ij})\bigr)$ by
the second part of (6.2). The term $-2\alpha\,p_{i0}$ does not contain $\theta_j$, by (6.3). So:

```math
H_{ij} = \frac{\partial g_i}{\partial \theta_j} = n_{ij}\, p_{ij}(1 - p_{ij}) \qquad (i \ne j) \qquad\qquad (6.5)
```

*Why the observed and expected information are equal.* The Fisher information is the expected
negative Hessian, $\mathcal{I} = -\mathbb{E}[H]$, where the expectation is over the judge's
random answers. The data enters (6.1) only through $W_i$, the win counts, and $W_i$ is a
constant term that vanished when differentiating. So (6.4) and (6.5) contain no data, only the
fitted probabilities, the expectation changes nothing, and

```math
\mathcal{I} = -H \qquad\qquad (6.6)
```

**Covariance.** Near the optimum $\ell_\alpha$ is approximately quadratic, so the posterior is
approximately Gaussian (Laplace approximation), equivalently the large-sample distribution of
the estimate:

```math
\hat\theta \approx \mathcal{N}\bigl(\theta,\ \Sigma\bigr), \qquad \Sigma = \mathcal{I}^{-1}. \qquad\qquad (6.7)
```

$\mathcal{I}$ in (6.6) is positive definite: by (6.4) and (6.5) it is a weighted graph Laplacian
plus a positive diagonal from the prior, so the inverse in (6.7) exists. Without the prior it would not: every row of the Laplacian sums to zero.

**Exact versus diagonal standard errors.**

- Exact: $\mathrm{se}_i = \sqrt{\Sigma_{ii}}$.
- Diagonal approximation (current code): $\mathrm{se}_i \approx 1 / \sqrt{\mathcal{I}_{ii}}$.

For any positive definite matrix $A$, Cauchy–Schwarz gives
$1 = (e_i^\top e_i)^2 \le A_{ii}\,[A^{-1}]_{ii}$, so
$[A^{-1}]_{ii} \ge 1 / A_{ii}$: **the diagonal version always underestimates the uncertainty.**
It treats every opponent's score as known exactly. With $n = 24$, inverting a $24 \times 24$
matrix is negligible, so the exact version costs nothing.

**What ranking actually needs: uncertainty of differences.** Whether joke $i$ is really better
than joke $j$ depends on $\hat\theta_i - \hat\theta_j$, whose variance

```math
\mathrm{Var}(\hat\theta_i - \hat\theta_j) = \Sigma_{ii} + \Sigma_{jj} - 2\,\Sigma_{ij} \qquad\qquad (6.8)
```

needs the off-diagonal covariance $\Sigma_{ij}$. The diagonal approximation cannot provide it.

**Centered scores.** Centering is the linear map $C = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$,
so $\mathrm{Cov}(\tilde\theta) = C\,\Sigma\,C^\top$. Differences are unaffected by centering.

## 7. The algorithm in pseudo-code

Pseudo-code for one prompt, written to explain the method, not to run. Each step names the
equation it computes.

```text
RANK(prompt, jokes 1..n, judge, alpha):

    # collect the games
    w[i][j] = 0 for all i, j                         # wins of i over j            (1.1)
    for each pair {i, j}, in random order:
        flip a fair coin to choose which of i, j is shown first
        ask the judge which joke is funnier
        w[winner][loser] += 1

    n[i][j] = w[i][j] + w[j][i]                      # games between i and j       (1.1)
    W[i]    = sum over j of w[i][j]                  # total wins of i             (1.1)

    # fit the scores (Section 5)
    pi[i] = 1 for every joke                         # start: every joke average
    repeat:
        for every joke i, using the current pi:
            D[i] = sum over j != i of n[i][j] / (pi[i] + pi[j])
                   + 2*alpha / (pi[i] + 1)           #                             (5.8)
        for every joke i:
            pi[i] = (W[i] + alpha) / D[i]            # update all jokes at once    (5.9)
    until no log(pi[i]) changed by more than tol
    theta[i] = log(pi[i])                            # still uncentered

    # uncertainty (Section 6)
    p[i][j] = sigma(theta[i] - theta[j])             # win probabilities           (2.1)
    p0[i]   = sigma(theta[i])                        # vs the dummy at 0
    I[i][j] = - n[i][j] * p[i][j] * (1 - p[i][j])    # off-diagonal of -H          (6.5)
    I[i][i] = sum over j != i of n[i][j] * p[i][j] * (1 - p[i][j])
              + 2*alpha * p0[i] * (1 - p0[i])        # diagonal of -H              (6.4)
    Sigma   = inverse of I                           #                             (6.7)
    se[i]   = sqrt(Sigma[i][i])

    # report
    theta[i] = theta[i] - mean(theta)                # centering, for display only
    return jokes sorted by theta (best first), with theta and se
```

Two points the pseudo-code makes visible:

- The fitting loop updates every joke from the **old** values, then replaces them all at once. This is
  allowed because the surrogate (5.8) splits into one independent term per joke.
- The uncertainty block uses the **uncentered** scores, because the dummy is at exactly 0 only before
  centering. Centering waits until the report.
