---
layout: post
title: 'The Bradley-Terry Model'
date: 2026-08-21 12:00
description: "From pairwise comparisons to RLHF reward models"
tags: [Machine Learning, Reinforcement Learning,Artificial Intelligence]
categories: Machine Learning
tabs: true
---
I got an interview that wanted me to write Leetcode-style problems in C language. After three days of hard work, They just asked me two RL problems. What can I say? Maybe just record them.

# 1. What's reward hacking and how to solve it?

Reward hacking: the policy model finds regions where the reward model is wrong (blind spots, spurious correlations) and exploits them to get a high score without actual quality improving. It's Goodhart's Law in action — once the RM's score becomes the optimization target, it stops being a valid measure of what you actually wanted.

Specific Problems:

* **Length bias**: for example, because the reward model scores longer responses higher, the policy model tends to generate longer answers — but the reasoning quality isn't necessarily better.
* **Format bias**: for example, the reward model prefers responses with markdown formatting, causing the policy model to favor generating markdown-formatted answers.
* **Out-of-distribution bias**: since the reward model's training data only covers 50% of the data distribution, when serving encounters the other 50% of scenarios, the RM's predictions become random or extreme — for instance, the model discovers that gibberish gets an extremely high score from the reward model, causing the policy model to start producing nonsense.

Solution:

* **Length alignment**: align the length of positive and negative examples, or add a length penalty.
* **Format alignment**: create negative examples by changing only the logic (not the formatting) of positive examples and add them to reward model training, forcing the model to focus on the response's logic rather than its formatting.

Fix the model:
Uncertainty estimation (filtering out low-confidence predictions)

* Have the RM output not just a score but also a "confidence" value. If, during RL, a given trajectory's RM confidence turns out to be extremely low (possibly a reward-hacking path), reduce its reward accordingly.

Regularizing RM training

* Constrain the RM's information bottleneck, forcing it to discard non-semantic features like word count and punctuation. InfoRM does this by introducing a "bottleneck layer" into the model, requiring it to use as little information as possible when predicting a score while keeping prediction accuracy as high as possible. Since punctuation, formatting, length, etc. contribute only weakly to response quality, they get filtered out first.

---

# 2.What's Bradley-Terry model and how to implement it out by Python? Also, how your function solve the problem of noise annotator?
To be frank, I definitely know what the Bradley-Terry model is based on my,low-key, strong machine learning and reinforcement learning background. But if you didn't tell me, I'm gonna get RL problems instead of C, and I have no time to review, that's not fair.

## The setup

For reward model $r_{\theta}(x,y)$,(x,y) is the (state,result), for llm state is the prompt. $\theta$ is the model parameter.

For the answer with preference $y_w$ outstanding from $y_l$, we write it mathematically:

$$P(y_w \succ y_l|x) = \frac{e^{r_{\theta}(x,y_w)}}{e^{r_{\theta}(x,y_w)}+e^{r_{\theta}(x,y_l)}}.$$

With a dataset of $N$ comparisons $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^{N}$, where $y_w$ is the preferred (winning) response and $y_l$ the losing one:

$$
\begin{aligned}
l(\theta|x) &= -\sum_{i=1}^{N} \ln \frac{\exp\big(r_{\theta}(x^{(i)}, y_w^{(i)})\big)}{\exp\big(r_{\theta}(x^{(i)}, y_w^{(i)})\big) + \exp\big(r_{\theta}(x^{(i)}, y_l^{(i)})\big)} \\[4pt]
&= -\sum_{i=1}^{N} \ln \frac{1}{1 + \exp\big(r_{\theta}(x^{(i)}, y_l^{(i)}) - r_{\theta}(x^{(i)}, y_w^{(i)})\big)} \\[4pt]
&= -\sum_{i=1}^{N} \ln \sigma\big(r_{\theta}(x^{(i)}, y_w^{(i)}) - r_{\theta}(x^{(i)}, y_l^{(i)})\big),
\end{aligned}
$$

where $\sigma(.)$ is the sigmoid function.




## Annotator noise

Human preference labels aren't clean, so the plain loss above needs correcting. Two ways to do it:

**Per-annotator inverse-temperature $\beta_a$** (needs annotator IDs) - this is Thurstone's 1927 Case V: each annotator has their own noise *scale* inside the same Gumbel model, so their comparisons follow

$$P_a(y_w \succ y_l) = \sigma\left(\beta_a(r_\theta(x,y_w)-r_\theta(x,y_l))\right)$$

$\beta_a \to \infty$ is a noiseless annotator, $\beta_a \to 0$ is pure guessing.

**Global label-flip rate $\epsilon$** (no IDs needed) - assume a flat fraction of labels are just flipped after the fact:

$$P_{obs}(y=1) = (1-\epsilon)\sigma(\Delta) + \epsilon\sigma(-\Delta)$$

Fitting the *uncorrected* sigmoid to noisy labels is provably biased - as $\Delta \to \infty$ the true model saturates at $1-\epsilon$, and an uncorrected sigmoid can only match that by shrinking every estimated gap toward zero.


## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """r_theta(x, y) -> scalar reward, given a pooled embedding of (x, y)."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, xy_embed):
        return self.head(self.backbone(xy_embed)).squeeze(-1)


def bt_reward_loss(r_chosen, r_rejected):
    """L_RM = -E[ log sigma(r_w - r_l) ]"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

Annotator-noise version:

```python
class AnnotatorNoiseModel(nn.Module):
    """Per-annotator beta_a, softplus-parameterized to keep beta > 0."""

    def __init__(self, num_annotators, init_beta=1.0):
        super().__init__()
        init_raw = torch.log(torch.expm1(torch.tensor(init_beta)))
        self.raw_beta = nn.Parameter(init_raw * torch.ones(num_annotators))

    def beta(self, annotator_ids):
        return F.softplus(self.raw_beta[annotator_ids])


def bt_loss_annotator_noise(r_chosen, r_rejected, annotator_ids, noise_model):
    delta = r_chosen - r_rejected
    beta_a = noise_model.beta(annotator_ids)
    return -F.logsigmoid(beta_a * delta).mean()


def bt_loss_flip_noise(r_chosen, r_rejected, eps):
    delta = r_chosen - r_rejected
    log_p, log_1mp = F.logsigmoid(delta), F.logsigmoid(-delta)
    log_lik = torch.logaddexp(
        torch.log(torch.tensor(1 - eps)) + log_p,
        torch.log(torch.tensor(eps)) + log_1mp,
    )
    return -log_lik.mean()
```

That's the whole arc: a Gumbel noise assumption gives you the sigmoid, MLE gives you the training loss, and swapping the noise model gives you a way to handle unreliable human labels without throwing away the theory.
