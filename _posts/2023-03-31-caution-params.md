---
title: 'Caution Parameters in "Cautious Adaptation for Reinforcement Learning in Safety-Critical Settings"'
author:
  name: Maren Eberle
  affiliations:
    name: BCCN Berlin
mathjax: true

date: 2023-03-31
---

<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/adapt_state_1_1_it9.gif" width="200" height="200">
  <figcaption>Figure 1. CARL State after adaptation.</figcaption>
   </p>
</figure>



Safe Reinforcement Learning (Safe RL) is a field that spans diverse approaches to develop algorithms that maximize a return without failing or executing risky behavior (García and Fernández, 2015). This respectively implies avoiding catastrophic events or avoiding actions that lead to highly varying returns. Hence, safety can be implemented by various constraints posed on RL agents.
Within the framework of Safety-Critical Adaptation (SCA), an RL agent learns to act in different sandbox environments in the pretraining phase before it is placed in a target environment in the adaptation phase. In the target environment, the agent acts 'cautious', meaning that even during its initial exploration, it will not only try to maximize the return, but also avoid catastrophic states (Zhang et al., 2020). SCA is inspired by human behavior: When driving a new car for the first time, we will for example be more cautious hitting the gas pedal. Nascent pilots who are trained in a flight simulator where they might be able to perform numerous stunts will refrain from looping the loop when flying a real plane for the first time.
Zhang et al. (2020) transfer this idea to RL and propose Cautious Adaptation in Reinforcement Learning (CARL). They introduce two algorithms tackling different notions of safety. The first is based on Low Reward Risk-Aversion (CARL Reward) and the second on Catastrophic State Risk-Aversion (CARL State). In this blog post, I will set the approaches into context and provide results of expanded behavior for each CARL generated through the modification of parameters critical for adaptation. While Zhang et al. (2020) describe two of the adaptation parameters as caution (tuning) parameters, I argue that a third one directly controls the degree of caution exercised by CARL. The methodology is based on the work of Zhang et al. (2020) and the corresponding [code](https://github.com/jesbu1/carl). The marginally adapted code for this project can be found here](https://github.com/Safe-RL-Team/CARL-params).

# Context
García and Fernández (2015) categorize the main tendencies in Safe RL. Apart from a direct modification of the exploration process, the second tendency is the modification of the optimization criterion. For CARL, the overall exploration process is modified since it relies on pretraining in several environments. However, the key idea for CARL is based on changing the optimization criterion from pretraining to adaptation (Fig. 2). CARL Reward can be classified as an approach with 'worst-case criterion under inherent uncertainty' since it only pays attention to the worst rewards of possible actions during adaptation. Hence, it is a pessimistic version of the optimization used for pretraining. This notion of safety suggests that a low reward is oftentimes corresponding to catastrophic actions. In contrast, the notion of safety for CARL State acknowledges that state safety and reward don’t necessarily go hand in hand. Therefore, the state safety is additionally learned during pretraining. During adaptation, the optimization criterion used in pretraining is complemented with the state safety. This corresponds to an approach with a 'risk-sensitive criterion based on the weighted sum of return and risk'.

<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/context.png" alt="Figure1" class = "center">
  <figcaption>Figure 2. Approaches for Safe RL (adapted: García and Fernández, 2015, p. 1440).</figcaption>
  </p>
</figure>

CARL can be used on top of any model-based (1.) agent with model predictive control (MPC) planning (2.) and an ensemble model (3.) such as probabilistic ensembles with trajectory sampling (PETS (4.); Chua et al., 2018).
1.	A model-based approach is chosen to enable learning the transition probabilities of the sandbox environments. Importantly, these environments should be as varied as possible to capture epistemic uncertainty caused by different environments and prepare for the transfer into the out-of-domain (OOD) target environment. The predictions of the model are essential for the behavior during adaptation, since unknown actions are evaluated. Compared to a model-free approach, a model-based approach is more sample-efficient, which is of advantageous, especially for adaptation (Nagabandi et al., 2017). CARL Reward specifically relies on comparing several possible actions because it chooses the worst performing subset of those.
2.	CARL is supposed to stay safe over several actions. It is not enough to take a very rewarding action in one step if this leads to a catastrophic state later on. Instead, lower rewards over a long period of time should be preferred if this prevents reaching catastrophic states. With MPC, long-term planning is feasible by comparing whole action sequences. For every step, the agent finds the optimal solution of an open-loop optimal control problem over several next timesteps, but only executes the very first action in the optimal sequence. It then learns from the consequences in the environment and computes a new solution for the next step. Repeatedly solving the open-loop problem and integrating feedback makes this a case of closed-loop MPC ([Caron](https://scaron.info/robotics/open-closed-loop-model-predictive-control.html), n.d.).
3. Using an ensemble model allows for the specific formulation of CARL State because the state safety is based on a vote of the networks in the ensemble. Furthermore, an ensemble captures epistemic uncertainty caused by finite training samples.
4. PETS captures not only epistemic, but also aleatoric uncertainty that is caused by system stochasticity. This is done by making each model in the ensemble a probabilistic neural network which outputs distributions over states. This makes PETS competitive with model-free algorithms in terms of asymptotic performance (Chua et al., 2018).


# Methods
### Pretraining
During pretraining, PETS is used to train the ensemble. In short, PETS consists of an ensemble of bootstraps of $$B$$ neural networks. Several Monte Carlo samples of a state $$s_0, ...$$ called particles, are propagated to receive action sequences $$A = [a_1, ..., a_H]$$. $$H$$ is the planning horizon considered with MPC. For each action in a sequence, the dynamics model $$f$$ predicts a distribution over possible next states $$s_{t+1} =f(s_{t},a_{t}), t \in [1, H]$$. The reward $$r^i$$ is then calculated over the whole state trajectory for each particle $$i \in [1, N]$$. The optimization criterion for PETS is called action score and defined as

$$
R(A)=\sum_{i=0}^{N} r^i / N.
$$

Using MPC, the best action sequence $$A^*$$ is chosen with $$A^* = argmax_{A}(R(A))$$ and only the first action $$a_1$$ is executed. Then, the process is repeated for each step until the task horizon. To prepare for CARL, the same PETS agent is trained over several sandbox environments. For CARL State, additionally to $$f$$, a second model $$g$$ is pretrained to predict state safety (see [CARL State](#carl-state)). The weights of $$g$$ are fixed after pretraining but the adaptation training samples are still used to update $$f$$.

### Adaptation
#### CARL Reward
During adaptation, the training process is only slightly modified. For CARL Reward, a generalized action score is used:

$$
R_\gamma(A)=\sum_{i:r^i \leq v_{100-\gamma}(r)} r^i / N
$$

with caution parameter $$\gamma$$ and $$v_k(r)$$ the value of $$k^{th}$$ percentile of rewards. Effectively, only the worst subset of the trajectories is used for the decision with MPC. The size of this subset is defined by the caution parameter $$\gamma$$: The larger $$\gamma$$, the smaller the subset and the more pessimistic CARL should act.

#### CARL State
For CARL State, a state safety model $$g$$ is introduced that generates the predicted catastrophe cost

$$
g(A)=\sum_{i=1}^{H} P(s_i \in CatastrophicSet).
$$

The safety of an action sequence is thus determined by the probability of each state in the corresponding trajectory to belong to the $$CatastrophicSet$$ that consists of all catastrophic states. The approximated probability of catastrophe is

$$
P(s_i \in CatastrophicSet)=\frac {\sum_{j=1}^{E} \delta(c_{\theta_j}(s_{i-1}, a_{i-1}) > \beta)}{E}
$$

with the number of environments $$E$$, the ensemble parameters $$\{\theta_1, ..., \theta_{E}\}$$ and the caution tuning parameter $$\beta$$. Each state is hence assigned a probability by calculating the fraction of networks that ‘vote’ for the state being catastrophic. $$ \beta$$ acts as a threshold such that a model ‘votes yes’ if $$ c_{\theta_j}$$ predicts the probability of a catastrophe to appear with $$> \beta$$. Therefore, modifying $$\beta$$ changes how conservative each model votes for states being catastrophic. A lower $$\beta$$ leads to higher probabilities of catastrophe and supposedly more cautious behavior. For the adapted action score for CARL State, another adaptation parameter is introduced to weight the scaled catastrophe with the original action score:

$$
R_{\lambda}(A)=R(A) - \lambda g(A)
$$

with penalty weight $$\lambda > 0$$.

### Caution Parameters
The adaptation parameters are modified in the OpenAI Gym cartpole environment. Here, different pole lengths are used to create sandbox and target environments. The sandbox environments are sampled from a domain of pole lengths $$d \in [0.4, 0.8]$$. Different target domains with pole lengths $$td \in \{1, 1.5, 2\}$$ are considered. The $$CatastrophicSet$$ consists of all states where the pole falls below the cart or the cart falls of the rail. $$B = 5$$, $$N = 20$$ and $$H = 25$$ for a task horizon of $$200$$ are used.

So far, three parameters for adaptation have been presented:
1.	Caution parameter $$\gamma$$ for CARL Reward,
2.	Caution parameter $$\beta$$ for CARL State, and
3.	Penalty weight $$\lambda$$ for CARL State.

Zhang et al. (2020) arbitrarily set $$\gamma = 50$$ and $$\lambda * g(A) = 10000 * E$$. For all environments considered in the original paper, an ablation on $$\beta$$ is performed. Instead of regulating how many catastrophes CARL State encounters, the authors observe no clear relationship between executed caution and the value of $$\beta \in \{0.25, 0.5, 0.75\}$$. $$\beta = 0.5$$ performes slightly better than the others for all environments and is thus used. Overall, with these settings, CARL State acts more cautious than CARL Reward.

I argue that for both versions of CARL, the parameter choice should be investigated to detect optimally cautious behavior. Furthermore, the penalty weight $$\lambda$$ would be better suited to tune CARL State’s caution than $$\beta$$. While $$\beta$$ changes how conservative the networks in the ensemble vote, $$\lambda$$ can be used to scale the influence of the state penalty on the reward further. With larger $$\lambda$$, CARL State should act more cautious.
To test this, an agent is pretrained and the weights are used for adaptation for CARL Reward with $$\gamma \in \{30, 50, 70\}$$. For CARL State, a penalty scaling factor $$\lambda_2$$ is prepended $$\lambda$$ in the modified action score to enable easier comparison with the original results and code. CARL State is evaluated with $$\lambda_2 \in \{0.5, 1.0, 2.0\}$$. The penalty is now added for each particle with
```python
def catastrophe_cost_fn(obs, cost, beta, penalty_scale):  # new: penalty_scale implemented for lambda_2
        catastrophe_mask = obs[..., -1] > beta / 100
        cost[catastrophe_mask] += penalty_scale * CONFIG_MODULE.CATASTROPHE_COST
        return cost
```

with `CONFIG_MODULE.CATASTROPHE_COST = 10000` and  `obs[..., -1]` the catastrophe probabilities per particle as in the original code.

The goals for CARL are to
1.	Produce high task rewards (metric: maximum reward),
2.	Produce minimal catastrophic events (metric: cumulative number of catastrophe), and
3.	Adapt quickly (metric: other goals achieved in early adaptation training iterations).


Cautious behavior should be especially pertinent in target environments that are further out of distribution from the pretraining environments.

# Results

### Pretraining
During pretraining, a PETS agent is trained in several domains of the cartpole environment. To provide an insight into the pretraining process, training episodes 1, 5, and 20 are displayed in Figure 5. The agent stops running into catastrophic states early in the pretraining process (Fig. 3) and quickly learns to consistently produce the maximum reward at the same time (Fig. 4).

<details>
<summary>Figure 3.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/pretrain_cat.png" alt="Figure1" class = "center">
  <figcaption>Figure 3. Catastrophic events over pretraining phase training iterations.</figcaption>
  </p>
</figure>
</details>

<details>
<summary>Figure 4.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/pretrain_r.png" alt="Figure1" class = "center">
  <figcaption>Figure 4. Returns over pretraining phase training iterations.</figcaption>
  </p>
</figure>
</details>


<figure>
  <p style="text-align:center;">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/pretraining_it0.gif" width="200" height="200">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/pretraining_it4.gif" width="200" height="200">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/pretraining_it19.gif" width="200" height="200">
  <figcaption>Figure 5. PETS during pretraining episodes 1, 5, and 20, respectively.</figcaption>
  </p>
</figure>


### Adaptation
During adaptation, CARL Reward and CARL State are considered with different caution parameters $$\gamma$$ and $$\lambda_2$$. For each target domain over all adaptation training iterations, the reward, the maximum reward, and cumulative catastrophic events are measured.

#### Rewards
For pole length $$td = 1$$, all CARL State versions achieve a higher reward than CARL Reward versions from the first iteration on (Fig. 6) and reach their maximum slightly faster, but produce a high reward less consistently. All CARLs finally achieve a solid reward $$> 190$$ (Fig. 9). For CARL Reward, all versions perform similarly, but CARL Reward with $$\gamma = 50$$ starts out with slightly higher rewards. The CARL State versions perform similarly to each other, but $$\lambda_2 = 1$$ has a less consistent return.
<details>
<summary>Figure 6.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/r1.png" alt="Figure1" class = "center">
  <figcaption>Figure 6. Returns over adaptation phase training iterations for pole length td = 1. Solid lines represent the maximum reward over iterations, dotted lines represent actual reward per iteration. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

For pole length $$td = 1.5$$, CARL Reward and CARL State start out with similar rewards and adapt similarly fast (Fig. 7). Noticeably, CARL Reward with $$\gamma = 70$$ clearly outperforms all other CARLs in terms of reward performance at the end of the adaptation phase with a reward $$>190$$ while the next best performer is CARL State with $$\lambda_2 = 1.5$$ with a reward of $$\approx 123$$ (Fig. 9). The other CARL Reward versions don't increase the return much over training. Within the CARL State versions, CARL with $$\lambda_2 = 1$$ performs worse than the others (Fig. 7).
<details>
<summary>Figure 7.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/r1_5.png" alt="Figure1" class = "center">
  <figcaption>Figure 7. Returns over adaptation phase training iterations for pole length td = 1.5. Solid lines represent the maximum reward over iterations, dotted lines represent actual reward per iteration. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

For pole length $$td = 2$$, all CARL Reward version start out with a lower reward than CARL State versions. However, they increase the return slightly over training, while CARL State versions generally don’t. Finally, CARL Reward and state versions end up with similarly low returns (Fig. 8, 9).
<details>
<summary>Figure 8.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/r2.png" alt="Figure1" class = "center">
  <figcaption>Figure 8. Returns over adaptation phase training iterations for pole length td = 2. Solid lines represent the maximum reward over iterations, dotted lines represent actual reward per iteration. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

Figure 9 shows an overall comparison of final maximum adaptation return for CARL versions. For a pole length $$td = 1$$, all versions show very high rewards. As expected, different caution parameters have a more different influence for further OOD pole lengths: For $$td = 1.5$$, within CARL Reward versions, the most cautious ($$\gamma = 70$$) has the highest return. For CARL State versions, all modifications of the original ($$\lambda_2 = 1$$) perform better than the original. Overall, higher $$\lambda_2$$ tend to have a higher return. For $$td = 2$$,  for CARL Reward and State, the most cautious versions ($$\gamma = 70$$ and $$\lambda_2 = 0.5$$) have the highest returns.
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/heat_r.png" alt="Figure1" class = "center">
  <figcaption>Figure 9. Maximum return for the final training episode during adaptation phase for all versions of CARL.</figcaption>
  </p>
</figure>

#### Catastrophic Events
For pole length $$td = 1$$, all CARL Reward versions cause a catastrophic event in the first training iteration, the two new versions ($$\gamma \in \{30, 70\}$$) even cause one in the second iteration. The CARL State versions cause maximum one catastrophic event. Importantly, CARL State with $$\lambda_2 = 2$$ can avoid any catastrophic events during adaptation (Fig. 10, 13).
<details>
<summary>Figure 10.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/cat1.png" alt="Figure1" class = "center">
  <figcaption>Figure 10. Catastrophic events over adaptation phase training iterations for pole length td = 1. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

For pole length $$td = 1.5$$, all CARL version cause a catastrophic event in every training iteration (Fig.11, 13). CARL Reward with $$\gamma = 70$$ is an exception: It avoids a catastrophic event in the last iteration (Fig. 11).
<details>
<summary>Figure 11.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/cat1_5.png" alt="Figure1" class = "center">
  <figcaption>Figure 11. Catastrophic events over adaptation phase training iterations for pole length td = 1.5. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

For pole length $$td = 2$$, all CARL versions cause catastrophic events in each iteration (Fig. 12, 13).
<details>
<summary>Figure 12.</summary>
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/cat2.png" alt="Figure1" class = "center">
  <figcaption>Figure 12. Catastrophic events over adaptation phase training iterations for pole length td = 2. The numbers after 'reward' or 'state' determine the value of the caution parameter.</figcaption>
  </p>
</figure>
</details>

Figure 13 shows a comparison of cumulative catastrophic events. For a less OOD pole length, CARL State outperforms CARL Reward. Within CARL State versions, a higher $$\lambda_2$$ leads to more cautious behavior. CARL State learns to avoid catastrophe faster than CARL Reward (Fig 10). For pole length $$td = 1.5$$, within CARL Reward, the expected effect of less catastrophic events for more pessimistic versions was observed. For further OOD pole lengths, more caution does not help prevent catastrophic events for both CARL Reward and CARL State.
<figure>
  <p style="text-align:center;">
  <img src="https://M-Eberle.github.io/CARL/assets/docs/heat_cat.png" alt="Figure1" class = "center">
  <figcaption>Figure 13. Cumulative catastrophic events over adaptation phase training iterations for all versions of CARL.</figcaption>
  </p>
</figure>


In Figure 14 and Figure 15, examples of the behavior for different CARL versions and pole lengths are displayed. CARL Reward with $$\gamma = 70$$ and CARL State with $$\lambda_2 = 2$$ are compared since they are the most cautious versions for the respective CARLs. It was shown that CARL State can prevent catastrophic events from the first iteration for a less OOD pole length of $$td = 1$$ (Fig. 10). Thus, the first iteration is compared for the two CARL versions in Figure 14. While Carl Reward moves the cart fast and causes it to derail, CARL State maneuvers the cart slower and can already keep the pole upright and the cart more central on the rail.
For a further OOD pole length of $$td = 1.5$$, the last adaptation training iteration is compared. Here, the most cautious CARL Reward with $$\gamma = 70$$ resulted in safer actions than all other CARLs (Fig. 15). While CARL Reward reliably keeps the pole upright, CARL State moves the cart jerkily and can thus not balance the pole.

<figure>
  <p style="text-align:center;">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/adapt_reward_30_1_it0.gif" width="200" height="200">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/adapt_state_2_1_it0.gif" width="200" height="200">
  <figcaption>Figure 14. Comparing the behavior of CARL Reward (left) and CARL State (right) in the first adaptation iteration for pole length td = 1.</figcaption>
  </p>
</figure>

<figure>
  <p style="text-align:center;">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/adapt_reward_30_1_5_it9.gif" width="200" height="200">
    <img src="https://M-Eberle.github.io/CARL/assets/docs/vids/adapt_state_2_1_5_it9.gif" width="200" height="200">
  <figcaption>Figure 15. Comparing the behavior of CARL Reward (left) and CARL State (right) in the last adaptation iteration for pole length td = 1.5.</figcaption>
  </p>
</figure>

To sum up, a more pessimistic CARL Reward version does not improve performance close to training domains but tends to act safer and produce higher returns for further OOD pole length $$td = 1.5$$. For $$td = 2$$, pessimism still leads to higher returns. Over all environments, adaptation is slow.

CARL State with more cautious settings performs well both in terms of catastrophe prevention and high rewards for the least OOD pole length and adapts quicker than CARL Reward. Further OOD with pole length $$td = 1.5$$, more caution does not prevent catastrophic events but keeps rewards higher. Contrary, for pole length $$td = 2$$, more caution diminishes the return. CARL States reaches its final performance for both measures earlier than CARL State.

# Discussion
In this project, different settings for the caution parameters during the adaptation with CARL Reward and CARL State were considered in several OOD target environments. Consistent with the findings from Zhang et al. (2020), CARL Reward acts safer than CARL State further OOD. New findings are that for CARL Reward, more pessimism and hence caution results in safer behavior especially further OOD. For CARL State, a higher penalty for catastrophic states leads to more cautious behavior when close to the training domain and increases rewards even further OOD. This confirms that $$\gamma$$ tunes caution for CARL Reward and $$\lambda_2$$ acts as a caution parameter for CARL State. Overall, more caution leads to safer behavior.

The validity of these findings is limited due to the small number of runs: For every version of CARL, only one run was used. For further research, different pretrainings with several runs per adaptation setting should be considered to generate mean values for the (maximum) reward and better explanatory power for the number of catastrophic events. Furthermore, tuning the caution parameters is only benefitting in terms of safe learning if, in the long-run, the optimal caution parameter can be determined before placing CARL in a target environment. Therefore, higher-dimensional target environments, e.g. as in Zhang et al. (2020), should be used to try and find a general relationship between environment complexity, task and optimal caution parameters.

As pointed out by Zhang et al. (2020), CARL State is generally better applicable for environments where state safety and low rewards do not always align. For CARL State, a future parameter search could investigate the interaction of different values for the original caution parameter $$\beta$$ and the newly confirmed caution parameter $$\lambda_2$$.

Since there is a trade-off in performance between CARL Reward and State and the degree to which the target environments is OOD, it would be worth a try to use a CARL that combines both notions of reward with an optimization criterion such as

$$
R_{\gamma, \lambda_2}(A)=\sum_{i:r^i \leq v_{100-\gamma}(r)} r^i / N - \lambda_2 \lambda g(A)
$$


in combination with an estimation of how far OOD the target environment lays. After the first adaptation training iteration, if the target environment is expected to be close to training domain, $$\lambda_2$$ could be increased and $$\gamma$$ set to $$0$$. If the target environment is expected to be further OOD, the $$\lambda_2$$ could be smaller and $$\gamma$$ could be increased.


# References

<p style="padding-left: 2em; text-indent: -2em;"> Caron, S. (n. d.). <a href="https://scaron.info/robotics/open-closed-loop-model-predictive-control.html"><i>Open loop and closed loop model predictive control</i></a>.</p>

<p style="padding-left: 2em; text-indent: -2em;"> Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). <i>Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models</i>.  <a href="https://arxiv.org/abs/1805.12114">doi: 10.48550/arXiv.1805.12114</a>.</p>

<p style="padding-left: 2em; text-indent: -2em;"> García, J. & Fernández, F. (2015). A Comprehensive Survey on Safe Reinforcement Learning. <i>J. Mach. Learn. Res.</i> 16(<i>1</i>), pp. 14371480.  <a href="https://dl.acm.org/doi/10.5555/2789272.2886795"> doi: 10.5555/2789272.2886795</a>.</p>

<p style="padding-left: 2em; text-indent: -2em;"> Nagabandi, A., Kahn, G., Fearing, R. S., & Levine, S. (2017). <i>Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning</i>. <a href="https://doi.org/10.48550/arXiv.1708.02596"> doi: 10.48550/arXiv.1708.02596</a>.</p>

<p style="padding-left: 2em; text-indent: -2em;"> Zhang, J., Cheung, B., Finn, C., Levine, S., & Jayaraman, D. (2020).  <i>Cautious Adaptation For Reinforcement Learning in Safety-Critical Settings</i>. <a href="https://arxiv.org/abs/2008.06622"> doi: 10.48550/arXiv.2008.06622</a>.</p>

