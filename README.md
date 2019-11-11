# Vanilla Policy Gradient

The objective of this algorithm is to maximise the expected reward over a trajectory:
$$
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta} }[R(\tau)]
$$
We learn the optimal policy by moving policy parameters $\theta$ towards gradient of the objective $\nabla_{\theta}J(\pi_{\theta})$

$$
\theta_{t+1} = \theta_{t} + \alpha \nabla_{\theta}J(\pi_{\theta})
$$

![Screenshot from 2019-11-10 12-59-56](/home/shivam/Pictures/Screenshot from 2019-11-10 12-59-56.png)



![Screenshot from 2019-11-10 13-05-25](/home/shivam/Pictures/Screenshot from 2019-11-10 13-05-25.png)

In order to estimate the expected gradient of the objective we sample a lot of trajectories and calculate the expected mean, hence the gradient of objective becomes:
$$
\nabla_{\theta} J(\pi_{\theta}) = \frac{1}{|D|}\sum_{\tau \in D}^{}{\sum_{t=0}^{T}{\nabla_{\theta}\log{\pi_{\theta}(a_t|s_t)}}R(\tau)}
$$


The simplest form of Policy Gradient algorithm can be implemented in this manner:

- *Initialise policy parameters $\pi_{\theta}$*
- *for one epoch of training k = 1 to K:*
	- *for one batch of training d = 1 to D:*
		- *reset environment*
		- *for one trajectory of training $\tau$:  t=0 to T:*
		  - *observe current state $s_t*
		  - *sample log probabilities of all action from current policy $\log \pi_{\theta}(a|s_t)$*
		  - *take action with highest log probability and observe $<r_t, s_{t+1}>$*
		  - store $<s_t, a_t, r_t, s_{t+1}>$ and $\log{\pi_{\theta}(a_t|s_t)}$
		- calculate $R(\tau) = \sum_{t=0}^{T}{\gamma^tr_t}$
		- calculate policy gradient $\sum_{t=0}^{T}{\nabla_{\theta}\log{\pi_{\theta}(a_t|s_t)}R(\tau)}$
	- *Calculate expected mean of policy gradient over all trajectories in batch*
	- *Apply gradient ascent for policy parameter*





## References

- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html