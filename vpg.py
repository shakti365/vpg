import argparse
import gym
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

print(args.debug)

# Create environment
env = gym.make("CartPole-v0")
num_actions = 2
num_obs_feats = 4
gamma = 0.9
num_episodes = 2
learning_rate = 0.001

if args.debug:
    print(num_actions, num_obs_feats)

# Define a policy
def policy(state):
    """Policy network"""
    state_ = tf.reshape(state, [-1, num_obs_feats])
    layer = tf.keras.layers.Dense(num_actions)
    logits = layer(state_)
    action = tf.random.categorical(logits, num_samples=1)
    action_ = tf.reshape(action, [-1])
    log_phi = tf.nn.log_softmax(logits)
    action_mask = tf.one_hot(action_, depth=2)
    log_phi_a = action_mask * log_phi
    return action_.numpy()[0], log_phi_a


# Reset batch
ep_loss = []
for episode in range(num_episodes):

    # Reset environment and sample current observation
    done = False
    ep_obs_t = []
    ep_a = []
    ep_r = []
    ep_obs_t1 = []
    ep_log_phi_a = []
    dis_reward = tf.constant(0, dtype=tf.float32)
    obs_t = env.reset()
    # Run an episode
    while not done:
        # Get action and log probability
        action, log_phi_a = policy(obs_t)

        # Take action in the environment
        obs_t1, r, done, _ = env.step(action)
        
        # Store transition
        ep_obs_t.append(obs_t)
        ep_a.append(action)
        ep_r.append(r)
        ep_obs_t1.append(obs_t1)
        ep_log_phi_a.append(log_phi_a)

        # Make next observation as current observation
        obs_t = obs_t1

    # If episode is complete
    if done:
        pow_ = tf.range(0, len(ep_obs_t), dtype=tf.float32)
        gamma_ = tf.constant(gamma,  dtype=tf.float32)
        ep_gamma = tf.math.pow(gamma_, pow_)
        reward = tf.constant(ep_r, dtype=tf.float32)
        dis_reward = tf.reduce_sum(ep_gamma * reward)

        loss_ = tf.reduce_sum([log_phi_a * dis_reward for log_phi_a in ep_log_phi_a])
        ep_loss.append(loss_)

# Calculate expected mean of loss over all episodes in a batch
loss = tf.reduce_mean(ep_loss)

optimize = tf.keras.optimizers.Adam(learning_rate).minimize(-loss)


if args.debug:
    print("loss: ", loss)