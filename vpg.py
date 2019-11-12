import gym
import tensorflow as tf

# Create environment
env = gym.make("CartPole-v0")
num_actions = 2
num_obs_feats = 4
gamma = 0.9
num_episodes = 100
learning_rate = 0.001
num_epochs = 100

# Define a policy
policy = tf.keras.Sequential([
    tf.keras.layers.Dense(num_actions)
])

for epoch in range(num_epochs):
    # Reset batch
    ep_loss = []
    ep_reward = []
    with tf.GradientTape() as tape:
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
                obs_t_ = obs_t.reshape(-1,num_obs_feats)
                logits = policy(obs_t_)
                action = tf.random.categorical(logits, num_samples=1)
                action_ = tf.reshape(action, [-1])
                action_mask = tf.one_hot(action_, depth=2)
                log_phi = tf.nn.log_softmax(logits)
                log_phi_a = action_mask * log_phi

                # Take action in the environment
                obs_t1, r, done, _ = env.step(action_.numpy()[0])
                
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
                dis_reward_ = dis_reward * tf.ones([1, len(ep_obs_t)])

                ep_log_phi_a_ = tf.reshape(tf.stack(ep_log_phi_a), (-1, len(ep_obs_t)))

                loss_ = tf.reduce_sum(ep_log_phi_a_ * dis_reward_)

                ep_loss.append(loss_)
                ep_reward.append(tf.reduce_sum(ep_r))
                
        # Calculate expected mean of loss over all episodes in a batch
        loss = tf.math.negative(tf.reduce_mean(ep_loss))

    avg_reward = sum(ep_reward) / num_episodes
    
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    if epoch%10 == 0:
        print(f"Epoch: {epoch} Average Reward: {avg_reward} Loss: {loss}")