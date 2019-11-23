import gym
import tensorflow as tf
import uuid
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import numpy as np


HP_LEARNING_RATE = hp.HParam("learning_rate", hp.RealInterval(0.001, 0.01))
METRIC_AVG_REWARD = "highest_avg_reward"


def run_vpg(hparams):

    tf.random.set_seed(0)
    # Create environment
    env = gym.make("CartPole-v0")
    num_actions = 2
    num_obs_feats = 4
    gamma = 0.9
    num_episodes = 64
    learning_rate = hparams[HP_LEARNING_RATE]
    num_epochs = 500

    EXPERIMENT_ID = f"HP-{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')}-{uuid.uuid4()}"
    LOGS_PATH = f"data/models/{EXPERIMENT_ID}/logs"
    CKPT_PATH = f"data/models/{EXPERIMENT_ID}/ckpt"

    writer = tf.summary.create_file_writer(LOGS_PATH)

    with writer.as_default():
        hp.hparams_config(hparams=[HP_LEARNING_RATE],
                          metrics=[hp.Metric(METRIC_AVG_REWARD,
                                             display_name="highest_avg_reward")])


        with writer.as_default():
            hp.hparams(hparams)

    # Define a policy
    policy = tf.keras.Sequential([
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(num_actions)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=policy)
    manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    global_episode_counter = 0
    highest_average_reward = 0
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
                global_episode_counter += 1
                while not done:
                    # Get action and log probability
                    obs_t_ = obs_t.reshape(-1, num_obs_feats)
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

                pow_ = tf.range(0, len(ep_obs_t), dtype=tf.float32)
                gamma_ = tf.constant(gamma,  dtype=tf.float32)
                ep_gamma = tf.math.pow(gamma_, pow_)
                reward = tf.constant(ep_r, dtype=tf.float32)
                dis_reward = tf.reduce_sum(ep_gamma * reward)
                dis_reward_ = dis_reward * tf.ones([1, len(ep_obs_t)])

                ep_log_phi_a_ = tf.reshape(tf.stack(ep_log_phi_a), (-1, len(ep_obs_t)))

                loss_ = tf.reduce_sum(ep_log_phi_a_ * dis_reward_)
                total_episode_reward = tf.reduce_sum(ep_r)

                ep_loss.append(loss_)
                ep_reward.append(total_episode_reward)

            # Calculate expected mean of loss over all episodes in a batch
            loss = tf.math.negative(tf.reduce_mean(ep_loss))

            with writer.as_default():
                tf.summary.scalar("epoch_loss", loss, epoch)

        avg_reward = sum(ep_reward) / num_episodes

        with writer.as_default():
            tf.summary.scalar("epoch_avg_reward", avg_reward, epoch)

        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))

        if avg_reward > highest_average_reward:
            highest_average_reward = avg_reward

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Average Reward: {avg_reward} Loss: {loss}")
            manager.save() 


    with writer.as_default():
        tf.summary.scalar(METRIC_AVG_REWARD, highest_average_reward,
                          step=1)


for learning_rate in np.linspace(HP_LEARNING_RATE.domain.min_value,
                      HP_LEARNING_RATE.domain.max_value, 5):
    hparams = {
        HP_LEARNING_RATE: learning_rate
    }
    run_vpg(hparams)
