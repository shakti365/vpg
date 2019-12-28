import gym
import tensorflow as tf


def play(experiment_id):

    # Create env
    env = gym.make("CartPole-v0")
    num_actions = 2
    num_obs_feats = 4

    CKPT_PATH = f"data/models/{experiment_id}/ckpt"

    policy = tf.keras.Sequential([
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(num_actions)
    ])

    ckpt = tf.train.Checkpoint(net=policy)
    manager = tf.train.CheckpointManager(ckpt, CKPT_PATH, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint).expect_partial()

    obs_t = env.reset()
    done = False
    env.render()
    input()
    while not done:
        env.render()
        obs_t_ = obs_t.reshape(-1, num_obs_feats)
        logits = policy(obs_t_)
        action = tf.argmax(logits, axis=1)
        obs_t1, r, done1, _ = env.step(action.numpy()[0])
        obs_t = obs_t1




if __name__=="__main__":
    experiment_id = "HP-2019-12-07-17-09-c9071aff-1f4a-4985-968f-d611ba776b71"
    play(experiment_id)
