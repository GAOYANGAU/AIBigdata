#!/usr/bin/env python3

"""
打砖块游戏: 训练脚本
使用 dqn 算法训练打砖块游戏模型, 目标为获取尽可能高的游戏分数
Contributor：huangsihan
Reviewer：xionglongfei
"""


import os
import time
import random
import threading

# 游戏环境库
import gym

# 计算使用
import numpy as np
import tensorflow as tf
from keras import backend as K

# 自定义库
from model import build_network

from game_env import AtariEnvironment

os.environ["KERAS_BACKEND"] = "tensorflow"

flags = tf.app.flags
flags.DEFINE_string('game', 'Breakout-v0', '游戏名, https://gym.openai.com/envs#atari')
flags.DEFINE_integer('agent_history_length', 4, '选用最近4帧作为一个状态')
flags.DEFINE_integer('resized_width', 84, '画面宽度')
flags.DEFINE_integer('resized_height', 84, '高度')
flags.DEFINE_float('learning_rate', 0.0001, '初始化学习率')

flags.DEFINE_integer('num_concurrent', 5, '线程数')

flags.DEFINE_string('summary_dir', 'summaries', 'tensorboard summaries 目录')
flags.DEFINE_string('experiment', 'dqn_breakout', '当前训练结果文件名')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', '模型目录')

flags.DEFINE_integer('tmax', 100000000, '训练步数')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, '减小 epsilon 值的步数')

flags.DEFINE_integer('target_network_update_frequency', 10000, '网络更新频率')
flags.DEFINE_integer('network_update_frequency', 32, '每个 actor 的更新频率')

flags.DEFINE_float('gamma', 0.99, '奖励值衰减系数')

flags.DEFINE_integer('checkpoint_interval', 600, '保存模块间隔步数')
flags.DEFINE_integer('summary_interval', 5, '记录 summary 间隔秒数')

flags.DEFINE_boolean('show_training', True, 'True: 显示训练图像')

FLAGS = flags.FLAGS

# 记录步数, 用来更新 T 网络, 每 T % target_network_update_frequency 更新一次
T = 0

TMAX = FLAGS.tmax

def get_num_actions():
    """
    获取该游戏可执行的动作
    """
    env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-v0" or FLAGS.game == "Breakout-v0"):
        num_actions = 3
    return num_actions

def build_graph(num_actions):
    """
    构建网络
    """
    # 创建 Q 网络
    s, q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="q-network")

    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # 同 Q 网络结构完全一致的目标网络 T(同文章中 target action-value 网络Q')
    st, target_q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="target-network")
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    # 将 Q 网络更新到 T 网络
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]

    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - action_q_values))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s,
                 "q_values" : q_values,
                 "st" : st,
                 "target_q_values" : target_q_values,
                 "reset_target_network_params" : reset_target_network_params,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update}

    return graph_ops

def setup_summaries():
    """
    记录训练日志
    """
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode_Reward", episode_reward)

    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Max_Q_Value", episode_ave_max_q)

    logged_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", logged_epsilon)

    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

    summary_op = tf.summary.merge_all()

    return summary_placeholders, update_ops, summary_op

def sample_final_epsilon():
    """
    结论来自 http://arxiv.org/pdf/1602.01783v1.pdf 5.1节
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Q-learning 学习线程
    """
    global TMAX, T

    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    st = graph_ops["st"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, update_ops, summary_op = summary_ops
    env = AtariEnvironment(gym_env=env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    s_batch = []
    a_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Starting thread ", thread_id, "with final epsilon ", final_epsilon)

    time.sleep(3 * thread_id)

    t = 0
    # 文章 22.6.2 小节, 第一层: 做 M 个 episode, 此处 M = TMAX
    while T < TMAX:
        s_t = env.get_initial_state()
        terminal = False

        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        # 文章 22.6.2 小节, 第二层: episode 内初始化一个 T 次的循环
        while True:
            # 在执行 Q 网络之前, 先保存 Q(s,a) 的值
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})

            # 选择下一个动作
            a_t = np.zeros([num_actions])
            action_index = 0

            # 文章 22.6.2 小节, 第三层: 用 episode 概率做随机动作
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            # 1-episode 的概率选择最大值
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # 减小 episode 值
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps

            # 在环境中做出动作
            # 下一个环境, 奖励值, 是否结束, 其他信息(保留值)
            s_t1, r_t, terminal, _ = env.step(action_index)

            # 累积梯度
            readout_j1 = target_q_values.eval(session = session, feed_dict = {st : [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + FLAGS.gamma * np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            # 更新环境
            s_t = s_t1
            T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # 文章 22.6.2 小节最后, 每隔 C 步设置 Q' = Q
            if T % FLAGS.target_network_update_frequency == 0:
                session.run(reset_target_network_params)

            # 更新 Q 网络
            if t % FLAGS.network_update_frequency == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict = {y : y_batch,
                                                          a : a_batch,
                                                          s : s_batch})
                s_batch = []
                a_batch = []
                y_batch = []

            if t % FLAGS.checkpoint_interval == 0:
                save_path = FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt"
                saver.save(session, save_path, global_step = t)

            # 回合结束打印日志
            if terminal:
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print("THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS", t/float(FLAGS.anneal_epsilon_timesteps))
                break

def train(session, graph_ops, num_actions, saver):
    """
    训练函数
    """
    # 设置训练环境, 每个环境对应一个线程
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    session.run(tf.global_variables_initializer())

    session.run(graph_ops["reset_target_network_params"])
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment

    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # 多线程启动actor-learner
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver)) for thread_id in range(FLAGS.num_concurrent)]
    for t in actor_learner_threads:
        t.start()

    # 展示训练页面
    last_summary_time = 0
    while True:
        if FLAGS.show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > FLAGS.summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
        if T > TMAX:
            break
    for t in actor_learner_threads:
        t.join()

def main(_):
    g = tf.Graph()
    session = tf.Session(graph=g)
    with g.as_default(), session.as_default():
        K.set_session(session)
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver()

        train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
    tf.app.run()
