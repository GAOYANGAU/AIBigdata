#!/usr/bin/env python3

"""
打砖块游戏: 测试脚本
使用训练好的模型自动玩打砖块游戏
Contributor：huangsihan
Reviewer：xionglongfei
"""


import os
import shutil
import gym

import numpy as np
import tensorflow as tf
from keras import backend as K

from model import build_network
from game_env import AtariEnvironment

flags = tf.app.flags
flags.DEFINE_string('game', 'Breakout-v0', '游戏名, https://gym.openai.com/envs#atari')
flags.DEFINE_integer('agent_history_length', 4, '选用最近4帧作为一个状态')
flags.DEFINE_integer('resized_width', 84, '画面宽度')
flags.DEFINE_integer('resized_height', 84, '高度')
flags.DEFINE_float('learning_rate', 0.0001, '初始化学习率')
flags.DEFINE_integer('num_eval_episodes', 3, '测试游戏的回合数')
flags.DEFINE_string('experiment', 'dqn_breakout', '当前训练结果文件名')
flags.DEFINE_string('checkpoint_path', 'checkpoints/dqn_breakout.ckpt', '模型路径, 默认模型为5线程训练了5000万回合')
flags.DEFINE_string('eval_dir', '/tmp/', '评估结果目录')

FLAGS = flags.FLAGS

def get_num_actions():
    env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-v0" or FLAGS.game == "Breakout-v0"):
        num_actions = 3
    return num_actions

def build_graph(num_actions):
    s, q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="q-network")

    network_params = q_network.trainable_weights
    q_values = q_network(s)

    st, target_q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="target-network")
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

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

def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print("Restored model weights from ", FLAGS.checkpoint_path)
    monitor_env = gym.make(FLAGS.game)
    wrappers_dir = FLAGS.eval_dir+"/"+FLAGS.experiment+"/eval"
    if os.path.exists(wrappers_dir) is True:
        shutil.rmtree(wrappers_dir)
    gym.wrappers.Monitor(monitor_env, wrappers_dir)

    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    env = AtariEnvironment(gym_env=monitor_env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    for i_episode in range(FLAGS.num_eval_episodes):
        print('第 %d 回合' % i_episode)
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session = session, feed_dict = {s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, _ = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print('最终得分:', ep_reward)
    monitor_env.close()

def main(_):
    g = tf.Graph()
    session = tf.Session(graph=g)
    with g.as_default(), session.as_default():
        K.set_session(session)
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver()
        evaluation(session, graph_ops, saver)

if __name__ == "__main__":
  tf.app.run()
