"""
NEAT 测试脚本, 挑选出最好的个体进行可视化测试
Contributor：huangsihan
Reviewer：xionglongfei
"""

import gym
import neat
import visualize
import numpy as np

from train import eval_genomes

GAME = 'CartPole-v0'
env = gym.make(GAME).unwrapped

def evaluation():
    """"""
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    winner = p.run(eval_genomes, 1)

    # 显示网络
    node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    visualize.draw_net(p.config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(net.activate(s))
            s, _, done, _ = env.step(a)
            if done: break

if __name__ == '__main__':
    evaluation()
