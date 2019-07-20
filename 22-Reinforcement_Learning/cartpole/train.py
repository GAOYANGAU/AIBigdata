"""
使用 NEAT 进行训练, 使 cartpole 游戏中的木杆不倒下
NEAT论文见: http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
Contributor：huangsihan
Reviewer：xionglongfei
"""

import neat
import numpy as np
import gym
import visualize

env = gym.make('CartPole-v0').unwrapped

# NEAT 网络配置文件, 具体细节不需要了解, 直接使用
CONFIG = "./config"
EP_STEP = 300
GENERATION_EP = 10

def eval_genomes(genomes, config):
    """
    对每一个 genome 的 net 测试 GENERATION_EP 个回合
    最后挑选所有回合中总 reward 最少回合当成这个 net 的 fitness
    """
    # 对每个 genome 进行测试
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ep_r = []
        # 开始 GENERATION_EP 回合的游戏
        for _ in range(GENERATION_EP):
            accumulative_r = 0.
            observation = env.reset()
            # 总共运行 EP_STEP 步
            for _ in range(EP_STEP):
                # 输入值为 [In0, In1, In2, In3], 表示一个当前状态
                action_values = net.activate(observation)
                action = np.argmax(action_values)
                # 做出动作, 进入下一个状态
                observation_, reward, done, _ = env.step(action)
                accumulative_r += reward
                if done:
                    break
                observation = observation_
            ep_r.append(accumulative_r)
        # 在所有环境中选择最小的分数
        genome.fitness = np.min(ep_r)/float(EP_STEP)

def run():
    """功能函数"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # 每 10 步保存一个文件
    pop.add_reporter(neat.Checkpointer(10))

    # 训练 10 代
    pop.run(eval_genomes, 10)

    # 在图表中展示训练
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    run()
