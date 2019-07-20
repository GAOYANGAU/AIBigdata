#!/usr/bin/env python3

"""
游戏环境文件: 基于 gym atari 构建打砖块游戏环境
Contributor：huangsihan
Reviewer：xionglongfei
"""


from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque

class AtariEnvironment(object):
    """
    封装雅达利(atari)游戏环境
    构建 resized_width 宽, resized_height 高的游戏画面, 并且每 agent_history_length 帧确定一个环境
    """
    def __init__(self, gym_env, resized_width, resized_height, agent_history_length):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length

        self.gym_actions = range(gym_env.action_space.n)
        if (gym_env.spec.id == "Pong-v0" or gym_env.spec.id == "Breakout-v0"):
            print("Doing workaround for pong or breakout")
            # 打砖块的 actor 的操作有三种选择, 分别是左, 右, 无操作
            self.gym_actions = [1,2,3]

        # 使用缓冲区
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        重置游戏环境
        """
        # 清理缓冲区
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)

        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        图像预处理
        1) 获取图像灰度
        2) 重设图像大小
        """
        return resize(rgb2gray(observation), (self.resized_width, self.resized_height), mode='constant')

    def step(self, action_index):
        """
        在当前的环境下, 做出一个动作
        构建当前状态 (连接 agent_history_length-1 的前一帧和后一帧), 并返回当前状态
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # 移除旧帧图像, 加入当前帧
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info
