# [English](#English)-[Chinese](#中文) Bilingual Log

# English
---

# Experiment log:

## Experimemt: Implementing Exploratory Work on PFF Block Type Data

- Experiment Summary: Develop a classifier model to carry out predictions on PFF block type data
- Experiment Results:

---

# 中文

# 实验日志:

## 实验: 实施对pff block type数据的探索工作

- 实验概要: 构建一个分类器模型，实施对pff block type数据的预测
- 实验结果: 成功
- 实验结论:
    - 标识符相关:
        - gameid是每场比赛的独立标识符
        - nflid是球员的独立标识符
        - playid在追踪数据中的每场比赛并不唯一
        - *在每场比赛的追踪数据中，nflid+playid`不能`作为唯一标识符*
        - *在每场比赛的球探数据中，nflid+playid`可以`作为唯一标识符，不会在一场比赛中重复*
        - 当在追踪数据中同一场中筛选相同的playid+gameid时，会发现frameId是一个连续的数字，推断：playId可能代表游戏中的一个阶段，因为对应的时间戳的跨度只有`3.821s`左右（这是一个均值, 最大`8s`，最小`2.6s`）
        - **所以，通过上述的事实，可以得出结论：每场比赛中的球探数据的`nflid+playid`组成的联合id，可以对应每场比赛中的一个时间段，这个时间段只有平均`3.8`秒，这是合理的，且可以使用的数据联合方式**
- 实验分析: 实验实施了基于两个LSTM形成的编解码器解构，实现序列到序列的神经网络。经过实验，初步全局准确率可以达到75%（实施的是序列到序列的分类任务），神经网络有待优化。
