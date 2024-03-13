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
 - 实验结论:
   - 标识符相关:
     - gameid是每场比赛的独立标识符
     - nflid是球员的独立标识符
     - playid在追踪数据中的每场比赛并不唯一
     - *在每场比赛的追踪数据中，nflid+playid`不能`作为唯一标识符*
     - *在每场比赛的球探数据中，gameid+playid`可以`作为唯一标识符，不会在一场比赛中重复*



