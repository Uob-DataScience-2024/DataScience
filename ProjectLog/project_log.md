# [English](#English)-[Chinese](#中文) Bilingual Log

# English
---

## 2024/03/21

### Implementation Content: Structured Dataset Object Implementation

#### Details:
- The three main `names` `PffData, PlayData, TrackingData` correspond to three types of dataset files respectively: `pffScoutingData.csv, plays.csv, week{number}.csv`.
- Correspondingly, the `Game{name}` class as the dataset for each game and the `{name}Item` class as the object for each piece of data have been implemented. The `Game{name}` class is an iterable object, supporting the `__getitem__` method.
- Moreover, based on these three datasets, by the uniqueness of `gameId` + `playId`, the three tables have been linked together, forming a new dataset object named `NFLData`.

---

## 2024/03/21

### Implementation Content: Enabling Datasets to Support Direct `to_tensor` Method

#### Details:
- To better use structured datasets in neural network dataset objects, a `to_tensor` method that can complete preprocessing directly from the dataset has been implemented.
- The `to_tensor` method allows for the specification of fixed labels through parameters. If no corresponding label data is provided, it can automatically generate a label array and will include the mapping of labels in the returned data. For numerical data, it will automatically map them to a range of 0-1, with the pre-mapping maximum and minimum values also included in the returned data_map.
- Additionally, the `to_tensor` method also automatically performs a series of preprocessing operations, including handling NA values.

---

## 2024/03/21

### Implementation Content: Implementation of Neural Network Model Construction

#### Details:
- Implemented a general LSTM GRU sequence labeling network.
- Implemented training config.
- Supports any input and output label, for instance, one can modify `input_feature` and `output_feature` in the configuration to select any data from NFLData as input and output, achieving very flexible experimental functionality. This makes it much easier to conduct controlled variable experiments in batches.
- Constructed a general experimental configuration example.
- Constructed the implementation of the training process.

---

## 2024/03/28

### Implementation Content: Preliminary Implementation of Data Visualization

#### Details:
- Implemented the basic function of data visualization, capable of playing tracking data.
- Implemented the function to generate video from visualized data.

---

# 项目日志

---

## 2024/03/21

### 实施内容: 结构化的数据集对象实现

#### 细节:
- `PffData, PlayData, TrackingData` 这三个主要`name`的数据分别对应 `pffScoutingData.csv, plays.csv, week{number}.csv` 这三种数据集文件
- 对应的实现了对应的 `Game{name}` 类作为每一场game对应的数据集 和 `{name}Item` 类作为每一条数据的对象。`Game{name}` 类是可以遍历的对象，支持`__getitem__`方法
- 同时，基于这三个数据，根据`gameId`+`playId`的唯一性，将三个数据表连接到了一起，组成了一个新的 `name` 名为 `NFLData` 的数据集对象

---

## 2024/03/21

### 实施内容: 使数据集支持直接`to_tensor`方法

#### 细节:
- 为了使得神经网络数据集对象更好的使用结构化的数据集，需要实现这么一个可以直接从数据集中完成预处理的`to_tensor`方法
- `to_tensor`方法通过各项参数，可以实现对输入数据集的label固定，如果没有对应的label数据覆盖的话，可以实现自动的label数组，并且会在返回的数据中包含label的映射。对于数字数据，则会自动将其映射到0-1，同时映射前的最大最小值也会包含在返回的data_map中。
- 同时，`to_tensor`方法还会自动进行na处理等一系列预处理操作

---

## 2024/03/21

### 实施内容: 实施神经网络模型的构建

#### 细节:
- 实现了通用LSTM GRU序列标记网络
- 实现了训练config
- 支持任意输入输出label，比如说可以在配置中修改`input_feature`和`output_feature`，来选择NFLData中的任意数据作为输入输出，实现了非常自由化的实验功能。使得后续批量进行控制变量法实验变得非常容易。
- 构建了通用化的实验配置例子
- 构造了训练过程的实现

---

## 2024/03/28

### 实施内容: 数据可视化初步实现

#### 细节:
- 实现了数据可视化的基本功能，可以播放tracking data的数据
- 实现了可视化数据生成mp4的功能

---

## 2024/04/08

### 实施内容: 完善可视化脚本

#### 细节:
- visualization.py
