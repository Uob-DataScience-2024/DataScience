# Chapter 2: Data Preparation

Explain the necessity of data preprocessing here.

## 2.1 Data Acquisition and Interpretation

Please state the source of our dataset here, along with a brief introduction to the dataset (e.g. which datasets it includes: PFF scouting data, etc.)

## 2.2 Data Processing

Here, explain the concept and necessity of data processing in data science. Then state what steps need to be carried out in our project(2.2.1, 2.2.2 's title).

### 2.2.1 Normalization

State here that the normalisation method we used is min-max normalisation.

Can refer to: An explanation of min-max normalisation:

Min-Max normalization, also known as feature scaling, is a data preprocessing technique used to transform features to scale to a specific range, typically between 0 and 1. This method is particularly effective in ensuring that different features have equal importance during model training, especially in algorithms that involve distance measurements, such as K-Nearest Neighbors (KNN) and Principal Component Analysis (PCA).

The formula for Min-Max normalization is:

$$
X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
$$

Here, \(X\) represents the original data point, and \(X_{\text{min}}\) and \(X_{\text{max}}\) are the minimum and maximum values in the data, respectively. By applying this formula, the original data \(X\) is linearly transformed to the range \([0, 1]\).

For example, if you have a feature with original values [10, 20, 30], after applying Min-Max normalization, these values would transform to \([0, 0.5, 1]\).

The advantage of this method is its simplicity and quick adjustment of data to a specified range. However, it has drawbacks, such as being highly sensitive to outliers. If there are outliers in the data, they can significantly influence \(X_{\text{min}}\) and \(X_{\text{max}}\), leading to a skewed distribution of the other normalized values within the range.

### 2.2.2 Datasets Merging

Details:
Due to the dataset being divided into several data files and types, with some IDs not being entirely unique, direct merging was not feasible. This merging method necessitated studying the dataset to derive the following identifier-related conclusions:

- gameId is the unique identifier for each game.
- nflId is the unique identifier for each player.
- playId is not unique across games in the tracking data.
- In the tracking data for each game, nflId+playId cannot serve as a unique identifier.
- In the scouting data for each game, nflId+playId can serve as a unique identifier and will not be repeated.
- When filtering for the same playId+gameId in the tracking data for a single game, it was observed that frameId is a continuous number, leading to the inference that playId may represent a stage within the game, as the corresponding timestamp span is only around 3.821s (a mean value, with a maximum of 8s and a minimum of 2.6s).
- Therefore, based on the above facts, it can be concluded that the combined ID consisting of nflId+playId from the scouting data for each game can correspond to a time segment within that game, with an average duration of only 3.8 seconds, which is reasonable and usable as a data merging approach.
- Initial exploration revealed that the unique playId lists in plays.csv and the tracking data for a single game are identical, implying that the data can be associated.

After exploring the data merging approach, we could implement a GameNFLData class that merges the base data on top of the existing base data classes (GamePffData, GameTrackingData, GamePlayData). Furthermore, the next generation of data loading code introduced a more convenient merging method: This version of data merging is faster and simpler. MergeNormData accepts the already loaded TrackingNormData, PffNormData, PlayNormData, GameNormData, and PlayerNormData as input parameters, automatically merging the three most important datasets: tracking, pff, and play. The remaining two datasets only need to be merged when necessary, controlled by the player_needed=False and game_needed=False parameters in the generate_dataset function of the DataGenerator.

## 2.3 Structured Data Loading

Details: 
The project employed object-oriented programming (OOP) to implement structured loading for the entire NFL dataset. This loading approach offers the benefit of clear organization and, in one version, the loaded dataset objects themselves were designed to be iterable, greatly simplifying the complexity of data retrieval in subsequent development processes.

The three main data components, namely PffData, PlayData, and TrackingData, correspond to the pffScoutingData.csv, plays.csv, and week{number}.csv files, respectively. For each of these components, corresponding classes were implemented: Game{name} for the dataset of each game and {name}Item for each data item. The Game{name} classes are iterable objects that support the __getitem__ method, enabling easy access to individual data items.

Furthermore, based on the uniqueness of the combination of gameId and playId, the three datasets were joined together to form a new dataset object named NFLData. This integration allows for seamless querying and retrieval of data across the different components.

The utilization of this structured data loading approach is highly convenient. By calling the .load method of the corresponding class, a dictionary-like object keyed by gameId is returned, containing Game{name}Data objects for each game. For instance, games = GameNFLData.loads(['xxx']) would load the specified games. Subsequently, accessing data for any gameId at any index becomes trivial.

This structured and object-oriented approach to data loading greatly enhances the friendliness and ease of use for subsequent development tasks, enabling developers to focus on higher-level analysis and processing of the NFL dataset.

## 2.4 Reusable Data Loader: Automated Data Preprocessing

Details:
There are two versions of this Data Loader in this project, corresponding to the initial way we processed data loading and the more advanced version we developed after gaining a deeper understanding of the data.:

1. Developer-oriented structured dataset objects, each supporting direct invocation of the .tensor() function to obtain preprocessed data ready for use in neural networks. By providing the names of columns as input features and the target features in the object members, the corresponding tensor can be obtained. The automatic preprocessing technique generates label tables for categories by analyzing the entire dataset, then converts them into index encodings acceptable as input to neural networks. Numeric data is automatically normalized, while temporal data is converted to timestamps. The function also supports overriding label tables: if the resize_range_overwrite and category_labels_overwrite parameters contain override configurations for specific features, the function directly uses these overrides instead of the automatically generated statistical functionality, resulting in a highly extensible design.

2. A version that loads data in a clear, simple, and easy-to-understand manner, storing the dataframe as an attribute within the object. In this version, the automatic preprocessing and normalization functionality becomes more concise and comprehensible, significantly enhancing readability and generalizability without sacrificing usability. The loaded TrackingNormData, PffNormData, PlayNormData, GameNormData, PlayerNormData, and MergeNormData objects are passed as initialization parameters to the DataGenerator. Data filtering can be performed directly on these classes before passing them to the generator or on the data generated by the DataGenerator. The generator supports generating tensors through the SimpleDataset class, which inherits from torch.utils.data.Dataset, and can also return NumPy arrays or dataframes. The generate_dataset function of the generator requires x_columns and y_column as inputs and, similar to the previous version, automatically handles label and numeric data. Whether to perform normalization on numeric data can be determined by the norm flag. This version provides a more flexible customization method for data preprocessing: directly using the data_type_mapping parameter, which is a dictionary where the key is the column name and the value is a function, equivalent to executing a map operation in Python, giving users more customization options.

# Data Exploration

To be updated in few hours....

# Data Modelling

To be updated in few hours....






