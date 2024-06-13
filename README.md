# Airline Passenger Satisfaction Prediction Project with Decision-Tree

# Project Description

  The project aims to analyze and predict passenger satisfaction with their airline travel experience using machine learning techniques. It utilizes a dataset containing various attributes related to passengers' demographics, travel preferences, flight details, and ratings for different services provided during the flight. By leveraging this dataset, the project seeks to understand the factors that influence passenger satisfaction and develop predictive models to forecast whether a passenger will be satisfied or dissatisfied based on their characteristics and flight-related factors.

  The project likely involves several steps, including data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, training, and evaluation. Machine learning algorithms such as decision trees, possibly with Gini index or entropy as splitting criteria, are employed to build predictive models. These models are then evaluated using performance metrics such as accuracy, precision, recall, and F1-score to assess their effectiveness in predicting passenger satisfaction.

# Imports

```python
import pandas as pd
import numpy as np
import random
```

# Clean Dataset

## Airplane

The CSV file contains detailed information about airline passengers' travel experiences and satisfaction levels. Each row in the CSV file represents a single passenger's feedback, while the columns represent different attributes and ratings associated with their travel experience. Key attributes include passenger demographics (e.g., gender, age), travel details (e.g., flight distance, type of travel), and ratings for various services provided during the flight (e.g., seat comfort, inflight entertainment). Additionally, the file includes columns for departure and arrival delays and the overall satisfaction level of passengers.

The dataset serves as the primary source of information for the project, providing valuable insights into passenger preferences, behaviors, and satisfaction levels. It is used for exploratory data analysis, feature engineering, model training, and evaluation to develop effective predictive models for understanding and predicting passenger satisfaction with airline travel.

- Read csv file

```python
data = pd.read_csv("csv/Airplane.csv") 
```

- Dropping missing data

```python
data = data.dropna()
```

- Converting non-numeric values to numeric

```python
data["satisfaction"] = data["satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1}).astype(int)
data["Customer Type"] = data["Customer Type"].map({"disloyal Customer": 0, "Loyal Customer": 1}).astype(int)
data["Type of Travel"] = data["Type of Travel"].map({"Personal Travel": 0, "Business travel": 1}).astype(int)
data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1}).astype(int)
data["Class"] = data["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2}).astype(int)
```

- Categorizing continuous data

- -- Arrival Delay in Minutes

```python
data.loc[data["Arrival Delay in Minutes"] <= 5, "Arrival Delay in Minutes"] = 0
data.loc[(data["Arrival Delay in Minutes"] > 5), "Arrival Delay in Minutes"] = 1
```

- -- Age

```python
data.loc[data["Age"] <= 20, "Age"] = 0
data.loc[(data["Age"] > 20) & (data["Age"] <= 39), "Age"] = 1
data.loc[(data["Age"] > 39) & (data["Age"] <= 60), "Age"] = 2
data.loc[(data["Age"] > 60), "Age"] = 3
```

- -- Cleanliness

```python
data.loc[data["Cleanliness"] < 3, "Cleanliness"] = 0
data.loc[data["Cleanliness"] == 3, "Cleanliness"] = 1
data.loc[(data["Cleanliness"] > 3), "Cleanliness"] = 2
```

- -- Flight Distance

```python
data.loc[data["Flight Distance"] <= 1000, "Flight Distance"] = 0
data.loc[(data["Flight Distance"] > 1000) & (data["Flight Distance"] <= 2000), "Flight Distance"] = 1
data.loc[(data["Flight Distance"] > 2000) & (data["Flight Distance"] <= 3000), "Flight Distance"] = 2
data.loc[(data["Flight Distance"] > 3000), "Flight Distance"] = 3
```

- -- Departure Delay in Minutes

```python
data.loc[data["Departure Delay in Minutes"] <= 5, "Departure Delay in Minutes"] = 0
data.loc[(data["Departure Delay in Minutes"] > 5) & (data["Departure Delay in Minutes"] <= 25), "Departure Delay in Minutes"] = 1
data.loc[(data["Departure Delay in Minutes"] > 25), "Departure Delay in Minutes"] = 2
```

- Selecting the last 10,000 rows as test data

```python
test = data.tail(10000)
```

- Removing the last 10,000 rows from the data frame

```python
data = data.head(90000)
```

- Separating data for satisfied and neutral or dissatisfied

```python
satisfaction_0 = data[data['satisfaction'] == 0]
satisfaction_1 = data[data['satisfaction'] == 1]
random.seed(43)
```

- Selecting 10,000 random samples from each group

```python
satisfaction_0_random = random.sample(satisfaction_0.index.tolist(), 10000)
satisfaction_1_random = random.sample(satisfaction_1.index.tolist(), 10000)
```

- Combining these two data sets

```python
data = pd.concat([data.loc[satisfaction_0_random], data.loc[satisfaction_1_random]])
```

- Dropping unnecessary columns

```python
data = data.drop(["index", "id", "Gender"], axis=1)
test = test.drop(["index", "id", "Gender"], axis=1)
```

- show clean data

```python
# Data is not shown due to large size
# data
```

```python
print(data[["Age","satisfaction"]].groupby(["Age"],as_index=False).mean());
```

       Age  satisfaction
    0    0      0.245791
    1    1      0.456366
    2    2      0.634953
    3    3      0.253857

Restaurant

```python
# data = pd.read_csv("csv/Restaurant.csv");
# test = data.tail(1);
# data = data.head(10);
```

# Tree Node

```python
class TreeNode:
    def __init__(self, feature=None, value=None, subtrees=None):
        self.feature = feature
        self.value = value # for leaf node
        self.subtrees = subtrees if subtrees is not None else {}
```

# Gini index / Entropy

**Gini index**

```python
# Function to calculate Gini index
def calculate_gini(labels):
    total_samples = len(labels)
    if total_samples is None:
        return 0
    
    gini = 1
    unique_labels, counts = np.unique(labels, return_counts=True)
    for count in counts:
        probability = count / total_samples
        gini -= probability ** 2
    # This part is written based on the mathematical formula of the Gini index
    return gini

# Function to select the feature with the best Gini index
def best_feature_gini(df, target_name):
    best_feature = None
    min_gini = float('inf')
    
    # This loop checks all features present in the input DataFrame
    for feature in df.columns:
        if feature == target_name:
            continue
        
        # All possible values of a feature are stored in this variable
        unique_values = df[feature].unique()
        weighted_gini = 0
        
        # Calculate Gini for all possible values
        for value in unique_values:
            subset = df[df[feature] == value]
            subset_size = len(subset)
            subset_gini = calculate_gini(subset[target_name])
            weighted_gini += (subset_size / len(df)) * subset_gini
        
        # If the calculated Gini is less than the minimum Gini, update the minimum value
        if weighted_gini < min_gini:
            min_gini = weighted_gini
            best_feature = feature
    
    return best_feature

```

**Entropy**

```python
def calculate_entropy(labels):
    total_samples = len(labels)
    if total_samples == 0:
        return 0
    
    entropy = 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    for count in counts:
        probability = count / total_samples
        entropy -= probability * np.log2(probability)
    # This loop implements the entropy formula
    return entropy

# Function to select the best feature using entropy
def best_feature_entropy(df, target_name):
    best_feature = None
    min_entropy = float('inf')
    
    # This loop checks all features present in the input DataFrame
    for feature in df.columns:
        if feature == target_name:
            continue
        
        # All possible values of a feature are stored in this variable
        unique_values = df[feature].unique()
        weighted_entropy = 0
        
        # Calculate entropy for all possible values
        for value in unique_values:
            subset = df[df[feature] == value]
            subset_size = len(subset)
            subset_entropy = calculate_entropy(subset[target_name])
            weighted_entropy += (subset_size / len(df)) * subset_entropy
        
        # If the calculated entropy is less than the minimum entropy, update the minimum value
        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            best_feature = feature
    
    return best_feature
```

# Build Desision Tree

**Gini index**

```python
# Function to build a decision tree using Gini index
def build_tree_gini(df, target_name=None):
    # If all samples have the same target value, create a leaf node with that value
    if len(df[target_name].unique()) == 1:
        return TreeNode(value=df[target_name].iloc[0])
    
    # If only one feature is left (excluding the target), create a leaf node with the most frequent target value
    if len(df.columns) == 1:
        return TreeNode(value=df[target_name].mode()[0])
    
    # Select the best feature using the Gini index
    best_feature = best_feature_gini(df, target_name)
    
    # Create a decision node with the selected feature
    current_node = TreeNode(feature=best_feature)
    
    # Recursively build subtrees for each value of the selected feature
    unique_values = df[best_feature].unique()
    for value in unique_values:
        subset = df[df[best_feature] == value]
        subtree = build_tree_gini(subset, target_name)
        current_node.subtrees[value] = subtree
    
    return current_node

```

**Entropy**

```python
def build_tree_entropy(df, target_name=None):
    # If all samples have the same target value, create a leaf node with that value.
    if len(df[target_name].unique()) == 1:
        return TreeNode(value=df[target_name].iloc[0])
    
    # If only one feature remains (excluding the target), create a leaf node with the most frequent target value.
    if len(df.columns) == 1:
        return TreeNode(value=df[target_name].mode()[0])
    
    # Select the best feature using entropy.
    best_feature = best_feature_entropy(df, target_name)
    
    # Create a decision node with the selected feature.
    current_node = TreeNode(feature=best_feature)
    
    # Recursively build subtrees for each value.
    unique_values = df[best_feature].unique()
    for value in unique_values:
        subset = df[df[best_feature] == value]
        subtree = build_tree_entropy(subset, target_name)
        current_node.subtrees[value] = subtree
    
    return current_node

```

# Predict

```python
def predict(tree, data_point):
    if tree.value is not None:
        return tree.value
    
    # Recursively traverses the tree
    feature = tree.feature
    feature_value = data_point[feature]
    
    if feature_value in tree.subtrees:
        return predict(tree.subtrees[feature_value], data_point)
    else:
        return 0
```

# Run & accuracy

```python
target_name = 'satisfaction'; # Specifying the target column
decision_tree_gini = build_tree_gini(data, target_name) # Calculating the tree with Gini index
print("Gini Done!")
decision_tree_entropy = build_tree_entropy(data, target_name) # Calculating the tree with entropy
print("entropy Done!")
```

    Gini Done!
    entropy Done!

```python
# Predicting all test file and saving it for both Gini and Entropy trees
predictions_gini = test.apply(lambda row: predict(decision_tree_gini, row), axis=1)
predictions_entropy = test.apply(lambda row: predict(decision_tree_entropy, row), axis=1)

# Calculating accuracy for both Gini and Entropy
accuracy_gini = (predictions_gini == test[target_name]).mean()
accuracy_entropy = (predictions_entropy == test[target_name]).mean()

print("Gini Accuracy = ",(accuracy_gini * 100),"%")
print("Entropy Accuracy = ",(accuracy_entropy * 100),"%")
```

    Gini Accuracy =  93.46 %
    Entropy Accuracy =  93.32000000000001 %

# Print Tree

```python
def print_tree(tree, depth=0, parent_feature=None, value = None):
    if tree is None:
        return

    if tree.feature is not None:
        if parent_feature is not None:
            print("|    " * (depth - 1) + f"{value} -->|-{tree.feature}")
        else:
            print(f"{tree.feature}")
        for value, subtree in tree.subtrees.items():
            print_tree(subtree, depth + 1, f"{tree.feature}",value=value)

    else:
        print("|    " * (depth -1),int(value),"-->",tree.value)

# The following line is to display the tree
# print_tree(decision_tree_gini); 
```

    * The tree is not shown because it is too big 

**Finish**
