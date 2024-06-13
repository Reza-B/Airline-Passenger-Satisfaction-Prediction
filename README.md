# Airline Passenger Satisfaction Prediction Project -- Decision Tree and Neural Network

## Project Description

The project aims to analyze and predict passenger satisfaction with their airline travel experience using machine learning techniques. It utilizes a dataset containing various attributes related to passengers' demographics, travel preferences, flight details, and ratings for different services provided during the flight. By leveraging this dataset, the project seeks to understand the factors that influence passenger satisfaction and develop predictive models to forecast whether a passenger will be satisfied or dissatisfied based on their characteristics and flight-related factors.

The project involves several steps, including data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, training, and evaluation. Initially, decision tree algorithms, possibly with Gini index or entropy as splitting criteria, are employed to build predictive models. These models are evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their effectiveness in predicting passenger satisfaction.

## Imports

```python
import pandas as pd
import numpy as np
import random
```

## Clean Dataset

### Airplane

The CSV file contains detailed information about airline passengers' travel experiences and satisfaction levels. Each row in the CSV file represents a single passenger's feedback, while the columns represent different attributes and ratings associated with their travel experience. Key attributes include passenger demographics (e.g., age), travel details (e.g., flight distance), and ratings for various services provided during the flight (e.g., cleanliness). Additionally, the file includes columns for departure and arrival delays and the overall satisfaction level of passengers.

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
data.loc[data["Arrival Delay in Minutes"] > 5, "Arrival Delay in Minutes"] = 1
```

- -- Age

```python
data.loc[data["Age"] <= 20, "Age"] = 0
data.loc[(data["Age"] > 20) & (data["Age"] <= 39), "Age"] = 1
data.loc[(data["Age"] > 39) & (data["Age"] <= 60), "Age"] = 2
data.loc[data["Age"] > 60, "Age"] = 3
```

- -- Cleanliness

```python
data.loc[data["Cleanliness"] < 3, "Cleanliness"] = 0
data.loc[data["Cleanliness"] == 3, "Cleanliness"] = 1
data.loc[data["Cleanliness"] > 3, "Cleanliness"] = 2
```

- -- Flight Distance

```python
data.loc[data["Flight Distance"] <= 1000, "Flight Distance"] = 0
data.loc[(data["Flight Distance"] > 1000) & (data["Flight Distance"] <= 2000), "Flight Distance"] = 1
data.loc[(data["Flight Distance"] > 2000) & (data["Flight Distance"] <= 3000), "Flight Distance"] = 2
data.loc[data["Flight Distance"] > 3000, "Flight Distance"] = 3
```

- -- Departure Delay in Minutes

```python
data.loc[data["Departure Delay in Minutes"] <= 5, "Departure Delay in Minutes"] = 0
data.loc[(data["Departure Delay in Minutes"] > 5) & (data["Departure Delay in Minutes"] <= 25), "Departure Delay in Minutes"] = 1
data.loc[data["Departure Delay in Minutes"] > 25, "Departure Delay in Minutes"] = 2
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

## Tree Node

```python
class TreeNode:
    def __init__(self, feature=None, value=None, subtrees=None):
        self.feature = feature
        self.value = value
        self.subtrees = subtrees if subtrees is not None else {}
```

## Gini index / Entropy

**Gini index**

```python
# Function to calculate Gini index
def calculate_gini(labels):
    total_samples = len(labels)
    if total_samples == 0:
        return 0
    
    gini = 1
    unique_labels, counts = np.unique(labels, return_counts=True)
    for count in counts:
        probability = count / total_samples
        gini -= probability ** 2
    
    return gini

# Function to select the feature with the best Gini index
def best_feature_gini(df, target_name):
    best_feature = None
    min_gini = float('inf')
    
    for feature in df.columns:
        if feature == target_name:
            continue
        
        unique_values = df[feature].unique()
        weighted_gini = 0
        
        for value in unique_values:
            subset = df[df[feature] == value]
            subset_size = len(subset)
            subset_gini = calculate_gini(subset[target_name])
            weighted_gini += (subset_size / len(df)) * subset_gini
        
        if weighted_gini < min_gini:
            min_gini = weighted_gini
            best_feature = feature
    
    return best_feature
```

**Entropy**

```python
# Function to calculate entropy
def calculate_entropy(labels):
    total_samples = len(labels)
    if total_samples == 0:
        return 0
    
    entropy = 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    for count in counts:
        probability = count / total_samples
        entropy -= probability * np.log2(probability)
    
    return entropy

# Function to select the feature with the best entropy
def best_feature_entropy(df, target_name):
    best_feature = None
    min_entropy = float('inf')
    
    for feature in df.columns:
        if feature == target_name:
            continue
        
        unique_values = df[feature].unique()
        weighted_entropy = 0
        
        for value in unique_values:
            subset = df[df[feature] == value]
            subset_size = len(subset)
            subset_entropy = calculate_entropy(subset[target_name])
            weighted_entropy += (subset_size / len(df)) * subset_entropy
        
        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            best_feature = feature
    
    return best_feature
```

## Build Decision Tree

**Gini index**

```python
# Function to build a decision tree using Gini index
def build_tree_gini(df, target_name):
    if len(df[target_name].unique()) == 1:
        return TreeNode(value=df[target_name].iloc[0])
    
    if len(df.columns) == 1:
        return TreeNode(value=df[target_name].mode()[0])
    
    best_feature = best_feature_gini(df, target_name)
    current_node = TreeNode(feature=best_feature)
    
    unique_values = df[best_feature].unique()
    for value in unique_values:
        subset = df[df[best_feature] == value]
        subtree = build_tree_gini(subset, target_name)
        current_node.subtrees[value] = subtree
    
    return current_node
```

**Entropy**

```python
# Function to build a decision tree using entropy
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

## Predict

```python
# Function to predict using the decision tree
def predict(tree, data_point):
    if tree.value is not None:
        return tree.value
    
    feature = tree.feature
    feature_value = data_point[feature]
    
    if feature_value in tree.subtrees:
        return predict(tree.subtrees[feature_value], data_point)
    else:
        return 0
```

## Run & Accuracy

```python
# Specify the target column
target_name = 'satisfaction'

# Build decision trees using Gini index and entropy
decision_tree_gini = build_tree_gini(data, target_name)
print("Gini Decision Tree Built!")

decision_tree_entropy = build_tree_entropy(data, target_name)
print("Entropy Decision Tree Built!")

# Predicting all test data points and calculating accuracy
predictions_gini = test.apply(lambda row: predict(decision_tree_gini, row), axis=1)
predictions_entropy = test.apply(lambda row: predict(decision_tree_entropy, row), axis=1)

# Calculating accuracy for both Gini and Entropy
accuracy_gini = (predictions_gini == test[target_name]).mean() * 100
accuracy_entropy = (predictions_entropy == test[target_name]).mean() * 100

print(f"Gini Accuracy: {accuracy_gini:.2f}%")
print(f"Entropy Accuracy: {accuracy_entropy:.2f}%")
```

## Print Tree

```python
# Function to print the decision tree (recursive)
def print_tree(tree, depth=0, parent_feature=None, value=None):
    if tree is None:
        return

    if tree.feature is not None:
        if parent_feature is not None:
            print("|    " * (depth - 1) + f"{value} -->|-{tree.feature}")
        else:
            print(f"{tree.feature}")
        for value, subtree in tree.subtrees.items():
            print_tree(subtree, depth + 1, f"{tree.feature}", value=value)

    else:
        print("|    " * (depth - 1), int(value), "-->", tree.value)

# Example of printing the decision tree (uncomment to display)
# print_tree(decision_tree_gini)
```

# Neural Network

# Imports

```python
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
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
data
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

# Neural Network Implementation

```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Extracting the target variable 'satisfaction' and the feature variables
y = data['satisfaction']
x = data.drop(["satisfaction"], axis=1)

# Splitting the data into training and testing sets with a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Normalizing the feature data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building and training the MLPRegressor model
model = MLPRegressor(hidden_layer_sizes=(100, 50),
                     max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluating the model performance on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

```

### Explanation

#### Data Preparation

1. **Target and Features Extraction**: The target variable `satisfaction` is extracted from the dataset `data`, while the feature variables are obtained by dropping the `satisfaction` column from `data`.

2. **Train-Test Split**: The dataset is split into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets using `train_test_split` from `sklearn.model_selection`. Here, 80% of the data is used for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). The `random_state` parameter ensures reproducibility of results.

#### Data Normalization

3. **Normalization**: Standardization of the feature data is performed using `StandardScaler` from `sklearn.preprocessing`. This step ensures that all features are on the same scale, which is important for the neural network model to converge efficiently during training.

#### Model Building and Training

4. **MLPRegressor Model**: An MLPRegressor model is initialized with two hidden layers containing 100 and 50 neurons respectively (`hidden_layer_sizes=(100, 50)`). `max_iter=500` specifies the maximum number of iterations for training, and `random_state=42` ensures reproducibility.

5. **Model Training**: The model is trained on the scaled training data (`X_train_scaled`, `y_train`) using the `fit` method.

#### Model Evaluation

6. **Evaluation Metrics**: After training, the model predicts the satisfaction scores (`y_pred`) for the test set (`X_test_scaled`). The Mean Squared Error (MSE) is computed between the actual satisfaction scores (`y_test`) and the predicted scores (`y_pred`) using `mean_squared_error` from `sklearn.metrics`. Lower MSE values indicate better model performance.

#### Conclusion

This script demonstrates the process of building an MLPRegressor model for predicting passenger satisfaction based on airline travel data. It covers data preparation steps such as feature extraction, train-test split, and normalization, followed by model training and evaluation. The MSE provides a quantitative measure of how well the model predicts satisfaction scores on unseen data, aiding in assessing its effectiveness and guiding potential model improvements.

```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Extracting the target variable 'satisfaction' and the feature variables
y = data['satisfaction']
x = data.drop(["satisfaction"], axis=1)

# Splitting the data into training and testing sets with a test size of 20%
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Normalizing the feature data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building and training the MLPRegressor model
model = MLPRegressor(hidden_layer_sizes=(100, 50),
                     max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluating the model performance on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

```

#### Data Preparation

1. **Target and Features Extraction**: The target variable `satisfaction` is extracted from the dataset `data`, while the feature variables are obtained by dropping the `satisfaction` column from `data`.

2. **Train-Test Split**: The dataset is split into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets using `train_test_split` from `sklearn.model_selection`. Here, 80% of the data is used for training (`X_train`, `y_train`) and 20% for testing (`X_test`, `y_test`). The `random_state` parameter ensures reproducibility of results.

#### Data Normalization

3. **Normalization**: Standardization of the feature data is performed using `StandardScaler` from `sklearn.preprocessing`. This step ensures that all features are on the same scale, which is important for the neural network model to converge efficiently during training.

#### Model Building and Training

4. **MLPRegressor Model**: An MLPRegressor model is initialized with two hidden layers containing 100 and 50 neurons respectively (`hidden_layer_sizes=(100, 50)`). `max_iter=500` specifies the maximum number of iterations for training, and `random_state=42` ensures reproducibility.

5. **Model Training**: The model is trained on the scaled training data (`X_train_scaled`, `y_train`) using the `fit` method.

#### Model Evaluation

6. **Evaluation Metrics**: After training, the model predicts the satisfaction scores (`y_pred`) for the test set (`X_test_scaled`). The Mean Squared Error (MSE) is computed between the actual satisfaction scores (`y_test`) and the predicted scores (`y_pred`) using `mean_squared_error` from `sklearn.metrics`. Lower MSE values indicate better model performance.

#### Conclusion

This script demonstrates the process of building an MLPRegressor model for predicting passenger satisfaction based on airline travel data. It covers data preparation steps such as feature extraction, train-test split, and normalization, followed by model training and evaluation. The MSE provides a quantitative measure of how well the model predicts satisfaction scores on unseen data, aiding in assessing its effectiveness and guiding potential model improvements.

```python
from sklearn.metrics import accuracy_score

def compare_y_test_and_y_pred(y_test, y_pred, threshold=0.5):
    # Convert predicted values to binary classification
    y_pred_class = (y_pred > threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Example usage
compare_y_test_and_y_pred(y_test, y_pred)
```

### Explanation

#### Function `compare_y_test_and_y_pred`

1. **Purpose**: This function compares the predicted values (`y_pred`) with the actual values (`y_test`) and calculates the accuracy of a binary classification task based on a specified threshold.

2. **Parameters**:
   - `y_test`: The actual target values from the test set.
   - `y_pred`: The predicted target values from the model.
   - `threshold`: (Optional) Threshold value used to convert predicted probabilities to binary predictions. Defaults to 0.5.

3. **Steps**:
   - **Convert Predictions**: Predicted values (`y_pred`) are converted into binary class labels (`y_pred_class`) by comparing each prediction against `threshold` and converting values greater than `threshold` to 1 and less than or equal to `threshold` to 0.

   - **Calculate Accuracy**: Using `accuracy_score` from `sklearn.metrics`, the accuracy of the binary classification is computed by comparing `y_test` (actual values) with `y_pred_class` (predicted values).

4. **Output**: The function prints the calculated accuracy as a percentage with two decimal places.

#### Example Usage

- `compare_y_test_and_y_pred(y_test, y_pred)`: This function call compares the predicted satisfaction scores (`y_pred`) with the actual satisfaction scores (`y_test`) using a default threshold of 0.5. It then prints the accuracy of the binary classification task based on these predictions.

This function is useful for evaluating the performance of models that predict binary outcomes, such as predicting passenger satisfaction (satisfied or dissatisfied) in this project. Adjusting the `threshold` parameter allows for exploring different trade-offs between sensitivity and specificity in the classification predictions.
