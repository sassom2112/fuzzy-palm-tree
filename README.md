# Building and Training a Single Neuron Classifier for Binary Classification: 
## Manual vs. Library-Based Approaches

### Problem Introduction
In this project, we're solving a binary classification problem to predict the color of wine (red or white) based on 11 chemical features such as acidity, residual sugar, alcohol content, and more. The target label is binary, where:

+ 1 represents red wine
+ 0 represents white wine

The dataset includes two subsets (red and white wines), and the goal is to build a model that can accurately classify the wine color based on these features.

### The project compares two approaches:

+ Manual Implementation of the classification model, including data splitting, standardization, and training.
+ Library-based Implementation using popular machine learning tools like scikit-learn.

By comparing these two approaches, we aim to understand:
+ How much control we gain by manually implementing machine learning steps.
+ How using external libraries like scikit-learn can streamline the development process and help avoid potential errors or inefficiencies.

### Manual Implementation
In this approach, we:
+ Manually split the dataset into training and test sets using numpy.
+ Manually standardize the features to prevent numerical instability during training.
+ Implemented a Single Neuron Classifier using logistic regression to predict wine color. We use the binary cross-entropy loss and gradient descent to train the model.

This approach demonstrates how to manually perform key tasks of machine learning, which offers more control and deeper insights into each step of the process. However, it is more error-prone and less efficient compared to using well-tested libraries.

### Library-based Implementation (scikit-learn)
In this implementation, we:

+ Use scikit-learn's train_test_split to handle data splitting, shuffling, and separation of training and test data.
+ Use StandardScaler to standardize the features consistently.
+ Use logistic regression from scikit-learn to build a model, automatically optimizing and performing prediction using simple, well-optimized functions.

This approach shows the convenience and reliability of using external libraries like scikit-learn, which reduce the likelihood of errors and significantly speed up the workflow.

## Instructions to Run Each Implementation

To run both implementations, you'll need to set up the following dependencies:

+ Install Python (if not already installed): Python 3.x is required.
+ Install the necessary libraries. You can install these by running:
```bash
pip install numpy pandas matplotlib
```

If you're running the library-based implementation, you also need scikit-learn:
```bash
pip install scikit-learn
```

### Running the Manual Implementation
Download the dataset (winequality-red.csv and winequality-white.csv) in `./data/` directory. Each notebook has a upload snippet where you can upload your code directly on youre notebook as long as the dataset is downloaded loacally:
```python
from google.colab import files
uploaded = files.upload()
```

Open the Jupyter notebook (manual_wine_classification.ipynb).

Run the notebook cells in order. This notebook manually splits the dataset, standardizes the data, and trains a logistic regression model without the use of scikit-learn.

The notebook will output the loss during training and the final test accuracy.

### Running the Library-based Implementation

Open the Jupyter notebook (sklearn_wine_classification.ipynb).

This notebook uses scikit-learn's built-in tools to split the dataset, standardize the data, and train a logistic regression model.

The notebook will display the model's performance, including the final test accuracy.
