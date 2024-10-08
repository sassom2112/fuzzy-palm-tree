# Problem Introduction
In this project, we're solving a binary classification problem to predict the color of wine (red or white) based on 11 chemical features such as acidity, residual sugar, alcohol content, and more. The target label is binary, where:

- `1` represents red wine
- `0` represents white wine

The dataset includes two subsets (red and white wines), and the goal is to build a model that can accurately classify the wine color based on these features.

# Approach and Iterations
This project is broken down into three iterations, each representing a step forward in learning and understanding deep learning fundamentals:

## First Attempt: Manual Gradient Descent and Loss Calculation
File: `Single_Neuron_Classifier_One`

In the first iteration, I built a logistic regression classifier from scratch and manually implemented gradient descent for model training.


### What I Learned:
- I manually implemented gradient descent, which gave me a understanding of how updates happen with respect to the weights and bias during training.

- I explored the structure of training loops, loss computation (using squared error), and how parameters are updated in each epoch.

This approach helped me solidify my understanding of the training process in machine learning and provided insight into the limitations of manual implementation.

#### What Could Be Improved:
While implementing the training loop was educational, manual loops for gradient updates were not as efficient as when I introduced libraries in later iterations.
The Squared error loss is not well-suited for binary classification problems; instead, binary cross-entropy (log loss) is a better choice for classification tasks.


## Second Attempt: Binary Cross-Entropy Loss and Manual Train-Test Split
File: `Single_Neuron_Classifier_Two`

In the second iteration, I introduced binary cross-entropy loss to my  classification problem which yielded similar results. I also implemented a manual train-test split for better validation and resulting in a faster runtime.

### What I Learned:
I introduced binary cross-entropy loss into the model, which is the appropriate loss function for binary classification tasks. I learned how to manually split the dataset into training and testing sets, ensuring proper evaluation of the model's performance.

#### What Could Be Improved:
The second iteration give me faster runtimes but resulted in similar predicitons. I realized that moving to a machine learning framework would yield better results and wow did the model improve on the third attempt!!!

## Third Attempt: Scaling and Using train_test_split (Library-based Implementation)
File: `Single_Neuron_Classifier_using_sklearn`

In the final iteration, I adopted the use of the scikit-learn library to handle scaling, data splitting, and logistic regression. This approach was much more efficient.

### What I Learned:
I used scikit-learn's train_test_split to simplify data splitting, ensuring randomization and proper separation of training and testing data.

I incorporated standardization with StandardScaler, which drasticlly improved model convergence during training.

By utilizing scikit-learn's logistic regression, I was able to leverage well-optimized functions to reduce implementation complexity and streamline the workflow.

### Where to next?
- Using mini-batch gradient descent or stochastic gradient descent (SGD) for better performance on larger datasets.
- Further exploration into visualizing more metrics (such as accuracy during training) and comparing training and validation performance would help detect overfitting.
- Explore Deep Learning Frameworks like (PyTorch or TensorFlow) to build more complex models while still understanding the underlying operations.
- Introduce Regularization: To prevent overfitting, I will explore techniques like L2 regularization, dropout, and early stopping.
- Implement multilayer perceptrons (MLPs) to explore the benefits of deeper architectures.
- Hyperparameter Tuning: I plan to experiment more with learning rates, batch sizes, and the number of epochs to optimize model performance.

# Overall Learning
<img src="data/2024-10-08%2012_11_20-Classifier_NLL_Loss.ipynb%20-%20Colab.png" alt="NLL" width="200"/>
<img src="data/2024-10-08 12_08_27-Single Neuron Classifier.ipynb - Colab.png" alt="NLL" width="200"/>
<img src="data/2024-10-08 12_15_38-Single Neuron Classifier using sklearn.ipynb - Colab.png" alt="NLL" width="200"/>

Implementing basic neural networks and training loops from scratch was invaluable for gaining a deep understanding of the underlying mechanics of model learning. However, after refactoring the code for efficiency and incorporating optimized libraries like scikit-learn, I achieved a significantly shorter runtime, reduced the number of required epochs 🤩, and improved the accuracy of the predicted probabilities.


### Instructions to Run Each Implementation
Dependencies:
- Install `Python 3.x`(if not already installed).
- Install the required libraries by running the following commands:
```bash
pip install numpy pandas matplotlib scikit-learn
```

#### Running the First and Second Iterations (Manual Implementations):
Download the dataset `(winequality-red.csv and winequality-white.csv)` and upload after you open the Notebook to run it using this Colab snippet:
```python
from google.colab import files
uploaded = files.upload()
```
#### These notebooks will:

- Manually split the dataset, standardize the data, and train the logistic regression model without the use of scikit-learn.
- Output the training loss and the final test accuracy.

#### Running the Third Iteration (Library-based Implementation with scikit-learn):
- This notebook uses scikit-learn's built-in tools to handle data splitting, standardization, and logistic regression.

- The model will display:
  - loss during training.
  - The final test accuracy after training.

# Conclusion
This project highlights the progression from manual implementation of machine learning algorithms to utilizing established libraries such as scikit-learn. 

Each iteration builds upon the previous, enhancing both understanding and model performance, while progressively reducing the complexity of implementation.
