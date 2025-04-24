# Linear Regression

"""
Linear Regression is a machine learning algorithm that is used to predict the output of a function.
It is a simple algorithm that is used to predict the output of a function.

Foundation for NN 

Regression is opposite of Classification 

Classification is a supervised learning algorithm that is used to predict the output of a function.
Example: 
Classify a person as diabetic or not diabetic. There are 2 categories/classes

Regression does not have a fixed number of categories.
Classification has a fixed number of categories.

Linear : 
h(x, y, z) = w1x + w2y + w3z + b (w1, w2, w3 are weights and b is bias)

Over training, we will adjust the weights and bias to minimize the error.

Example:
Predict how tall a person will be given their weight, height, and their parents' height. + some base height

psuedo code:
    for some number of iterations:
        get_model_prediction() - get the model's current prediction
        get_error() - we want this to be as close to 0 as possible
        get_derivatives() - this will be used to perform gradient decent
        update_weights() - updating the weights and bias to minimize the error
    
    Mean Squared Error (MSE): 
    - how well a model's predictions match the actual values by measuring the 
        average of the squared differences between them, with smaller values indicating a better fit
        
        for every N in our N training example
            get the model's prediction - truth value and square that and divide by N
            N SUM i=1 to N (y - y')^2 / N
            
    
    We would use vectors and matrices to calcuate the derivatives.
    
    [x y z] * = w1x + w2y + w3z (DOT PRODUCT)
    [w1
     w2
     w3]

"""

import numpy as np
from numpy.typing import NDArray

def get_model_prediction(X, weights):
    # X is an Nx3 NumPy array and is the dataset that we want to predict with
    # weights is a 3x1 NumPy array
    # HINT: np.matmul() will be useful
    prediction = np.matmul(X, weights)
    
    # To show what happened behind the scenes for the matrix multiplication: 
    
    """
    This is a dot product 
    prediction = np.zeros(len(X))
    
    for i in range(len(X)):
        for j in range(len(weights)):
            prediction[i] += X[i][j] * weights[j] 
            # => 0.3745401188473625 * 1.0 + 0.9507143064099162 * 2.0 + 0.7319939418114051 * 3.0
    """
    # return np.round(your_answer, 5)
    return np.round(prediction, 5)

def get_error(model_prediction, ground_truth):
    # equation is the mean squared error
    # Equation for MSE: mean of the squares of the difference between the model prediction and the ground truth as shown below
    error = np.mean(np.square(np.array(model_prediction) - np.array(ground_truth)))
    return round(error, 5)

    
if __name__ == "__main__":
    X = [[0.3745401188473625, 0.9507143064099162, 0.7319939418114051]]
    weights = [1.0, 2.0, 3.0]
    prediction = get_model_prediction(X, weights)
    print(prediction)
    
    # What do the model predictions tell us? 
        # Answer: predicted values of the data that we have based on the model
    # What do the ground truth values tell us?
        # Answer: actual values of the data that we have
    
    # What next? 
        # Compare the model prediction to the ground truth
    model_prediction=[[0.37454012], [0.95071431], [0.73199394]] 
                    # underfit,      overfit,       overfit 
    ground_truth=[    [0.59865848], [0.15601864],  [0.15599452]]
    # Comparing the .37 to .59 -> model is underfitting
    # Comparing the .95 to .16 -> model is overfitting
    # Comparing the .73 to .16 -> model is overfitting 
    error = get_error(model_prediction, ground_truth)
    print(error)
    
    # What is the error?
        # Answer: the model is overfitting the data because the error is high, 
        # we want it to be low as close to 0 as possible