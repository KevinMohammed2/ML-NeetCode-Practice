import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        """
        Update Rule:
        1. new_weight1 = old_weight1 - derivative(w1) * learning_rate
        2. repeat this for all the weights
        3. update weights
        4. repeat this for num_iterations
        """
        
        # We are going to iterate over the number of iterations to get as close to the correct answers
        for i in range(num_iterations):
            # Get the model prediction, required for the derivative
            model_pred = self.get_model_prediction(X, initial_weights)
            
            # Get the derivatives, required to update the weights
            # Parameters: model_pred, Y, N, X, desired_weight index of the array
            dw1 = self.get_derivative(model_pred, Y, len(X), X, 0)
            dw2 = self.get_derivative(model_pred, Y, len(X), X, 1) 
            dw3 = self.get_derivative(model_pred, Y, len(X), X, 2) 

            # Update the weights with the derivatives and the learning rate
            # Initial Weight = Initial Weight - Derivative * Learning Rate
            # initial_weights[0] -= dw1 * self.learning_rate (Another way of writing it)
            initial_weights[0] = initial_weights[0] - dw1 * self.learning_rate
            initial_weights[1] = initial_weights[1] - dw2 * self.learning_rate
            initial_weights[2] = initial_weights[2] - dw3 * self.learning_rate
            print(initial_weights)

        # return the updated weights
        return np.round(initial_weights, 5)

        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        pass

if __name__ == "__main__":
    
    # We see that the predicted added up will equal the ground truth
    X = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float64)
    Y = np.array([6, 3], dtype=np.float64)
    
    # increasing iterations increase the accuracy to the truth value
    num_iterations = 1000
    initial_weights = [0.2, 0.1, 0.6]
    results = Solution().train_model(X, Y, num_iterations, initial_weights)
    print(results)
    
# Input:
# X = [[1, 2, 3], [1, 1, 1]]
# Y = [6, 3]
# num_iterations = 10
# initial_weights = [0.2, 0.1, 0.6]

# Output:
# [0.50678, 0.59057, 1.27435]