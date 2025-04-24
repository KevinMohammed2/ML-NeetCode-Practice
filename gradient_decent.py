# Gradient Decent Understanding
"""
    Gradient Decent is an optimization algorithm that is used to find the minimum value of a function.
    It is a local search algorithm that uses the derivative of the function to find the minimum value.
    
    The derivative of a function is the rate of change of the function at a point.
    
    Definitions of Variables:
    iterations: The number of times the algorithm will run.
    learning_rate: The rate at which the algorithm will learn.
    init: The initial value of the algorithm.
    
    Use Cases: 
    1. Finding the minimum value of a function.
    2. Finding the minimum value of a cost function.
    3. Finding the minimum value of a loss function.
"""

def get_minimizer (iterations, learning_rate, init):
    
    minimizer = init
    
    for _ in range(iterations):
        derivative = 2 * minimizer
        print(f"Minimizer: {minimizer}, Derivative: {derivative}")
        minimizer = minimizer - learning_rate * derivative
    return round(minimizer, 5)


if __name__ == "__main__":
    
    # iterations = 0
    # learning_rate = 0.01
    # init = 5
    # get_minimizer(iterations, learning_rate, init)
    # Returns 5
    
    iterations = 10
    learning_rate = 0.01 
    init = 5
    get_minimizer(iterations, learning_rate, init)