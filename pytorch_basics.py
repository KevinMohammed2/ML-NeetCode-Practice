# Pytorch Basics

import torch
import torch.nn
from torchtyping import TensorType

# Helpful functions:
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.mean.html
# https://pytorch.org/docs/stable/generated/torch.cat.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        """
        Explaination: 
        We want to reshape this tensor from a 3x4 matrix to a (MxN//2) x 2 matrix. - floor division = // 
        torch.reshape() will be used and the parameters are: (tensor, (dim0, dim1))
        """
                
        print(to_reshape)
        shape = to_reshape.size()
        m = shape[0]
        n = shape[1]
        math = (m * n // 2)
        to_reshape = torch.reshape(to_reshape, (math, 2))
        return torch.round(to_reshape, decimals = 4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        """
        Explaination: 
        We want to average the rows of this tensor. 
        torch.mean() will be used and the parameters are: (tensor, dim = 0)
        dim = 0 means we want to average the rows
        dim = 1 means we want to average the columns
        """
        
        
        meaned = torch.mean(to_avg, dim = 0)
        return torch.round(meaned, decimals = 4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        """
        Explaination: 
        We want to concatenate these two tensors. 
        torch.cat() will be used and the parameters are: ((tensor1, tensor2), dim = 1)
        dim 1 means we will concatenate the columns
        dim 0 means we will concatenate the rows
        
        Visual Example: 
        Cat 1 tensor
        ([[0.5765, 0.2743, 0.6465],
        [0.8625, 1.7877, 0.6030]])
        
        Cat 2 tensor
        ([[-0.6152,  0.4082],
        [ 0.0710,  1.5451]]) 

        Concatenated tensor(
        [[ 0.5765,  0.2743,  0.6465, -0.6152,  0.4082],
        [ 0.8625,  1.7877,  0.6030,  0.0710,  1.5451]])
        
        The dimensions that we want to concatenate must be the same in order for us to do it
        
        Cat1 = (2,3)
        Cat2 = (2,2)
        """
        
        print("\n\nCat 1", cat_one)
        print("Cat 2", cat_two, "\n\n")
        concat = torch.cat((cat_one, cat_two), 1)
        return torch.round(concat, decimals = 4)


    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        """
        Explaination:
        We want to find the mean squared error between these two tensors. 
        torch.nn.functional.mse_loss() will be used and the parameters are: (prediction, target)
        prediction = (5,)
        target = (5,)
        
        The MSE means we want to find the mean of the squares of the difference between the prediction and the target
        
        Equation for MSE: mean of the squares of the difference between the model prediction and the ground truth as shown below
        error = np.mean(np.square(np.array(model_prediction) - np.array(ground_truth)))
        return round(error, 5)
    
        """
        
        mse_loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(mse_loss, decimals = 4)


if __name__ == "__main__":
    solution = Solution()
    
    to_reshape = torch.ones(3, 4)
    print(solution.reshape(to_reshape))
    
    
    to_avg = torch.tensor([[0.8088, 1.2614, -1.4371],
                          [-0.0056, -0.2050, -0.7201]])
    print(solution.average(to_avg))
    
    
    cat_one = torch.randn(2,3)
    cat_two = torch.randn(2,2)
    print(solution.concatenate(cat_one, cat_two))
    
    
    pred = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    print(solution.get_loss(pred, target))
    
    
    
    
    # No need to change this code
    
    #   def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
    #     # torch.reshape() will be useful - check out the documentation
    #     shape = to_reshape.size()
    #     m = shape[0]
    #     n = shape[1]
    #     math = m * n // 2
    #     reshaped = torch.reshape(to_reshape, (math, 2))
    #     return torch.round(reshaped, decimals = 4)

    # def average(self, to_avg: TensorType[float]) -> TensorType[float]:
    #     # torch.mean() will be useful - check out the documentation
    #     meaned = torch.mean(to_avg, dim=0)
    #     return torch.round(meaned, decimals = 4)

    # def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
    #     # torch.cat() will be useful - check out the documentation
    #     concat = torch.cat((cat_one, cat_two), 1)
    #     return torch.round(concat, decimals = 4)


    # def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
    #     # torch.nn.functional.mse_loss() will be useful - check out the documentation
    #     mse_loss = torch.nn.functional.mse_loss(prediction, target)
    #     return torch.round(mse_loss, decimals = 4)