# Digit Classifier 

import torch
import torch.nn as nn 
from torchtyping import TensorType
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# NN Models 

# Example 
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # nn.Linear parameters are (in_features, out_features, bias = True, device = None, dtype = None)
        
        # Dropout : used to solve the problem of overfitting 
        # Training accuracy is higher than Testing Accuracy
        # Training is memorizing the training data and Testing is memorizing the testing data
        # Too complex, too many nodes, too many weights
        
        # nn.Dropout(p = 0.2) - 20% dropout - Node will be turned off and reduce complexity by not taking into account 
        #                                     any information from that node -> makes it simpler -> less overfitting 
        
        # nn.ReLU()
        # ReLU - Rectified Linear Unit 
        # ReLU(x) = max(0, x)
        # ReLU is a non-linear activation function
        # It is used to introduce non-linearity into the network - makes it more complex
        
        
        # Define the layers
        self.first_layer = nn.Linear(4, 6) # Input is 4 layers and then we have 6 layers
        self.second_layer = nn.Linear(6, 6) # Input is 6 layers and then we have 6 layers
        self.final_layer = nn.Linear(6, 2) # Input is 6 layers and then we have 2 layers - down projecting to 2
        
    def forward(self, x):
        # first_layer_output = self.first_layer(x) # Output of the first layer - could also be written as self.first_layer().forward(x)
        # To writen this concisely we can do the following:
        
        # This will call all the previous layers within themselves and make the code smaller. 
        # Feeding the output of one layer to the next
        return self.final_layer(self.second_layer(self.first_layer(x))) 
    


# Example 2 - Digit Classification
class DigitClassifier(nn.Module):
    def __init__(self):
        # Define the architecture here
        super().__init__()
        torch.manual_seed(8)
        self.first_layer = nn.Linear(784, 512) # Input is 784 layers and then we have 512 layers
        # ReLU: ReLU is a non-linear activation function that will introduce non-linearity into the network
        # ReLU(x) = max(0, x): We wonly want the positives to allow model to learn features without vanishing gradients
        # no negatives to introduce non-linearity and sparsity -> help with generalization and efficiency
        self.relu_activation = nn.ReLU() 

        # Dropout - used to solve the problem of overfitting, training accuracy is higher than testing accuracy
        # Will drop out 20% of the nodes to prevent overfitting
        self.dropout = nn.Dropout(p = 0.2)
        
        # final layer -> 10 nodes 
        self.final_layer = nn.Linear(512, 10)
        
        # Sigmoid : Used to determine a category for the result
        # It is on a 0-1 scale where the closer to 1 means the more likely it is that and the closer to 0 means the more likely it is that
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # Return the model's prediction to 4 decimal places
        torch.manual_seed(0)
        # We will be calling each funcation within itself to allow for the model to learn
        return torch.round((self.sigmoid(self.final_layer(self.dropout(self.relu_activation(self.first_layer(images)))))), decimals = 4)
        

# TEST CASE
digitTest = DigitClassifier()

images = torch.randn(1, 28 * 28)
print(digitTest(images))
    
    
    
model = MyModel()

# Need to train the model for some number of iterations

# THEN we can actually user the model and get predictions

example_datapoint = torch.randn(1, 4) # 1x4 tensor 

# print(model(example_datapoint))


from jaxtyping import Float


torch.manual_seed(0)

class DigitReconition(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2)
        self.projection = nn.Linear(512, 10)
        # self.sigmoid = nn.Sigmoid() - Not used when using CrossEntropyLoss
        
    def forward(self, images: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return self.projection(self.dropout(self.relu(self.first_layer(images))))

# This is he code to train the mode. Come up with the train_dataloader and run this in a jupyter notebook 

model = DigitReconition()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten from [1, 28, 28] to [784]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for epoch in range(epochs):
    for images, labels in train_dataloader:
        images = images.view(images.shape[0], 784)
        
        # Training Body
        model_prediction = model(images)
        optimizer.zero_grad()
        loss = loss_function(model_prediction, labels)
        loss.backward()
        optimizer.step()

model.eval()

for images, labels in test_dataloader:
    images = images.view(images.shape[0], 784)
    
    model_prediction = model(images)
    max, idx = torch.max(model_prediction, dim = 1)
    for i in range (len(images)):
        plt.imshow(images[i].view(28, 28))
        plt.show()
        print(idx[i].item())
    break