import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


# torch.tensor(python_list) returns a Python list as a tensor
class NLP_Intro(nn.Module):
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        
        # Get total set of words 
        # Tensor = dictionary 
        dictionary = set()
        
        # Add all words to the dictionary 
        for sentence in positive:
            for word in sentence.split():
                dictionary.add(word)

        for sentence in negative:
            for word in sentence.split():
                dictionary.add(word)

        sorted_dict = sorted(list(dictionary))

        # Build a mapping
        word_to_int = {}
        # Map each word to a int
        for i in range(len(sorted_dict)):
            word_to_int[sorted_dict[i]] = i + 1

        tensors = []
        for sentence in positive:
            curr_list = []
            for word in sentence.split(): 
                curr_list.append(word_to_int[word])
            tensors.append(torch.tensor(curr_list))
        
        for sentence in negative:
            curr_list = []
            for word in sentence.split():
                curr_list.append(word_to_int[word])
            tensors.append(torch.tensor(curr_list))
        # Write encode() - used to build the dataset 

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)

nlp = NLP_Intro()

# Get the dataset
positive=["Good case, Excellent value.","Great for the jawbone.","The mic is great.","If you are Razr owner...you must have this!","Highly recommend for any one who has a blue tooth phone"]
negative=["So there is no way for me to plug it in here in the US unless I go buy a converter.","Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!","I have to jiggle the plug to get it to line up right to get decent volume.","Needless to say, I wasted my money.","What a waste of money and time!"]

dataset = nlp.get_dataset(positive, negative)

print(dataset)