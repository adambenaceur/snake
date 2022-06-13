import torch
import torch.nn as neuralNetwork
import torch.optim as optimization 
import torch.nn.functional as functional
import os


class Linear_QNetwork(neuralNetwork.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__
        self.linear1 = neuralNetwork.Linear(input_size, hidden_size)
        self.linear2 = neuralNetwork.Linear(hidden_size, output_size)

    def forward(self, x):
        # moving the data through the layers 
        x = functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    

