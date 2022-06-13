import torch
import torch.nn as neuralNetwork
import torch.optim as optimizer 
import torch.nn.functional as functional
import os


class Linear_QNetwork(neuralNetwork.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = neuralNetwork.Linear(input_size, hidden_size)
        self.linear2 = neuralNetwork.Linear(hidden_size, output_size)

    def forward(self, x):
        # moving the data through the layers 
        x = functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
class Qtrainer:
    def __init__(self, model, learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = neuralNetwork.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state

        Q = self.model(state)



        # 2: Q_new =  r + y * max(next_prediction Q value) => only do this if not done
        # prediction.clone()
        #predictions[argmax(Action)] =  Q_new

        target = Q.clone()

        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        

        self.optimizer.zero_grad()
        loss = self.criterion(target, Q)
        loss.backward()

        self.optimizer.step()


