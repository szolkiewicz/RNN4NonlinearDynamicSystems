import torch.nn as nn
import torch

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size_1,hidden_size_2,out_size):
        super().__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.input_size = input_size
        self.lstm_1 = nn.LSTM(input_size,hidden_size_1, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_size_1,hidden_size_2, batch_first=True)
        self.linear = nn.Linear(hidden_size_2,out_size)
        self.hidden_1 = (torch.zeros(1,1,hidden_size_1), torch.zeros(1,1,hidden_size_1))
        self.hidden_2 = (torch.zeros(1,1,hidden_size_2), torch.zeros(1,1,hidden_size_2))
        
    def forward(self,seq):
        lstm_out_1 , self.hidden_1 = self.lstm_1(seq.view(-1,1,self.input_size),self.hidden_1)
        lstm_out_2 , self.hidden_2 = self.lstm_2(lstm_out_1,self.hidden_2)
        pred = self.linear(lstm_out_2.view(len(seq),-1))
        return pred
    