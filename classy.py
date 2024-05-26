import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self,input_size, hidden_size, out_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers=num_layers
        self.lstm_1 = nn.LSTM(input_size,hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size,out_size)
        # Initialize hidden states
        self.hidden = self.init_hidden(batch_size=1)

    def init_hidden(self, batch_size):
        # Create initial hidden and cell states for each layer
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)

    def forward(self, seq, hidden_state):
        lstm_out_1, self.hidden = self.lstm_1(seq.view(len(seq), -1, self.input_size), hidden_state)
        
        pred = self.linear(lstm_out_1.view(len(seq), -1))
        hidden_state = self.hidden
        return pred