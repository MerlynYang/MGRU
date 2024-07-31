import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cells

class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, arch_type, device, dropout):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        self.arch_type = arch_type
        self.device = device
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        else:
            raise ValueError('Other rnn_type not supported yet')
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Tanh()

    def init_hidden(self, batch_size, input_dtype):
        if self.rnn_type == 'GRU':
            return torch.zeros(1, batch_size, self.hidden_size, dtype=input_dtype, device=self.device)
        elif self.rnn_type == 'LSTM':
            return (torch.zeros(1, batch_size, self.hidden_size, dtype=input_dtype, device=self.device), torch.zeros(1, batch_size, self.hidden_size, dtype=input_dtype, device=self.device))

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input_seq.size(0), input_seq.dtype)
        # input_seq [batch_size, seq_len, input_size]
        # hidden [1, batch_size, hidden_size]
        # outputs [batch_size, seq_len, hidden_size]
        # hidden [1, batch_size, hidden_size]
        outputs, hidden = self.rnn(input_seq, hidden)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # stack : use all hidden states
        # slide : use the last hidden states
        if self.arch_type == 'stack': dropout = self.dropout(outputs)
        elif self.arch_type == 'slide': dropout = self.dropout(outputs[:, -1, :])
        
        decode = self.activation(self.output(dropout))
        return decode, hidden

class MRNNFModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, lag_k, arch_type, device, dropout, init_d_value:list=None):
        super(MRNNFModel, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lag_k = lag_k
        self.arch_type = arch_type
        self.device = device
        
        # rnn layer, default batch first is True
        if self.rnn_type == 'MGRUF':
            self.rnn = cells.MGRUFCell(self.input_size, self.hidden_size, self.lag_k)
        elif self.rnn_type == 'MLSTMF':
            self.rnn = cells.MLSTMFCell(self.input_size, self.hidden_size, self.lag_k)
        else: raise ValueError('Other rnn_type not supported yet')
        self.init_mem_para(init_d_value)
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    # init_d_value should be a vector with size hidden_size
    def init_mem_para(self, init_d_value=None):
        self.mem_para = Parameter(torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device), requires_grad=True)
        if not init_d_value is None:
            init_d_value = torch.tensor(init_d_value).float()
            mem_para_value = torch.log((2 * init_d_value) / (1 - 2 * init_d_value))
            self.mem_para = Parameter(mem_para_value, requires_grad=True)

    def init_hidden(self, batch_size, input_dtype):
        if self.rnn_type == 'MGRUF':
            return torch.zeros(self.lag_k, batch_size, self.hidden_size, dtype=input_dtype, device=self.device)
        elif self.rnn_type == 'MLSTMF':
            return (torch.zeros(1, batch_size, self.hidden_size, dtype=input_dtype, device=self.device), torch.zeros(self.lag_k, batch_size, self.hidden_size, dtype=input_dtype, device=self.device))

    # input : [batch_size, seq_len, input_size]
    # hidden : [lag_k, batch_size, hidden_size]
    # d_values : [hidden_size]
    # decode_seq : [batch_size, seq_len, hidden_size]
    def forward(self, input_seq, hidden=None):
        # calculate the weights
        d_values = 0.5 * self.sigmoid(self.mem_para)
        init_col = torch.arange(0, self.lag_k, dtype=torch.float32).reshape(-1, 1)
        weights_k_h = init_col.repeat(1, self.hidden_size)
        weights_k_h = (weights_k_h - d_values) / (weights_k_h + 1)
        weights_k_h.cumprod_(dim=0)
        self.weights_over_hiddens = weights_k_h.flip(0)
        
        seq_len = input_seq.shape[1]
        decode_seq = []
        decode = None
        
        for seq in range(seq_len):
            hidden = self.rnn(input_seq[:, seq, :], hidden, self.weights_over_hiddens)
            # MGRUF return updated historcial hidden states
            if self.rnn_type == 'MGRUF': decode = self.activation(self.output(self.dropout(hidden[-1])))
            # MLSTMF return new hidden states and updated historical cell states
            elif self.rnn_type == 'MLSTMF': decode = self.activation(self.output(self.dropout(hidden[0])))
            
            if self.arch_type == 'stack': decode_seq.append(decode)

        if self.arch_type == 'stack':
            decode_seq = torch.stack(decode_seq, dim=1)
            return decode_seq, hidden
        elif self.arch_type == 'slide':
            return decode, hidden

class MRNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, lag_k):
        super(MRNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lag_k = lag_k
        if self.rnn_type == 'MGRU':
            self.rnn = cells.MGRUCell(self.input_size, self.hidden_size, self.output_size, self.lag_k)
        elif self.rnn_type == 'MLSTM':
            self.rnn = cells.MLSTMCell(self.input_size, self.hidden_size, self.output_size, self.lag_k)
            
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'MGRU':
            return (weight.new_zeros(self.lag_k, bsz, self.hidden_size), weight.new_zeros(bsz, self.hidden_size))
        elif self.rnn_type == 'MLSTM':
            return (weight.new_zeros(bsz, self.hidden_size), weight.new_zeros(self.lag_k, bsz, self.hidden_size), weight.new_zeros(bsz, self.hidden_size))
    
    def forward(self, input_seq, hidden=None):
        seq_len = input_seq.shape[0]
        batch_size = input_seq.shape[1]
        outputs = torch.zeros(seq_len, batch_size, self.output_size, dtype=input_seq.dtype, device=input_seq.device)
        for seq in range(seq_len):
            outputs[seq, :, :], hidden = self.rnn(input_seq[seq, :, :], hidden)
        return outputs, hidden
