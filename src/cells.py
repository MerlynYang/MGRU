# This file defines the cell structure for the RNNs (MLSTMF, MGRUF, MLSTM, MGRU).
# This file is partially referenced from the following GitHub repository:
# https://github.com/huawei-noah/noah-research/tree/master/mRNN-mLSTM

import torch
import torch.nn as nn

class MGRUFCell(nn.Module):
    def __init__(self, input_size, hidden_size, lag_k):
        super(MGRUFCell, self).__init__()
        # initialization of parameters
        self.lag_k = lag_k
        self.input_size = input_size
        self.hidden_size = hidden_size
        # set the gates
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_tilde = nn.Linear(input_size + hidden_size, hidden_size)
        # weight initialization of weight in each gate
        # ! be careful with the method
        self.weight_init(self.update_gate)
        self.weight_init(self.reset_gate)
        
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def weight_init(self, gate, method='xavier_uniform'):
        if isinstance(gate, nn.Linear):
            if method == 'orthogonal': nn.init.orthogonal_(gate.weight)
            elif method == 'xavier_uniform': nn.init.xavier_uniform_(gate.weight)
            elif method == 'xavier_normal': nn.init.xavier_normal_(gate.weight)
            elif method == 'kaiming_uniform': nn.init.kaiming_uniform_(gate.weight)
            elif method == 'kaiming_normal': nn.init.kaiming_normal_(gate.weight)
            elif method == 'normal': nn.init.normal_(gate.weight)
            elif method == 'uniform': nn.init.uniform_(gate.weight)
            else: raise ValueError('Initialization method not implemented yet')
            nn.init.constant_(gate.bias, 0)

    def forward(self, sample, historical_hiddens, weights):
        # sample : [batch_size, input_size]
        # historical_hiddens : [lag_k, batch_size, hidden_size]
        batch_size = sample.shape[0]
        if historical_hiddens is None:
            historical_hiddens = torch.zeros(self.lag_k, batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
        # h_{t-1}
        last_hidden = historical_hiddens[-1,:]
        # [x_t h_{t-1}] : [batch_size, (input_size + hidden_size)]
        combined_x_h = torch.cat((sample, last_hidden), 1)
        # calculate the gates
        update_gate = self.update_gate(combined_x_h)
        update_gate = self.sigmoid(update_gate)
        reset_gate = self.reset_gate(combined_x_h)
        reset_gate = self.sigmoid(reset_gate)
        # [x_t (r_t * h_{t-1})] : [batch_size, (input_size + hidden_size)]
        combined_x_rh = torch.cat((sample, torch.mul(reset_gate, last_hidden)), 1)
        # calculate h_tilde : [batch_size, hidden_size]
        h_tilde = self.h_tilde(combined_x_rh)
        h_tilde = self.tanh(h_tilde)
        # h_t = - (weights * historical_hiddens) + u_t * h_{t-1}
        # weights : [k, hidden_size]
        # historical_hiddens : [k, batch_size, hidden_size]
        weighted_historical_hiddens = torch.einsum('kbh, kh->kbh', [historical_hiddens, weights]).sum(dim=0)
        updated_last = torch.mul(update_gate, h_tilde)
        hidden_new = torch.add(-weighted_historical_hiddens, updated_last)
        # hiddens : k * batch_size * hidden_size
        # hidden_now : batch_size * hidden_size
        updated_hiddens = torch.cat([historical_hiddens[1:, :], hidden_new.view([-1, hidden_new.shape[0], hidden_new.shape[1]])], 0)
        # if not historical_hiddens is None and torch.isnan(historical_hiddens).any():
        #     self.get_position(historical_hiddens)
        #     raise ValueError('historical_hiddens has nan')
        # if torch.isnan(weights).any():
        #     self.get_position(weights)
        #     raise ValueError('weights has nan')
        return updated_hiddens

class MLSTMFCell(nn.Module):
    def __init__(self, input_size, hidden_size, k):
        super(MLSTMFCell, self).__init__()
        # initialization of parameters
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        # set the gates
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.c_tilde = nn.Linear(input_size + hidden_size, hidden_size)
        # weight initialization of weights in each gate
        self.weight_init(self.i_gate)
        self.weight_init(self.o_gate)
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def weight_init(self, gate, method='xavier_uniform'):
        if isinstance(gate, nn.Linear):
            if method == 'orthogonal': nn.init.orthogonal_(gate.weight)
            elif method == 'xavier_uniform': nn.init.xavier_uniform_(gate.weight)
            elif method == 'xavier_normal': nn.init.xavier_normal_(gate.weight)
            elif method == 'kaiming_uniform': nn.init.kaiming_uniform_(gate.weight)
            elif method == 'kaiming_normal': nn.init.kaiming_normal_(gate.weight)
            elif method == 'normal': nn.init.normal_(gate.weight)
            elif method == 'uniform': nn.init.uniform_(gate.weight)
            else: raise ValueError('Initialization method not implemented yet')
            nn.init.constant_(gate.bias, 0)
        
    def forward(self, sample, hiddens_all, weights):
        batch_size = sample.shape[0]
        hidden_state = None
        cell_tensors = None
        if hiddens_all is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
            cell_tensors = torch.zeros(self.k, batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
        else:
            hidden_state = hiddens_all[0]
            cell_tensors = hiddens_all[1]
        combined_x_h = torch.cat((sample, hidden_state), 1)
        i_gate = self.i_gate(combined_x_h)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.o_gate(combined_x_h)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.c_tilde(combined_x_h)
        c_tilde = self.tanh(c_tilde)
        # i_{t} * c_tilde_{t}
        gated_c_tilde = torch.mul(i_gate, c_tilde)
        weighted_his_cell = torch.einsum('ijk,ik->ijk', [-cell_tensors, weights]).sum(dim=0)
        cell_new = torch.add(weighted_his_cell, gated_c_tilde)
        all_cell_states = torch.cat([cell_tensors, cell_new.view([-1, cell_new.size(0), cell_new.size(1)])], 0)
        updated_cell_states = all_cell_states[1:, :]
        hidden_state = torch.mul(self.tanh(cell_new), o_gate)
        
        return (hidden_state, updated_cell_states)

class MGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k, dropout=0.):
        super(MGRUCell, self).__init__()
        # initialization of parameters
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # set the gates
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.memory_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.h_tilde = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        # weight initialization of weight in each gate
        # self.weight_init(self.update_gate)
        # self.weight_init(self.reset_gate)
        # self.weight_init(self.memory_gate)
        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def weight_init(self, gate):
        if isinstance(gate, nn.Linear):
            nn.init.orthogonal_(gate.weight)
            nn.init.constant_(gate.bias, 0)
            
    def get_weight_1d(self, mem_para):
        weights = [1.] * (self.k + 1)
        # i = 0 => weight[k-1]
        # i = k-1 => weight[0]
        for i in range(0, self.k):
            weights[self.k - i - 1] = weights[self.k - i] * (i - mem_para) / (i + 1)
        return torch.cat(weights[0:self.k])

    def mem_filter(self, hiddens, mem_para):
        # weights : k * batch_size * hidden_size
        # mem_para : batch_size * hidden_size
        weights = torch.ones(self.k, mem_para.size(0), mem_para.size(1),
                             dtype=mem_para.dtype, device=mem_para.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                mem_para_1d = mem_para[batch, hidden].view([1])
                weights[:, batch, hidden] = self.get_weight_1d(mem_para_1d)
        # hiddens : k * batch_size * hidden_size
        # hidden_filtered : batch_size * hidden_size
        hidden_filtered = hiddens.mul(weights).sum(dim=0)
        return hidden_filtered
    
    def forward(self, sample, hiddens_all):
        # mem_para : d_{t-1}
        batch_size = sample.size(0)
        if hiddens_all is None:
            hiddens = torch.zeros(self.k, batch_size, self.hidden_size,
                                 dtype=sample.dtype, device=sample.device)
            mem_para = torch.zeros(batch_size, self.hidden_size,
                                   dtype=sample.dtype, device=sample.device)
        else:
            hiddens = hiddens_all[0]
            mem_para = hiddens_all[1]
        # h_{t-1}
        hidden = hiddens[-1]
        # [x_t h_{t-1}] : batch_size * [input_size + hidden_size]
        # [x_t h_{t-1} d_{t-1}]
        combined_x_h = torch.cat((sample, hidden), 1)
        combined_x_h_d = torch.cat((combined_x_h, mem_para), 1)
        # calculate the gates
        update_gate = self.update_gate(combined_x_h)
        update_gate = self.sigmoid(update_gate)
        reset_gate = self.reset_gate(combined_x_h)
        reset_gate = self.sigmoid(reset_gate)
        memory_gate = self.memory_gate(combined_x_h_d)
        memory_gate = 0.5 * self.sigmoid(memory_gate)
        # [x_t (r_t * h_{t-1})] : batch_size * [input_size + hidden_size]
        combined_x_rh = torch.cat((sample, torch.mul(reset_gate, hidden)), 1)
        # calculate h_tilde
        h_tilde = self.h_tilde(combined_x_rh)
        h_tilde = self.tanh(h_tilde)
        # h_t = weights * hiddens + u_t * h_{t-1}
        # weights : k * hiddens
        # hiddens : k * batch_size * hidden
        hiddens_filtered = -self.mem_filter(hiddens, memory_gate)
        updated_last = torch.mul(update_gate, h_tilde)
        hidden_now = torch.add(hiddens_filtered, updated_last)
        # hiddens : k * batch_size * hidden_size
        # hidden_now : batch_size * hidden_size
        cat_hidden_now = hidden_now.view([-1, hidden_now.size(0), hidden_now.size(1)])
        all_hiddens = torch.cat([hiddens, cat_hidden_now], 0)
        updated_hiddens = all_hiddens[1:, :]
        output = self.output((hidden_now))
        
        return output, (updated_hiddens, memory_gate)

class MLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k, dropout=0.):
        super(MLSTMCell, self).__init__()
        # initialization
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # gates
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.memory_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.c_tilde = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        # initialization
        # self.weight_init(self.i_gate)
        # self.weight_init(self.o_gate)
        # self.weight_init(self.memory_gate)
        # activation function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def weight_init(self, gate):
        if isinstance(gate, nn.Linear):
            nn.init.orthogonal_(gate.weight)
            nn.init.constant_(gate.bias, 0)
            
    def get_weight_1d(self, mem_para):
        weights = [1.] * (self.k + 1)
        # i = 0 => weight[k-1]
        # i = k-1 => weight[0]
        for i in range(0, self.k):
            weights[self.k - i - 1] = weights[self.k - i] * (i - mem_para) / (i + 1)
        return torch.cat(weights[0:self.k])

    def mem_filter(self, cell_states, mem_para):
        # weights : k * batch_size * hidden_size
        # mem_para : batch_size * hidden_size
        weights = torch.ones(self.k, mem_para.size(0), mem_para.size(1),
                             dtype=mem_para.dtype, device=mem_para.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                mem_para_1d = mem_para[batch, hidden].view([1])
                weights[:, batch, hidden] = self.get_weight_1d(mem_para_1d)
        # hiddens : k * batch_size * hidden_size
        # hidden_filtered : batch_size * hidden_size
        hidden_filtered = cell_states.mul(weights).sum(dim=0)
        return hidden_filtered
    
    def forward(self, sample, hiddens_all):
        batch_size = sample.shape[0]
        hidden_state = None
        cell_tensors = None
        if hiddens_all is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
            cell_tensors = torch.zeros(self.k, batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
            mem_para = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype, device=sample.device)
        else:
            hidden_state = hiddens_all[0]
            cell_tensors = hiddens_all[1]
            mem_para = hiddens_all[2]
        
        combined_x_h = torch.cat((sample, hidden_state), 1)
        combined_x_h_d = torch.cat((sample, hidden_state, mem_para), 1)
        i_gate = self.i_gate(combined_x_h)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.o_gate(combined_x_h)
        o_gate = self.sigmoid(o_gate)
        mem_para = self.memory_gate(combined_x_h_d)
        mem_para = 0.5 * self.sigmoid(mem_para)
        c_tilde = self.c_tilde(combined_x_h)
        c_tilde = self.tanh(c_tilde)
        
        cell_filtered = -self.mem_filter(cell_tensors, mem_para)
        gated_c_tilde = torch.mul(c_tilde, i_gate)
        cell_new = torch.add(cell_filtered, gated_c_tilde)
        all_cell_states = torch.cat([cell_tensors, cell_new.view([-1, cell_new.size(0), cell_new.size(1)])], 0)
        updated_cell_states = all_cell_states[1:, :]
        
        hidden_state = torch.mul(o_gate, self.tanh(cell_new))
        output = self.output((hidden_state))
        
        return output, (hidden_state, updated_cell_states, mem_para)
