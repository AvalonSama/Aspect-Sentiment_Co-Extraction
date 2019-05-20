import torch
import torch.nn as nn
import numpy as np

class RNN_layer(nn.Module):
    def __init__(self, args)
        super(RNN_layer,self).__init__()
        self.n_hidden = args.n_hidden
        self.n_layer = args.n_layer
        self.bi_direction = 2 if args.bi_direction else 1
        self.rnn_type = args.rnn_type
        model_mapping = {'RNN':nn.RNN, 'LSTM':nn.LSTM, 'GRU':nn.GRU}
        self.rnn = model_mapping[self.rnn_type](
            input_size = args.input_size,
            hidden_size = self.n_hidden,
            num_layers = self.n_layer,
            bias=True,
            batch_first = True,
            dropout = drop_prob,
            bidirectional = self.bi_direction
        )
        
    def forward(self, inputs, seq_len, out_type = 'last'):
        now_batch_size, max_seq_len, _ = inputs.size()
        sort_seq_len, sort_index = torch.sort(seq_len, descending = True)
        _, unsort_index = torch.sort(sort_index, dim=0, descending = False)
        inputs = torch.index_select(inputs, 0, sort_index)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, sort_seq_len, batch_first = True)
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            outputs, h_last = self.rnn(inputs)
        elif self.rnn_type == 'LSTM':
            outputs, (h_last, _) = self.rnn(inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True, total_length = max_seq_len)
        outputs = torch.index_select(outputs, 0, unsort_index)
        
        h_last = h_last.view(self.n_layer, self.bi_direction, now_batch_size, self.n_hidden)
        h_last = torch.reshape(h_last[-1].transpose(0,1),[now_batch_size,self.bi_direction*self.n_hidden])
        h_last = torch.index_select(h_last,0,unsort_index)
        if out_type == 'all':
            return outputs
        elif out_type == 'last':
            return h_last
        elif out_type == 'all_and_last':
            return outputs, h_last
        

        