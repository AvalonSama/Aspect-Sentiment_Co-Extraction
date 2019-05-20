import torch
import torch.nn as nn
import numpy as np

def getMask(inputs,seq_len):
	batch_size, max_seq_len, _ = inputs.size()
	query = torch.arange(0, max_seq_len, device=inputs.device).unsqueeze(0).float()
	mask = torch.lt(query, seq_len.unsqueeze(1).float())
	return mask





class Attention_layer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Attention_layer, self).__init__()
		self.M = nn.Sequential(
			nn.Linear(in_dim,out_dim),
			nn.Tanh()
		)
	def forward(self, key, value, query, seq_len):
		'''
		key: batch_size *  max_seq_len * hidden_size
		value = key
		query: batch_size * ca_num * embedding_dim
		'''
		batch_size, max_seq_len, hidden_size = value.size()
		temp_query = M(query)
		temp_alpha = torch.bmm(key,temp_query.transpose(1,-1)).transpose(1,-1) # batch_size * ca_num * max_seq_len 
		temp_alpha = torch.exp(temp_alpha)
		mask = getMask(value, seq_len)
		ca_num = temp_alpha.size(1)
		mask = mask.repeat(1,ca_num,1) #batch

