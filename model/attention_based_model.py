import sys
sys.path.append("..")
from data_prepare.utils import W2V, LoadData, get_Args
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from layer.rnn_layer import RNN_layer
from layer.attention_layer import 

args = get_Args()
word_dict, embedding = W2V(embeddingFile = args.embedding_file_path, wordBagFile = args.word_bag_path)
print("**************Loading Train Data****************")
train_data, train_len, train_label, category_data = LoadData(
	filename = args.train_file_path, 
	word_dict = word_dict,
	labellist = ['negative','positive','neutral','N/A'],
	categorylist = ['RESTAURANT','SERVICE','FOOD','DRINKS','AMBIENCE','LOCATION'], 
	maxseqlen = args.maxseqlen
)
print("**************Loading Test Data****************")
test_data, test_len, test_label, _ = LoadData(
	filename = args.test_file_path, 
	word_dict = word_dict,
	labellist = ['negative','positive','neutral','N/A'],
	categorylist = ['RESTAURANT','SERVICE','FOOD','DRINKS','AMBIENCE','LOCATION'], 
	maxseqlen = args.maxseqlen
)
category_data = torch.tensor(category_data, dtype = torch.int)
train_x = torch.tensor(train_data, dtype = torch.int)
train_y = torch.tensor(train_label, dtype = torch.int)
train_len = torch.tensor(train_len, dtype = torch.int)
test_x = torch.tensor(test_data, dtype = torch.int)
test_y = torch.tensor(test_data, dtype = torch.int)
test_len = torch.tensor(test_len, dtype = torch.int)



class AttentionBasedASCE(nn.Module):
	def __init__(self, embedding, category_data, args):
		super(AttentionBasedASCE,self).__init__()

		self.embedding_matrix = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float))
		self.category_data = category_data
		self.drop_out = nn.Dropout(args.drop_prob)
		self.rnn = RNN_layer(args)
		self.attention = Attention_layer()
		self.predictor = Predictor()
	
	def forward(self, inputs, seq_len):
		inputs = self.embedding(inputs.long())
		ca_matrix = self.embedding(self.category_data)
		
		H, _ = self.rnn(inputs, seq_len, out_type = 'all_and_last')
		attention_Based_R = self.attention(H, ca_matrix, seq_len)
		pred = self.predictor(attention_Based_R)
		return pred

model = AttentionBasedASCE(embedding, category_data, args)
model.cuda(0)
optimizer = torch.optim.Adam(
	model.parameters(), lr = args.learning_rate, weight_decay = args.l2_reg
)
train_data = Data.TensorDataSet(train_x, train_y, train_len)
test_data = Data.TensorDataSet(test_x, test_y, test_len)

train_loader = Data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
test_loader = Data.DataLoader(test_x, shuffle=True, batch_size = args.batch_size)

for it in range(1,args.iter_times):
	model.train()
	
	train_loss = []
	for (_x, _y, _len) in train_loader:
		_x, _y, _len = _x.cuda(0), _y.cuda(0), _len.cuda(0)
		pred = model(_x, _len)  #batchsize*24
		loss = -torch.sum(_y.float()*torch.log(pred))/_x.size()[0]
		train_loss.append(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	model.eval()
	t_x, t_y, t_len = test_x.cuda(0), test_y.cuda(0), test_len.cuda(0)
	pred = model(t_x, t_len)

	pred = pred.cpu().data.numpy()
	result = analysis(pred, t_y.cpu().data.numpy())
	print(result['Acc'])

	







