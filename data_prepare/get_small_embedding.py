from utils import read_tsv, FileManager, sepratewords
from nltk.tokenize import WordPunctTokenizer
import numpy as np

filelist = ["../data/Restaurants15_Train.tsv", "../data/Restaurants15_Test.tsv"]
bigEmbedding = dict()
smallEmbedding = dict()
wordbag = []
resultEmb = []
print("reading big embedding....")
cnt = 0
with open("/home/clli/w2v/glove.840B.300d.txt", 'r', encoding='utf-8') as f:
    data = [line.replace("\n","") for line in f.readlines()]
    for line in data:
        cnt+=1
        if cnt%100000==0:
            print(cnt)
        line = line.split()
        word =' '.join(line[:-300])
        embedding = [num for num in line[-300:]]
        bigEmbedding[word] = embedding
print("getting small embedding..")
wordbag.append('<PAD>')
resultEmb.append(' '.join([str(0.0)]*300))
for filename in filelist:
    data = read_tsv(filename)
    for line in data:
        text = line[0]
        text2id = []
        sep_text = sepratewords(text).split()
        
        for word in sep_text:
            if word in smallEmbedding:
                continue
            elif word not in bigEmbedding:
                temp_emb = np.random.rand(300)
                smallEmbedding[word] = ' '.join([str(num) for num in list(temp_emb)])
                wordbag.append(word)
                resultEmb.append(smallEmbedding[word])
            else:
                smallEmbedding[word] = ' '.join(bigEmbedding[word])
                wordbag.append(word)
                resultEmb.append(smallEmbedding[word])

FileManager("../data/restaurant_wordbag.data", 'w', wordbag)
FileManager("../data/restaurant_embedding.data", 'w', resultEmb)





            

        
