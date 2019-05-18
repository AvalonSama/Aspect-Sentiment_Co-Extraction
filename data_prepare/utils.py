import csv
from nltk.tokenize import WordPunctTokenizer
import numpy as np
def read_tsv(filename):
    lines = []
    with open(filename,'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            lines.append(line)
    return lines

def FileManager(filename,optype,DataSet=None):
    if optype == 'r':
        with open(filename,'r',encoding='utf-8') as f:
            lines = [line.replace("\n","") for line in f.readlines()]
            return lines
    else:
        with open(filename,'w',encoding='utf-8') as f:
            for data in DataSet:
                f.write(data+'\n')

def sepratewords(text):
    result = ' '.join(WordPunctTokenizer().tokenize(text))
    result = result.replace("doesn ' t","does not")
    result = result.replace("don ' t","do not")
    result = result.replace("can ' t","can not")
    result = result.replace("' ve","'ve")
    result = result.replace("couldn ' t","could not")
    return result

def W2V(embeddingFile, wordBagFile):
    word_dict = dict()
    embedding = []
    words = FileManager(wordBagFile, 'r')
    embedding = FileManager(embeddingFile, 'r')
    cnt = 0
    for word in words:
        word_dict[word] = cnt
        cnt+=1
    embedding = np.array([line.split() for line in embedding])
    print("word dict length: {}".format(len(word_dict)))
    print("embedding dimention: {}".format(np.shape(embedding)))
    return word_dict, embedding

def LoadData(filename, word_dict, labellist, maxseqlen):
    data = read_tsv(filename)
    text2id = []
    label2id = []
    category2id = []
    cnt = 0
    labelmap = dict()
    for label in labellist:
        labelmap[label]=cnt
        cnt+=1
    for line in data:
        temp_text2id = []
        temp_category2id = []
        text = line[0]
        aspect_term = line[1]
        aspect_category = line[2]
        sentiment = labelmap[line[3]]
        sep_text = sepratewords(text).split()
        for word in sep_text:
            temp_text2id.append(word_dict[word])
        length = len(temp_text2id)
        if len(temp_text2id)<maxseqlen:
            temp_text2id += [0]*(maxseqlen-length)
        else:
            temp_text2id = temp_text2id[0:maxseqlen]
        text2id.append(temp_text2id)
    text2id = np.array(text2id)
    print("text shape: {}".format(np.shape(text2id)))



            



            
if __name__ == "__main__":
    word_dict, _ = W2V("../data/restaurant_embedding.data","../data/restaurant_wordbag.data")
    LoadData("../data/Restaurants15_Train.tsv", word_dict ,['negative','positive','neutral'], 80)
