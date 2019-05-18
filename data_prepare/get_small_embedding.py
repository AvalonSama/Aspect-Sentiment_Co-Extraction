from utils import read_tsv
from nltk.tokenize import WordPunctTokenizer

filelist = []
bigEmbedding = dict()

def sepratewords(text):
    result = ' '.join(WordPunctTokenizer().tokenize(text))
    result = result.replace("doesn 't","does not")
    result = result.replace("don 't","do not")
    result = result.replace("can 't","can not")
    result = result.replace("' ve","'ve")
    result = result.replace("couldn 't","could not")
    return result

with open("", 'r', encoding='utf-8') as f:
    data = [line.replace("\n","") for line in f.readlines()]
    for line in data:
        line = line.split()
        word =' '.join(line.split()[:-300])
        embedding = [num for num in line.split()[-300:]]
        bigEmbedding[word] = embedding

for filename in filelist:
    data = read_tsv(filename)
    for line in data:
        text = line[0]
        sep_text = 
        
