import csv
def read_tsv(filename):
    with open(filename,'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        return reader


def LoadTrainAndTestFile(filename):
    return
            
if __name__ == "__main__":
    read_tsv("../data/Restaurants15_Train.tsv")
