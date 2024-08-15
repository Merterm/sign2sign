from vocabulary import *
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import os 
train_path = "Data/dgs_csl/"
for fname in os.listdir(train_path):
    #vocab_ = Vocabulary()
    if "src" in fname and "gloss" in fname and "train" in fname:
        vocab_ = Vocabulary()
        input_file = open(train_path + fname, "r", encoding='utf-8').readlines()
    

        for line in input_file:
            vocab_._from_list(line.strip().split())
        vocab_.to_file("Configs/dgs_vocab.txt")

    elif "trg" in fname and "gloss" in fname and "train" in fname:
        vocab_ = Vocabulary()
        with open(train_path + fname, 'r') as f:
            data = f.readlines()
        #input_file = open(train_path + fname, "r", encoding='utf-8').readlines()
        for line in data:
            vocab_._from_list(line.strip().split())
        vocab_.to_file("Configs/csl_vocab.txt")


