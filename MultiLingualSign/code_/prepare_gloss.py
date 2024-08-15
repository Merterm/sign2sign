import os 
import pandas as pd 

'''
This script is used to prepare the input file for the How2sign train/dev/test sentences sets, 
and use a trained model (text-gloss) to extract the corresponding glosses.
'''

def prepare_data():
    flist = ["train", "val", "test"]
    sent_list = []
        
    for fname in flist:
        print(fname)
        to_write = open("../how2sign/data_for_gloss_extraction/how2sign_%s_text.txt"%fname, "w")
        
        df = pd.read_csv("../how2sign/how2sign_realigned_%s.csv"%fname, sep="\t")
        # print(df)
        print(list(df))
        video_ids = list(df['VIDEO_ID'])
        sent_ids = list(df['SENTENCE_ID'])
        sents = list(df['SENTENCE'])
        for i in range(len(list(df['SENTENCE']))):
            
            to_write.write(sents[i].lower() + "\n")
            
if __name__ == "__main__":
    prepare_data()
    
