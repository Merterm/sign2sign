import os 
import pandas as pd 

def keyword(string):
    keywords = ['rain', 'sunny', 'hot', 'degree', 'weather', 'wind', 'blow', 'sun', 'temperature']
    for key in keywords:
        if key in string.split():
            return True 
    return False

def find_bobsl_weather():
    flist = os.listdir("../bobsl/subtitles/audio-aligned")
    flist_manual = os.listdir("../bobsl/subtitles/manually-aligned")
    sent_list = []
    total = 0
    to_write = open("../bobsl/bobsl_test_sent.tsv", "w")
    print(len(flist))
    for fname in flist_manual:
        # print(fname)
        
        if ".DS_Store" in fname or "weather" in fname:
            continue 
        if fname not in flist_manual:
            continue 
        
        lines = open("../bobsl/subtitles/manually-aligned/" + fname).readlines()[2:]
        time_line = ""
        for line in lines:
            
            if len(line.strip()) == 0:
                continue 
            elif "-->" in line:
                time_line = line 
                print(time_line)
                continue 
            total += 1
            # if keyword(line.strip()) is True:
            sent_list.append(line)
            to_write.write(fname + "\t" + time_line.strip() + "\t" + line)  
    print(len(sent_list))
    print("Total Sentence :", total)
    
def find_how2sign_weather():
    flist = os.listdir("../how2sign")
    sent_list = []
    
    to_write = open("../how2sign/how2sign_test_sent.tsv", "w")
    to_write.write("FNAME" + "\t" + "VIDEO_ID" + "\t" + "SENTENCE_ID" + "\t" + "SENTENCE" + "\n")
    for fname in flist:
        print(fname)
        if ".DS_Store" in fname or "train" in fname or "dev" in fname:
            continue 
        df = pd.read_csv("../how2sign/" + fname, sep="\t")
        # print(df)
        print(list(df))
        video_ids = list(df['VIDEO_ID'])
        sent_ids = list(df['SENTENCE_ID'])
        sents = list(df['SENTENCE'])
        for i in range(len(list(df['SENTENCE']))):
            # if keyword(sent):
            
            to_write.write(fname + "\t" + video_ids[i] + "\t" + sent_ids[i] + "\t" + sents[i] + "\n")
    # print(len(sent_list))
                
    
        
        # break
    
if __name__ == "__main__":
    # find_how2sign_weather()  
    find_bobsl_weather()  
        