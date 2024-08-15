import csv
import torch 
import pickle
import gzip 
import sys 
import argparse 
import os 
import math 
import numpy as np 
import pandas as pd 
import jieba
from tqdm import tqdm 
from nltk.tokenize import word_tokenize 

def process_joints_into_progressive_format(seqs):
    new_seqs = []
    for idx, line in enumerate(list(seqs)):
        joints = line
        counter_val = idx/len(list(seqs))
        # seqs.append([float(y) for y in joints])
        new_seqs += normalize([float(y) for y in joints]) 
        new_seqs += [counter_val]
    
    return new_seqs

def normalize(joints):
    n_ref = [1e-8, 1e-8, 1e-8]
    left_shoulder = [0.16732784, 0.00540292,0.01452863]
    right_shoulder = [-0.16523061,  0.00936848,  0.02367572]
    
    cur_left_shoulder = joints[6:9]
    cur_right_shoulder = joints[15:18]
    
    
    cur_dist  = math.hypot(cur_left_shoulder[0] - cur_right_shoulder[0], cur_left_shoulder[1] - cur_right_shoulder[1])
    ori_dist = math.hypot(left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1])
    factor = ori_dist / cur_dist 
    
    modified_joints = []
    
    for i in range(0, len(joints), 3):
        for val in range(3):
            modified_joints.append(joints[i+val] + ( n_ref[val] - joints[3+val]))
       
    new_joints = []
    for i in range(0, len(joints), 3):
        for val in range(3):
            new_joints.append(n_ref[val] +  (modified_joints[i+val] - n_ref[val]) * factor)
    return  new_joints


def normalizeChString(s):
    s = "".join(s.split())
    s = ' '.join(str(x) for x in s)
    return s.strip()

def convert_data_CSL(fname='dev', cur_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/CSL/sample/"):
    
    # create the mapping.
    name_file = open(cur_path + "../data/" + "%s.name"%fname).readlines()
    gloss_file = open(cur_path + "../data/" + "%s.gloss"%fname).readlines()
    text_file = open(cur_path + "../data/" + "%s.text"%fname).readlines()
    
    sent_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(text_file))}
    gloss_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(gloss_file))}
    print(len(sent_mapping))
    print(sent_mapping)
    
    names = os.listdir(cur_path)
    
    no_duplicate = []
    file_list = []
    for i in tqdm(range(len(names))):
        name = names[i]
        new_name = name.replace("_JSON_OUTPUT", "")
        print("###", new_name)
        if new_name in no_duplicate:
            print("processed ", new_name)
            continue 
        elif new_name.split(".")[0] not in list(sent_mapping):
            print("not in current split", new_name)
            continue 
        else:
            no_duplicate.append(new_name)
            print("Start")
        
            try:
                filein = open(cur_path +name, "r", encoding='utf-8').readlines()
            except:
                print("Skipped ", name)
                continue
            cur_name = new_name.split(".")[0]
            
            seqs = []
            new_seqs = []
            for line in filein:
                joints = line.strip().split("\t")
                seqs.append([float(y) for y in joints])
                new_seqs.append(normalize([float(y) for y in joints]))
            
            new_seqs = np.asanyarray(new_seqs)
            sign_tensor = torch.Tensor(new_seqs)
            
            word_list = normalizeChString(sent_mapping[new_name.split(".")[0]])
            gloss_list =" ".join(gloss_mapping[new_name.split(".")[0]].split("\t"))                       
            dict_ = {'name': cur_name.strip(),
                    'signer': "Signer01",
                    'gloss': gloss_list,
                    'text': word_list,
                    'sign': sign_tensor}
            print(dict_)
        # raise EOFError
        
        file_list.append(dict_)
    print("In total processed", len(file_list))
    f = gzip.open("../SLT_data/CSL_no_split.yang.%s"%fname, "wb")
    pickle.dump(file_list, f)  
 
def convert_data_CSL_PT(fname='dev', cur_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/CSL/sample/"):
    
    # create the mapping.
    name_file = open(cur_path + "../data/" + "%s.name"%fname).readlines()
    gloss_file = open(cur_path + "../data/" + "%s.gloss"%fname).readlines()
    text_file = open(cur_path + "../data/" + "%s.text"%fname).readlines()
    
    sent_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(text_file))}
    gloss_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(gloss_file))}
    print(len(sent_mapping))
    print(sent_mapping)
    
    names = os.listdir(cur_path)
    
    
    language_pair_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/CSL/"
        
    fin1 = open(language_pair_path + "%s.text"%fname, "w", encoding='utf-8')
    fin2 = open(language_pair_path + "%s.files"%fname, "w", encoding='utf-8')
    fin3 = open(language_pair_path + "%s.gloss"%fname, "w", encoding='utf-8')
    fin4 = open(language_pair_path + "%s.skels"%fname, "w", encoding='utf-8')
    no_duplicate = []
    file_list = []
    for i in tqdm(range(len(names))):
        name = names[i]
        new_name = name.replace("_JSON_OUTPUT", "")
        print("###", new_name)
        if new_name in no_duplicate:
            print("processed ", new_name)
            continue 
        elif new_name.split(".")[0] not in list(sent_mapping):
            print("not in current split", new_name)
            continue 
        else:
            no_duplicate.append(new_name)
            print("Start")
        
            try:
                filein = open(cur_path +name, "r", encoding='utf-8').readlines()
            except:
                print("Skipped ", name)
                continue
            cur_name = new_name.split(".")[0]
            
            seqs = []
            new_seqs = []
            for line in filein:
                joints = line.strip().split("\t")
                seqs.append([float(y) for y in joints])
                new_seqs.append(normalize([float(y) for y in joints]))
            
            new_seqs = np.asanyarray(new_seqs)
            sign_tensor = torch.Tensor(new_seqs)
            
            word_list = normalizeChString(sent_mapping[new_name.split(".")[0]])
            gloss_list =" ".join([y.strip() for y in gloss_mapping[new_name.split(".")[0]].split("\t")])                       
            dict_ = {'name': cur_name.strip(),
                    'signer': "Signer01",
                    'gloss': gloss_list,
                    'text': word_list,
                    'sign': sign_tensor}
            print(dict_)
            
            word_list = " ".join(jieba.cut(word_list, cut_all=False)) + "\n"
            fin1.write(word_list)
            fin2.write(cur_name.strip() + "\n")
            fin3.write(gloss_list + "\n")
            processed_tensor = process_joints_into_progressive_format(sign_tensor) 
            assert(len(processed_tensor) // 151 == len(processed_tensor) / 151)
            fin4.write(" ".join(["%.5f"%(y) for y in processed_tensor])+ "\n")
    
    fin1.close()
    fin2.close()
    fin3.close()
    fin4.close()
    
def convert_data_how2sign_PT(fname='dev', cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/"):
    # if fname == "dev":
    #     fname = 'val'
    #     special = "dev"
        
    file_list = []
    
    # create the mapping.
    if fname == "dev":
        csv_file = pd.read_csv(cur_path + "../" + "how2sign_realigned_%s.csv"%"dev", sep="\t")
    else:
        csv_file = pd.read_csv(cur_path + "../" + "how2sign_realigned_%s.csv"%fname, sep="\t")
    name_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), list(csv_file['SENTENCE']))}
    
    # generated_gloss.
    gloss_file = open(cur_path + "../" + "data_for_gloss_extraction/" + "pred_%s.txt"%fname).readlines()
    gloss_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), [y.strip() for y in gloss_file])}
    
    if fname == "val":
        fname = 'valid'
    names = os.listdir(cur_path + fname + "_all/")
    
    no_duplicate = []
    
    language_pair_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/ASL/"
        
    fin1 = open(language_pair_path + "%s.text"%fname, "w", encoding='utf-8')
    fin2 = open(language_pair_path + "%s.files"%fname, "w", encoding='utf-8')
    fin3 = open(language_pair_path + "%s.gloss"%fname, "w", encoding='utf-8')
    fin4 = open(language_pair_path + "%s.skels"%fname, "w", encoding='utf-8')
    
    # for name in all_lan1:
    #     if name1 == "csl":
    #         fin.write(" ".join(jieba.cut(lang1_sent[name].strip(), cut_all=False)) + "\n")
    #     else:
    #         fin.write(lang1_sent[name].strip() + "\n")
        
    # fin.close()
    # fin = open(language_pair_path + "%s.src.gloss"%split, "w", encoding='utf-8')
    # for name in all_lan1:
    #     if name1 == "csl":
    #         fin.write(" ".join(lang1_gloss[name].strip().split("\t")) + "\n")
    #     else:
    #         fin.write(lang1_gloss[name].strip() + "\n")
    # fin.close()
    # with open(language_pair_path + "%s.src.file"%split, "w", encoding='utf-8') as fin:
    #     for name in all_lan1:
    #         fin.write(name + "\n")
    # fin.close()
    
    for i in tqdm(range(len(names))):
        name = names[i]
        if name in no_duplicate:
            continue 
        else:
            no_duplicate.append(name)
            
        try:
            filein = open(cur_path + fname + "_all/" +name, "r", encoding='utf-8').readlines()
        except:
            print("Skipped ", name)
            continue
        cur_name = name.split(".")[0]
        
        seqs = []
        new_seqs = []
        for line in filein:
            joints = line.strip().split("\t")
            seqs.append([float(y) for y in joints])
            new_seqs.append(normalize([float(y) for y in joints]))
        
        new_seqs = np.asanyarray(new_seqs)
        sign_tensor = torch.Tensor(new_seqs)
        
        word_list = " ".join(word_tokenize(name_mapping[cur_name].lower().strip()))
        gloss_list = gloss_mapping[cur_name]
        
        if gloss_list.strip() == "" or len(gloss_list.split()) <= 2:
            gloss_list = word_list.upper()
        print(gloss_list.split())
        dict_ = {'name': cur_name.strip(),
                 'signer': "Signer01",
                 'gloss': gloss_list,
                 'text': word_list,
                 'sign': sign_tensor}
        
        
        fin1.write(word_list + "\n")
        fin2.write(cur_name.strip() + "\n")
        fin3.write(gloss_list + "\n")
        processed_tensor = process_joints_into_progressive_format(sign_tensor) 
        assert(len(processed_tensor) // 151 == len(processed_tensor) / 151)
        fin4.write(" ".join(["%.5f"%(y) for y in processed_tensor])+ "\n")
        
        # break
        
    fin1.close()
    fin2.close()
    fin3.close()
    fin4.close()
       
def convert_data_how2sign(fname='dev', cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/"):
    if fname == "dev":
        fname = 'val'
        
    file_list = []
    
    # create the mapping.
    csv_file = pd.read_csv(cur_path + "../" + "how2sign_realigned_%s.csv"%fname, sep="\t")
    name_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), list(csv_file['SENTENCE']))}
    
    # generated_gloss.
    gloss_file = open(cur_path + "../" + "data_for_gloss_extraction/" + "pred_%s.txt"%fname).readlines()
    gloss_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), [y.strip() for y in gloss_file])}
    
    if fname == "val":
        fname = 'valid'
    names = os.listdir(cur_path + fname + "_all/")
    
    no_duplicate = []
    for i in tqdm(range(len(names))):
        name = names[i]
        if name in no_duplicate:
            continue 
        else:
            no_duplicate.append(name)
            
        try:
            filein = open(cur_path + fname + "_all/" +name, "r", encoding='utf-8').readlines()
        except:
            print("Skipped ", name)
            continue
        cur_name = name.split(".")[0]
        
        seqs = []
        new_seqs = []
        for line in filein:
            joints = line.strip().split("\t")
            seqs.append([float(y) for y in joints])
            new_seqs.append(normalize([float(y) for y in joints]))
        
        new_seqs = np.asanyarray(new_seqs)
        sign_tensor = torch.Tensor(new_seqs)
        
        word_list = " ".join(word_tokenize(name_mapping[cur_name].lower().strip()))
        gloss_list = gloss_mapping[cur_name]
        
        if gloss_list.strip() == "" or len(gloss_list.split()) <= 2:
            gloss_list = word_list.upper()
        print(gloss_list.split())
        dict_ = {'name': cur_name.strip(),
                 'signer': "Signer01",
                 'gloss': gloss_list,
                 'text': word_list,
                 'sign': sign_tensor}
        print(dict_)
        
        
        file_list.append(dict_)
    
    f = gzip.open("../SLT_data/how2sign.yang.%s"%fname, "wb")
    pickle.dump(file_list, f)

def convert_data_dgs(fname="dev", cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/"):
    name_file = open(cur_path +   "%s.files"%fname).readlines()
    gloss_file = open(cur_path +  "%s.gloss"%fname).readlines()
    text_file = open(cur_path + "%s.text"%fname).readlines()
    
    skel_file = open(cur_path + "%s.skels"%fname).readlines()
    
    file_list = []
    write_path = (cur_path + "../SLT_data/phoenix14t.yang.%s"%fname)
    trg_size = 151
    skip_frames = 1
    with open(write_path, "w") as f_in:
        lines = skel_file
        for index in range(len(lines)):

            pruned_tensor = []

            
            trg_line = lines[index]
            # convert joints to tensors.
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]
            trg_frames = [y[:-1] for y in trg_frames]
            sign_tensor = torch.Tensor(trg_frames)
            _ , ref_max_idx = torch.max(sign_tensor[:, -1], 0)
            # print(ref_max_idx)
            # if ref_max_idx == 0: ref_max_idx += 1
            # Cut down frames by counter
            sign_tensor_cut = sign_tensor[:int(ref_max_idx) + 1,:].cpu().numpy()
            # print(type(sign_tensor_cut))
            trg_frames_cut = [y[:-1] for y in sign_tensor_cut]
            # print(len(trg_frames_cut), len(trg_frames))
            # print(glosses[index].strip())
            dict_ = {'name': name_file[index].strip(),
                    'signer': "Signer08",
                    'gloss': gloss_file[index].strip(),
                    'text': text_file[index].lower().strip(),
                    'sign': torch.Tensor(trg_frames)}
            
            file_list.append(dict_)
    print(len(file_list))
    f = gzip.open("../SLT_data/dgs.yang.%s"%fname, "wb")
    pickle.dump(file_list, f)
    
def check_unique(fname='dev',cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/"):
    if fname == "dev":
        fname = 'val'
    if fname == "val":  
        fname = 'valid'
    file_list = []
    
    names = os.listdir(cur_path + fname + "_all/")
    print(fname)
    print(len(set(names)))
    
if __name__ == "__main__":
    # convert_data_how2sign(fname='dev')
    # convert_data_how2sign(fname='train')
    # convert_data_how2sign(fname='test')
    
    # check_unique(fname='dev')
    # check_unique(fname='train')
    # check_unique(fname='test')
    
    # convert_data_CSL(fname='dev')
    # convert_data_CSL(fname='train')
    # convert_data_CSL(fname='test')
    
    # convert_data_dgs(fname='train')
    # convert_data_dgs(fname='dev')
    
    # convert_data_dgs(fname='test')
    
    convert_data_CSL_PT(fname='train')
    convert_data_CSL_PT(fname='dev')
    convert_data_CSL_PT(fname='test')
    
    convert_data_how2sign_PT(fname='train')
    convert_data_how2sign_PT(fname='dev')
    convert_data_how2sign_PT(fname='test')