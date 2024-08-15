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


def read_pair_file(name="train"):
    data_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/"
    tmp_name = "identified_paraphrase_CSL_HOW2SIGN_PHOENIX.tsv"
    if "test" in name:
        tmp_name = tmp_name.replace(".tsv", "_test.tsv")
    file_a = open(data_path + tmp_name, "r")
    pair = {}
    for line in file_a.readlines():
    # print(line)
        try:
            sent1, sent2, score, map, idx_pair = line.strip().split("\t\t")
            if map not in pair:
                pair[map.strip()] = []
            # print(sent1, sent2)
            if len(set(map.split("-"))) != 1 and float(score) > 0.55:
                pair[map.strip()].append("|".join([sent1, sent2, score, idx_pair]))
        except:
            continue
    return pair

def load_pairs(name='train', query="dgs_csl"):
    pairs = read_pair_file(name)
    dgs, csl, asl = 'PHOENIX', "CSL", "how2sign"
    # combs = ["-".join([a,b]) for a in [dgs, csl, asl] for b in [dgs, csl, asl] if a != b]
    dgs_csl = ["-".join([dgs, csl]), "-".join([csl, dgs])]
    dgs_asl =  ["-".join([dgs, asl]), "-".join([ asl, dgs])]
    csl_asl = ["-".join([csl, asl]), "-".join([ asl, csl])]
    
    mappings = {'dgs_csl': dgs_csl, "dgs_asl" : dgs_asl, "csl_asl" : csl_asl}
    target = mappings[query]
    
    list_sent_pair = []
    for pair_name in pairs:
        
        # collect the sent pairs for each langauge pair.
        if pair_name in target:
            if pair_name != target[0]:
                
                for str_ in pairs[pair_name]:
                    sent1, sent2 = str_.split("|")[1], str_.split("|")[0]
                    list_sent_pair.append((sent1, sent2))
            else:
               
                for str_ in pairs[pair_name]:
                    sent1, sent2 = str_.split("|")[0], str_.split("|")[1]
                    list_sent_pair.append((sent1, sent2))
    # print(len(list_sent_pair))
    # print(list_sent_pair[-10:])/
    return list_sent_pair  
# Retrieve pairs and group up videos.

def process_joints_into_progressive_format(seqs):
    for idx, line in enumerate(list(seqs)):
        joints = line.value
        counter_val = idx/len(list(seqs))
        # seqs.append([float(y) for y in joints])
        new_seqs += normalize([float(y) for y in joints]) 
        new_seqs += [counter_val]
    
    return new_seqs
    
    
def process_joints_into_progressive_TF(fpath):
    try:
        filein = open(fpath, "r", encoding='utf-8').readlines()
        
        seqs = []
        new_seqs = []
        for idx, line in enumerate(filein):
            joints = line.strip().split("\t")
            counter_val = idx/len(filein)
            # seqs.append([float(y) for y in joints])
            new_seqs += normalize([float(y) for y in joints]) 
            new_seqs += [counter_val]
        
        return new_seqs
    
    except:
        print("Skipped ", fpath)
        # raise EOFError
    
def load_sig_joints_asl(name='test'):
    
    if name != "test":
        for name_ in [name]:
            
            cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/"
            sent_mapping_joint = {}
            for fname in os.listdir(cur_path + "%s_all/"%name_):
                sent_name = fname.split(".")[0]
                
                try:
                    joints = process_joints_into_progressive_TF(cur_path + "%s_all/"%name_ + fname)
                    sent_mapping_joint[sent_name] = joints
                    # print(joints)
                    # print(len(joints))
                except:
                    print("Error ")
                    # break
                # break
            sign_joint_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/%s_all"
            csv_file = pd.read_csv(cur_path + "../" + "how2sign_realigned_%s.csv"%name_, sep="\t")
            sent_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), list(csv_file['SENTENCE']))}
            gloss_file = open(cur_path + "../" + "data_for_gloss_extraction/" + "pred_%s.txt"%name).readlines()
            gloss_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), [y.strip() for y in gloss_file])}
    
    else:
        cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/"
        sent_mapping_joint = {}
        
        # generated_gloss.
        
        for fname in os.listdir(cur_path + "%s_all/"%name):
            sent_name = fname.split(".")[0]
            
            try:
                joints = process_joints_into_progressive_TF(cur_path + "%s_all/"%name + fname)
                sent_mapping_joint[sent_name] = joints
                # print(joints)
            except:
                print("Error ")
                pass
            # break
        sign_joint_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/how2sign/tmp/%s_all"
        csv_file = pd.read_csv(cur_path + "../" + "how2sign_realigned_%s.csv"%name, sep="\t")
        sent_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), list(csv_file['SENTENCE']))}
        gloss_file = open(cur_path + "../" + "data_for_gloss_extraction/" + "pred_%s.txt"%name).readlines()
        gloss_mapping = {i:j for i,j in zip(list(csv_file['SENTENCE_NAME']), [y.strip() for y in gloss_file])}
    
    print("Processed ASL in total: ", len(sent_mapping_joint))
    sent_mapping = {key: value for key,value in sent_mapping.items() if key in sent_mapping_joint.keys()}
    gloss_mapping = {key: value for key,value in gloss_mapping.items() if key in sent_mapping_joint.keys()}
    
    return sent_mapping, gloss_mapping, sent_mapping_joint 

def load_sig_joints_csl(name='test'):
    
    cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/CSL/sample/"
    sent_mapping_joint = {}
    
        
    name_file = open(cur_path + "../data/" + "%s.name"%name).readlines()
    gloss_file = open(cur_path + "../data/" + "%s.gloss"%name).readlines()
    text_file = open(cur_path + "../data/" + "%s.text"%name).readlines()
    
    sent_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(text_file))}
    gloss_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(gloss_file))}
    
    count = 0
    all_possible_name = list([y.split(".")[0] for y in sent_mapping])
    for fname in os.listdir(cur_path):
        sent_name = fname.split(".")[0].replace("_JSON_OUTPUT", "")
        if not sent_name in all_possible_name:
            continue 
        else:
            count += 1
            if count % 100 == 0:
                print("processed CSL for ", count, count/len(os.listdir(cur_path)))
                
            # if count  == 100:
            #     break
            try:
                joints = process_joints_into_progressive_TF(cur_path + fname)
                sent_mapping_joint[sent_name] = joints
            except:
                print("Error in CSL")
                # break
            # break
    sent_mapping = {key: value for key,value in sent_mapping.items() if key in sent_mapping_joint.keys()}
    gloss_mapping = {key: value for key,value in gloss_mapping.items() if key in sent_mapping_joint.keys()}
    
    print("Processed CSL in total: ", len(sent_mapping_joint))
    return sent_mapping, gloss_mapping, sent_mapping_joint 

def load_sig_joints_dgs(name='test'):
    cur_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/"
        
    name_file = open(cur_path + "%s.files"%name).readlines()
    gloss_file = open(cur_path  + "%s.gloss"%name).readlines()
    text_file = open(cur_path  + "%s.text"%name).readlines()
    joint_file = open(cur_path  + "%s.skels"%name).readlines()
    
    sent_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(text_file))}
    gloss_mapping = {i.strip(): j.strip() for i,j in zip(list(name_file), list(gloss_file))}
    sent_mapping_joint = {i.strip(): j.strip() for i,j in zip(list(name_file), list(joint_file))}
    print("Processed DGS in total: ", len(sent_mapping_joint))
    # print(list(sent_mapping_joint.values())[-1])
    return sent_mapping, gloss_mapping, sent_mapping_joint 


def load_dataset_with_name(lang_name="dgs", split='test'):
    if lang_name == "dgs":
        return load_sig_joints_dgs(name=split)
    elif lang_name == "csl":
        return load_sig_joints_csl(name=split)
    else:
        return load_sig_joints_asl(name=split)
    
### retrieve pair and write into the corresponding section.
def retrieve_and_write(pair_name="dgs_csl"):
    for split in ['train']:
    # for split in ['test']:
        name_pairs = load_pairs(name=split, query=pair_name)
        name1, name2 = pair_name.split("_")
        
        # load corresponding files.
        lang1_sent, lang1_gloss, lang1_joints = load_dataset_with_name(lang_name=name1, split=split)
        lang2_sent, lang2_gloss, lang2_joints = load_dataset_with_name(lang_name=name2, split=split)
        
        all_lan1 = []
        all_lan2 = []
        for item in name_pairs:
            sent1, sent2 = item 
            
            lang1_names = []
            lang2_names = []
            # 
            # search for sent1; 
            sent1_key = [key for key, val in lang1_sent.items() if val == sent1]
            sent2_key = [key for key, val in lang2_sent.items() if val == sent2]
            # 
            # sent2.
            # print(sent1_key)
            # print(sent2_key)
            
            for a in sent1_key:
                for b in sent2_key:
                    lang1_names.append(a)
                    lang2_names.append(b)
            all_lan1 += lang1_names
            all_lan2 += lang2_names  
            
            
        print("In Total produced pairs ", pair_name, split, len((all_lan1)))
        # write files.
        language_pair_path = "/Users/yangzhong/Desktop/Research/MultiLingualSign/language_pair/%s/"%pair_name
        
        fin = open(language_pair_path + "%s.src.text"%split, "w", encoding='utf-8')
        for name in all_lan1:
            if name1 == "csl":
                fin.write(" ".join(jieba.cut(lang1_sent[name].strip(), cut_all=False)) + "\n")
            else:
                fin.write(lang1_sent[name].strip() + "\n")
            
        fin.close()
        fin = open(language_pair_path + "%s.src.gloss"%split, "w", encoding='utf-8')
        for name in all_lan1:
            if name1 == "csl":
                fin.write(" ".join(lang1_gloss[name].strip().split("\t")) + "\n")
            else:
                fin.write(lang1_gloss[name].strip() + "\n")
        fin.close()
        with open(language_pair_path + "%s.src.file"%split, "w", encoding='utf-8') as fin:
            for name in all_lan1:
                fin.write(name + "\n")
        fin.close()
        
        # with open(language_pair_path + "%s.src.skls"%split, "w") as fin:
        #     for name in all_lan1:
        #         # process gloss.
        #         target = lang1_joints[name]
        #         if type(target) != str:
        #             target_str = " ".join(["%.5f"%(y) for y in target])
        #             # print(target_str)
        #         else:
        #             target_str = target.strip()
        #         fin.write(target_str + "\n")
        # fin.close()
        
        # 2
        fin = open(language_pair_path + "%s.trg.text"%split, "w", encoding='utf-8')
        for name in all_lan2:
            if name2 == "csl":
                
                fin.write(" ".join(jieba.cut(lang2_sent[name].strip(), cut_all=False)) + "\n")
            else:
                fin.write(lang2_sent[name].strip() + "\n")
                
        fin.close()
        fin = open(language_pair_path + "%s.trg.gloss"%split, "w", encoding='utf-8')
        for name in all_lan2:
            if name2 == "csl":
                fin.write(" ".join(lang2_gloss[name].strip().split("\t")) + "\n")
            else:
                fin.write(lang2_gloss[name].strip() + "\n")
        fin.close()
        with open(language_pair_path + "%s.trg.file"%split, "w", encoding='utf-8') as fin:
            for name in all_lan2:
                fin.write(name + "\n")
        fin.close()
        
        # with open(language_pair_path + "%s.trg.skls"%split, "w") as fin:
        #     for name in all_lan2:
        #         # process gloss.
        #         target = lang2_joints[name]
        #         if type(target) != str:
        #             target_str = " ".join(["%.5f"%(y) for y in target])
        #             # print(target_str)
        #         else:
        #             target_str = target.strip()
        #         fin.write(target_str + "\n")
        # fin.close()
        
                
                 
        # write joints.
        
        
        
if __name__ == "__main__":
    # load_pairs(name='test', query="csl_asl")     
    # load_sig_joints_asl(name='test')  
    # load_sig_joints_csl(name='test')
    # load_sig_joints_dgs(name='test')
    # retrieve_and_write(pair_name="dgs_csl")
    retrieve_and_write(pair_name="dgs_asl")
    # retrieve_and_write(pair_name="csl_asl")

