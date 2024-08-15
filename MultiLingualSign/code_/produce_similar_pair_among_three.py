import os 
import numpy as np 
import pandas as pd 

def find_all_pairs(fpath):
    src_text = open(fpath + "/test.src.text", "r").readlines()
    src_title = open(fpath + "/test.src.file", "r").readlines()
    src_gloss = open(fpath + "/test.src.gloss", "r").readlines()
    trg_text = open(fpath + "/test.trg.text", "r").readlines()
    trg_title = open(fpath + "/test.trg.file", "r").readlines()
    trg_gloss = open(fpath + "/test.trg.gloss", "r").readlines()
    
    return src_text, src_title, src_gloss, trg_text, trg_title, trg_gloss 


def find_three():
    src_text_csl,  src_title_csl, src_gloss_csl, trg_text_asl, trg_title_asl, trg_gloss_asl = find_all_pairs("../language_pair/csl_asl")
    src_text_dgs,  src_title_dgs, src_gloss_dgs, trg_text_csl, trg_title_csl, trg_gloss_csl = find_all_pairs("../language_pair/dgs_csl")
    
    for idx1, text_1 in enumerate(src_text_csl):
        for idx2, text_2 in enumerate(trg_text_csl):
            if text_1.strip() == text_2.strip():
                print(trg_text_asl[idx1], " || ",  trg_gloss_asl[idx1])
                print(src_text_csl[idx1], "|| ", src_gloss_csl[idx1])
                print(src_text_dgs[idx2], " || ",src_gloss_dgs[idx2] )
                print("NAMES")
                print(src_title_csl[idx1])
                print(trg_title_asl[idx1])
                print(src_title_dgs[idx2])
                print()
        # break
      
if __name__ == "__main__":
    find_three()