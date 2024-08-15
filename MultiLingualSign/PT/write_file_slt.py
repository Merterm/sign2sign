import torch 
import pickle 
import gzip
import sys
import argparse
import os 

def convert_data(name="dev", cur_path="", target_path="../../../slt/new_data/"):
    trg_size = 151
    skip_frames = 1
    print(name)
    isExist = os.path.exists(path)
    print(target_path)
    if not isExist:
  
    # Create a new directory because it does not exist 
    os.makedirs(path)
    print("The new directory is created!")


    glosses = open("%s.gloss"%name, "r").readlines()
    textes = open("%s.text"%name, "r").readlines()
    names = open("%s.files"%name, "r").readlines()
    # write folder.
    write_path = (target_path + "phoenix14t.yang.%s"%name)
    file_list = []

    with open("%s.skels"%name, "r") as f_in:
        lines = f_in.readlines()
        for index in range(len(lines)):
            print(index)
            trg_line = lines[index]
            # convert joints to tensors.
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]
            # remove the last counter.
            
            trg_frames = [y[:-1] for y in trg_frames]
            # print(glosses[index].strip())
            dict_ = {'name': names[index].strip(),
                    'signer': "Signer08",
                    'gloss': glosses[index].strip(),
                    'text': textes[index].lower().strip(),
                    'sign': torch.Tensor(trg_frames)}
            
            file_list.append(dict_)
    f = gzip.open(write_path, 'wb')
    pickle.dump(file_list, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare Data for SLT")
    # parser.add_argument("--name", default="dev", type=str,
    #                         help="Data source on dev/test/train.")
    parser.add_argument("--cur_path", default="Data/train_gloss_between_no_repeat", type=str,
                            help="Data source on dev/test/train.")
                                               
    args = parser.parse_args()

    convert_data('dev', args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    convert_data("test", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1]))
    convert_data("train", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1]))
    # trg_size = 151
    # skip_frames = 1
    # gold = open("test.skels", "r").readlines()
    # with open("../../output_gloss2P.skels", "r") as f_in:
    #     lines = f_in.readlines()
    #     for index in range(len(lines)):
    #         trg_line = lines[index].strip()

    #         gold_line = gold[index].strip()
    #         # convert joints to tensors.
    #         trg_line = trg_line.split(" ")
    #         trg_line = [(float(joint) + 1e-8) for joint in trg_line]
    #         trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

    #         gold_line = gold_line.split(" ")
    #         # print(gold_line)
    #         gold_line = [(float(joint) + 1e-8) for joint in gold_line]
    #         gold_frames = [gold_line[i:i + trg_size] for i in range(0, len(gold_line), trg_size*skip_frames)]
    #         print(torch.Tensor(trg_frames).shape)
    #         # print(gold_frames)
    #         print(torch.Tensor(gold_frames).shape)
    #         print("###")