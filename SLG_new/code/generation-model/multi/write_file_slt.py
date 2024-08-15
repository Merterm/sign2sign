import torch 
import pickle 
import gzip
import sys
import argparse
import os 

def convert_data(name="test", cur_path="", output_skeleton_path="",  target_path="../slt/new_data/"):
    trg_size = 151
    skip_frames = 1
    print(name)
    isExist = os.path.exists(target_path)
    print(target_path)
    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(target_path)
        print("The new directory is created!")

    _, lan1, lan2 = output_skeleton_path.split(".")[0].split("_")
    if os.path.exists("Data/%s_%s"%(lan1, lan2)):
        cur_path = "Data/%s_%s"%(lan1, lan2)

        glosses = open(cur_path + '/' + "%s.trg.gloss"%name, "r",encoding='utf-8').readlines()
        textes = open(cur_path + '/'+ "%s.trg.text"%name, "r", encoding='utf-8').readlines()
        names = open(cur_path + '/'+ "%s.trg.file"%name, "r").readlines()
    # write folder.
    elif os.path.exists("Data/%s_%s"%(lan2, lan1)):
        print("reversed")
        cur_path = "Data/%s_%s"%(lan2, lan1)
        print(cur_path)
        glosses = open(cur_path + '/' + "%s.src.gloss"%name, "r", encoding='utf-8').readlines()
        textes = open(cur_path + '/'+ "%s.src.text"%name, "r", encoding='utf-8').readlines()
        names = open(cur_path + '/'+ "%s.src.file"%name, "r").readlines()
    print(len(glosses))
    print(len(textes))
    #raise EOFError
    write_path = target_path + "%s.yang.test"%output_skeleton_path.split(".")[0]

    #train_write_path = (target_path + "phoenix14t.yang.model.%s"%name)
    file_list = []
    
    #file_train_list = []

    #if name == "test" or name == "dev":
     #   output_skeleton_path = "pred_%s_Base_%s.skels"%(name, cur_path.split("/")[1].replace("_", "."))
      #  if "PT" in cur_path:
       #     output_skeleton_path = "pred_%s_Base_Gloss2P_counter.skels"%(name)
    #else:
     #   output_skeleton_path = cur_path + '/'+ "%s.skels"%name
        
    #train_output_skeleton_path = cur_path + '/'+ "%s.skels"%name

    all_tensors = []
    with open(output_skeleton_path, "r") as f_in:
        lines = f_in.readlines()
        for index in range(len(lines)):

            pruned_tensor = []

            #print(glosses[index])
            trg_line = lines[index]
            # convert joints to tensors.
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

            sign_tensor = torch.Tensor(trg_frames)
            _ , ref_max_idx = torch.max(sign_tensor[:, -1], 0)
            # if ref_max_idx == 0: ref_max_idx += 1
            # Cut down frames by counter
            sign_tensor_cut = sign_tensor[:ref_max_idx+1,:].cpu().numpy()

            trg_frames_cut = [y[:-1] for y in sign_tensor_cut]
            print(len(trg_frames_cut), len(trg_frames))
            print(torch.Tensor(trg_frames_cut).shape)# print(glosses[index].strip())
            dict_ = {'name': names[index].strip()+str(index),
                    'signer': "Signer08",
                    'gloss': glosses[index].strip(),
                    'text': textes[index].lower().strip(),
                    'sign': torch.Tensor(trg_frames_cut)}
            
            file_list.append(dict_)
    print("In Total", len(file_list)) 
    # with open(train_output_skeleton_path, "r") as f_in:
    #     lines = f_in.readlines()
    #     for index in range(len(lines)):
    #         print(glosses[index])
    #         trg_line = lines[index]
    #         # convert joints to tensors.
    #         trg_line = trg_line.split(" ")
    #         trg_line = [(float(joint) + 1e-8) for joint in trg_line]
    #         trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]
    #         # remove the last counter.
            
    #         trg_frames = [y[:-1] for y in trg_frames]
    #         # print(glosses[index].strip())
    #         dict_ = {'name': names[index].strip(),
    #                 'signer': "Signer08",
    #                 'gloss': glosses[index].strip(),
    #                 'text': textes[index].lower().strip(),
    #                 'sign': torch.Tensor(trg_frames)}
            
    #         file_train_list.append(dict_)
    f = gzip.open(write_path, 'wb')
    pickle.dump(file_list, f)

    # z = gzip.open(train_write_path, 'wb')
    # pickle.dump(file_train_list, z)


def convert_data_src(name="test", cur_path="", output_skeleton_path="",  target_path="../slt/new_pipeline_data/"):
    #f = gzip.open(write_path, 'wb')
    #pickle.dump(file_list, f)
    trg_size = 151
    skip_frames = 1
    print(name)
    isExist = os.path.exists(target_path)
    print(target_path)
    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(target_path)
        print("The new directory is created!")

    _, lan1, lan2 = output_skeleton_path.split(".")[0].split("_")
    if os.path.exists("Data/%s_%s"%(lan1, lan2)):
        cur_path = "Data/%s_%s"%(lan1, lan2)

        glosses = open(cur_path + '/' + "%s.src.gloss"%name, "r",encoding='utf-8').readlines()
        textes = open(cur_path + '/'+ "%s.src.text"%name, "r", encoding='utf-8').readlines()
        names = open(cur_path + '/'+ "%s.src.file"%name, "r").readlines()
        skel_path = cur_path + "/" + "%s.src.skls"%name
    # write folder.
    elif os.path.exists("Data/%s_%s"%(lan2, lan1)):
        print("reversed")
        cur_path = "Data/%s_%s"%(lan2, lan1)
        print(cur_path)
        glosses = open(cur_path + '/' + "%s.trg.gloss"%name, "r", encoding='utf-8').readlines()
        textes = open(cur_path + '/'+ "%s.trg.text"%name, "r", encoding='utf-8').readlines()
        names = open(cur_path + '/'+ "%s.trg.file"%name, "r").readlines()
        skel_path = cur_path + "/" + "%s.trg.skls"%name
    print(len(glosses))
    print(len(textes))
    #raise EOFError
    write_path = target_path + "%s.yang.test"%output_skeleton_path.split(".")[0]

    #train_write_path = (target_path + "phoenix14t.yang.model.%s"%name)
    file_list = []
    
    #file_train_list = []

    #if name == "test" or name == "dev":
     #   output_skeleton_path = "pred_%s_Base_%s.skels"%(name, cur_path.split("/")[1].replace("_", "."))
      #  if "PT" in cur_path:
       #     output_skeleton_path = "pred_%s_Base_Gloss2P_counter.skels"%(name)
    #else:
     #   output_skeleton_path = cur_path + '/'+ "%s.skels"%name
        
    #train_output_skeleton_path = cur_path + '/'+ "%s.skels"%name

    all_tensors = []
    
    with open(skel_path, "r") as f_in:
        lines = f_in.readlines()
        for index in range(len(lines)):

            pruned_tensor = []

            #print(glosses[index])
            trg_line = lines[index]
            # convert joints to tensors.
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

            sign_tensor = torch.Tensor(trg_frames)
            _ , ref_max_idx = torch.max(sign_tensor[:, -1], 0)
            # if ref_max_idx == 0: ref_max_idx += 1
            # Cut down frames by counter
            sign_tensor_cut = sign_tensor[:ref_max_idx+1,:].cpu().numpy()

            trg_frames_cut = [y[:-1] for y in sign_tensor_cut]
            print(len(trg_frames_cut), len(trg_frames))
            print(torch.Tensor(trg_frames_cut).shape)# print(glosses[index].strip())
            dict_ = {'name': names[index].strip()+str(index),
                    'signer': "Signer08",
                    'gloss': glosses[index].strip(),
                    'text': textes[index].lower().strip(),
                    'sign': torch.Tensor(trg_frames_cut)}
            
            file_list.append(dict_)
    print("In Total", len(file_list)) 
    f = gzip.open(write_path, 'wb')
    pickle.dump(file_list, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare Data for SLT")
    # parser.add_argument("--name", default="dev", type=str,
    #                         help="Data source on dev/test/train.")
    parser.add_argument("--cur_path", default="Data/train_gloss_between_no_repeat", type=str,
                            help="Data source on dev/test/train.")
                                               
    args = parser.parse_args()

    #convert_data('dev', args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    #convert_data("test", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    #convert_data("train", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    convert_data_src("test", output_skeleton_path="output_csl_dgs.skels")
    convert_data_src("test", output_skeleton_path="output_dgs_csl.skels")
    convert_data_src("test", output_skeleton_path="output_asl_csl.skels")
    convert_data_src("test", output_skeleton_path="output_csl_asl.skels")
    convert_data_src("test", output_skeleton_path="output_dgs_asl.skels")
    convert_data_src("test", output_skeleton_path="output_asl_dgs.skels")
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
