import pandas as pd
import matplotlib.pyplot as plt
import torch
import gzip
import pickle

# analyzing cross lingual time
# Sign language frame counts across sign language pairs.


class PairLoader:
    def __init__(self, pairname, train_data=False):

        data_path = "../language_pair/{}/".format(pairname)
        fname = "train" if train_data else "test"

        self.src_frames_file = data_path + "{}.src.skls".format(fname)
        self.src_sent_file = data_path + "{}.src.text".format(fname)
        self.trg_frames_file = data_path + "{}.trg.skls".format(fname)
        self.trg_sent_file = data_path + "{}.trg.text".format(fname)

        self.src_frames = self.load_skel_file(self.src_frames_file)
        self.trg_frames = self.load_skel_file(self.trg_frames_file)
        self.src_texts = self.load_text_file(self.src_sent_file)
        self.trg_texts = self.load_text_file(self.trg_sent_file)

        # evaluate the relative length (frame / text length.)
        self.eval_relative_len(self.src_frames,
                               self.trg_frames,
                               self.src_texts,
                               self.trg_texts)

    def eval_relative_len(self,
                          src_frames,
                          trg_frames,
                          src_texts,
                          trg_texts):
        # divide the length of src frames by the texts length.
        src_len = [len(item) for item in src_frames]
        trg_len = [len(item) for item in trg_frames]
        src_text_len = [len(item) for item in src_texts]
        trg_text_len = [len(item) for item in trg_texts]
        # continue divide the lengths
        src_relative_len = [src_len[i] / src_text_len[i]
                            for i in range(len(src_len))]
        trg_relative_len = [trg_len[i] / trg_text_len[i]
                            for i in range(len(trg_len))]
        # print the result
        print("src relative length: {}".format(src_relative_len))
        print("trg relative length: {}".format(trg_relative_len))

        # plot the distribution of relative length with titles in two subplots.
        # src
        plt.subplot(2, 1, 1)
        plt.hist(src_relative_len, bins=20)
        plt.title("src relative length")
        # trg
        plt.subplot(2, 1, 2)
        plt.hist(trg_relative_len, bins=20)
        plt.title("trg relative length")
        # show the plot
        plt.show()

        # describe the distribution of relative length.
        print("src relative length describe: {}".format(
            pd.Series(src_relative_len).describe()))
        print("trg relative length describe: {}".format(
            pd.Series(trg_relative_len).describe()))

    def load_skel_file(self, file_path):
        trg_size = 150
        skip_frames = 1
        # Implement your file loading logic here
        fin = open(file_path, "r").readlines()
        joint_lines = [item.split(" ") for item in fin]

        frames = []
        for trg_line in joint_lines:
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size]
                          for i in range(0, len(trg_line), trg_size*skip_frames)]
            frames.append(trg_frames)
            # return frames_data
        return frames

    def load_text_file(self, file_path):
        # handle for chinese.
        fin = open(file_path, "r").readlines()
        texts = []
        for line in fin:
            texts.append(line.split())
        return texts


"""
class DatasetLoaer is used to load the dataset. where we can load two files, 
one is the skel file, the other is the text file.
"""


def convert_data_dgs(fname="train", cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/"):
    name_file = open(cur_path + "%s.files" % fname).readlines()
    gloss_file = open(cur_path + "%s.gloss" % fname).readlines()
    text_file = open(cur_path + "%s.text" % fname).readlines()

    skel_file = open(cur_path + "%s.skels" % fname).readlines()

    file_list = []
    # write_path = (cur_path + "../SLT_data/phoenix14t.yang.%s" % fname)
    trg_size = 151
    skip_frames = 1

    lines = skel_file

    video_len = []
    gloss_len = []
    text_len = []
    for index in range(len(lines)):

        pruned_tensor = []

        trg_line = lines[index]
        # convert joints to tensors.
        trg_line = trg_line.split(" ")
        trg_line = [(float(joint) + 1e-8) for joint in trg_line]
        trg_frames = [trg_line[i:i + trg_size]
                      for i in range(0, len(trg_line), trg_size*skip_frames)]
        trg_frames = [y[:-1] for y in trg_frames]
        sign_tensor = torch.Tensor(trg_frames)
        _, ref_max_idx = torch.max(sign_tensor[:, -1], 0)
        # print(ref_max_idx)
        # if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by counter
        sign_tensor_cut = sign_tensor[:int(
            ref_max_idx) + 1, :].cpu().numpy()
        # print(type(sign_tensor_cut))
        trg_frames_cut = [y[:-1] for y in sign_tensor_cut]
        # print(len(trg_frames_cut), len(trg_frames))
        # print(glosses[index].strip())

        video_len.append(len(trg_frames))
        gloss_len.append(len(gloss_file[index].strip().split()))
        text_len.append(len(text_file[index].strip().split()))
        dict_ = {'name': name_file[index].strip(),
                 'signer': "Signer08",
                 'gloss': gloss_file[index].strip(),
                 'text': text_file[index].lower().strip(),
                 'sign': torch.Tensor(trg_frames)}

        file_list.append(dict_)

    # compute the ratio of video length divided by gloss length per instance.
    # print(video_len)
    len_ratio = [video_len[i] / text_len[i] for i in range(len(video_len))]

    # describe the distribution of len_ratio.
    print("len_ratio describe: {}".format(pd.Series(len_ratio).describe()))

    # plot the distribution of len_ratio with title in one plot.
    plt.hist(len_ratio, bins=20)
    plt.title("len_ratio for %s" % fname)
    plt.show()
    return len_ratio


# Example usage

dgs_loader = convert_data_dgs(
    fname="train", cur_path="/Users/yangzhong/Desktop/Research/MultiLingualSign/PT/ASL/")
print(dgs_loader[0])
