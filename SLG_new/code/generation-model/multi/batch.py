# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch
import torch.nn.functional as F

from constants import TARGET_PAD

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch,  model):

        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths  = torch_batch.src
        #print(self.src)
        self.src_gloss, self.src_gloss_lengths = torch_batch.src_gloss
        trg_gloss, trg_gloss_lengths = torch_batch.trg_gloss
        pad_index = model.pad_index 
        self.src_gloss_mask = (self.src_gloss != pad_index).unsqueeze(1)
        #self.trg_gloss_mask = (self.trg_gloss != pad_index).unsqueeze(1)
        #print(self.trg_gloss_mask.shape)
        #print(self.src_gloss.shape)
        
        #print(self.src_gloss)
        #print(self.src_gloss.shape)
        #raise EOFError
        #print("###")
        #self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        self.file_paths = torch_batch.file_paths
        self.use_cuda = model.use_cuda
        self.target_pad = TARGET_PAD
        # Just Count
        self.just_count_in = model.just_count_in
        # Future Prediction
        self.future_prediction = model.future_prediction
        
        #self.src_lengths = torch_batch.src.shape[1]
        #print(self.src_lengths)
        #self.src =  torch_batch.src
        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = torch_batch.trg.shape[1]
         #   self.trg_gloss_lengths = self.trg_gloss.shape[1]

            # trg_input is used for teacher forcing, last one is cut off
            # Remove the last frame for target input, as inputs are only up to frame N-1
            self.trg_input = trg.clone()[:, :-1,:]
            self.src_input = self.src.clone()
         #   self.src_lengths = src_lengths
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg.clone()[:, 1:, :]
            self.trg_gloss_input = trg_gloss[:, :-1]
            self.trg_gloss_length = trg_gloss_lengths - 1
            self.trg_gloss = trg_gloss[:, 1:]
            self.trg_gloss_mask = (self.trg_gloss != pad_index).unsqueeze(1)
            self.gloss_ntokens = (self.trg_gloss != pad_index).data.sum().item()
            # Just Count
            #self.trg_gloss = self.
            if self.just_count_in:
                # If Just Count, cut off the first frame of trg_input
                self.trg_input = self.trg_input[:, :, -1:]

            # Future Prediction
            if self.future_prediction != 0:
                # Loop through the future prediction, concatenating the frames shifted across once each time
                future_trg = torch.Tensor()
                # Concatenate each frame (Not counter)
                for i in range(0, self.future_prediction):
                    future_trg = torch.cat((future_trg, self.trg[:, i:-(self.future_prediction - i), :-1].clone()), dim=2)
                # Create the final target using the collected future_trg and original trg
                self.trg = torch.cat((future_trg, self.trg[:,:-self.future_prediction,-1:]), dim=2)

                # Cut off the last N frames of the trg_input
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]
            #print("##", self.trg_input.shape)
            # Target Pad is dynamic, so we exclude the padded areas from the loss computation
            trg_mask = (self.trg_input != self.target_pad).unsqueeze(1)
            src_mask = (self.src_input != self.target_pad).unsqueeze(1)
            #print("filter", src_mask.shape)
            src_pad_amount = self.src_input.shape[1] - self.src_input.shape[2]
            self.src_mask = src_mask
            self.src_mask = (self.src_input != self.target_pad)[..., 0].unsqueeze(1)
            #(F.pad(input=src_mask.double(), pad=(src_pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            # This increases the shape of the target mask to be even (16,1,120,120) -
            # adding padding that replicates - so just continues the False's or True's
            pad_amount = self.trg_input.shape[1] - self.trg_input.shape[2]
            # Create the target mask the same size as target input
            self.trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            self.ntokens = (self.trg != 0).data.sum().item()
            #print("end", self.src_mask.shape)
        if self.use_cuda:
            self._make_cuda()

    # If using Cuda
    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()
            self.src_gloss = self.src_gloss.cuda()
            self.trg_gloss_input = self.trg_gloss_input.cuda()
            self.src_gloss_lengths = self.src_gloss_lengths.cuda()
            self.src_gloss_mask = self.src_gloss_mask.cuda()
            self.trg_gloss_mask = self.trg_gloss_mask.cuda()
            self.trg_gloss = self.trg_gloss.cuda()
