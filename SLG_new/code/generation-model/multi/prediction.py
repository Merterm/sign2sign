
import numpy as np
import math

import torch
from torchtext.data import Dataset

from helpers import bpe_postprocess, load_config, get_latest_checkpoint, \
    load_checkpoint, calculate_dtw
from model import build_model, Model, TranslationModel, build_translation_model
from batch import Batch
from data import load_data, make_data_iter
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     model2: TranslationModel,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     recognition_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None):

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=True, train=False)

    #pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    model2.eval()
    # don't track gradients during validation
    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(torch_batch=valid_batch,
                          #pad_index = pad_index,
                          model = model)
            targets = batch.trg

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss, _ = model.get_loss_for_batch(
                    batch, loss_function=loss_function, recognition_loss_function=recognition_function)
                #print(batch.src_gloss)
                #translation_loss, _, _, _ = model2(return_type="loss",
                 #                               src=batch.src_gloss,
                  #                              trg_input=batch.trg_gloss,
                   #                             src_mask=src_gloss_mask,
                    #                            src_length=src_gloss_lengths,
                     #                           trg_mask=trg_gloss_mask)
               
               # print("valid batch loss", batch_loss)
                translation_loss, _, _, _ = model2(return_type="loss",
                                                src_gloss=batch.src_gloss,
                                                trg_gloss_input=batch.trg_gloss_input,
                                                trg_gloss = batch.trg_gloss,
                                                src_gloss_mask=batch.src_gloss_mask,
                                                src_gloss_lengths=batch.src_gloss_lengths,
                                                trg_gloss_mask=batch.trg_gloss_mask)
                valid_loss += (batch_loss + translation_loss)
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # If not just count in, run inference to produce translation videos
            if not model.just_count_in:
                # Run batch through the model in an auto-regressive format
                gloss_output, output, attention_scores = model.run_batch(
                                            batch=batch,
                                            max_output_length=max_output_length)
                #print(gloss_output)
                #raise EOFError
            # If future prediction
            if model.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                train_output = torch.cat((train_output[:, :, :train_output.shape[2] // (model.future_prediction)], train_output[:, :, -1:]),dim=2)
                # Cut to only the first frame prediction + add the counter
                targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),dim=2)

            # For just counter, the inference is the same as GTing
            if model.just_count_in:
                output = train_output

            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            #print(targets.shape)
            #print(output.shape)
            # Add the source sentences to list, by using the model source vocab and batch indices
    #        valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
     #                            range(len(batch.src))])
            valid_inputs.extend(['dummy'] * targets.shape[0])
            # Calculate the full Dynamic Time Warping score - for evaluation
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            # Can set to only run a few batches
            # if batches == math.ceil(20/batch_size):
            #     break
            batches += 1

        # Dynamic Time Warping scores
        current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths
