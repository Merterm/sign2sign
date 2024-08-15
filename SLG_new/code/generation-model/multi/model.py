# coding: utf-8
"""
Module to represents whole models
"""
import tensorflow as tf
#tf.config.set_visible_devices([1], "GPU")
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Tuple
from itertools import groupby
from helpers import freeze_params
from initialization import initialize_model
from embeddings import Embeddings
from encoders import Encoder, TransformerEncoder
from decoders import Decoder, TransformerDecoder, TransformerDecoderWord
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from search import greedy
from vocabulary import Vocabulary
from batch import Batch
from loss import XentLoss

class TranslationModel(nn.Module):
    """
    translation model for gloss to gloss.
    """
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: Embeddings,
            trg_embed: Embeddings,
            src_vocab: Vocabulary,
            trg_vocab: Vocabulary,
    ) -> None:
        super().__init__()
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder 
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab 
        self.pad_index = self.trg_vocab.pad_index
        self.bos_index = self.trg_vocab.bos_index
        self.eos_index = self.trg_vocab.eos_index
        self.unk_index = self.trg_vocab.unk_index 
        self._loss_function = None     
    
    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, cfg:Tuple):
        loss_type, label_smoothing= cfg
        self._loss_function = XentLoss(pad_index=self.pad_index,
                                       smoothing=label_smoothing).cuda()

    def forward(self,
            return_type: str = None,
            **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if return_type == "loss":
            assert self.loss_function is not None 
            #assert "trg" in kwargs and "trg_mask" in kwargs 
            out, _, _, _ = self._encode_decode(**kwargs)
            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)
            #print(kwargs)
            batch_loss = self._loss_function(log_probs, targets=kwargs['trg_gloss'])
            # count correct tokens.
            trg_mask = kwargs["trg_gloss_mask"].squeeze(1)
            assert kwargs['trg_gloss'].size() == trg_mask.size()
            n_correct = torch.sum(
                                    log_probs.argmax(-1).masked_select(trg_mask).eq(
                                                            kwargs["trg_gloss"].masked_select(trg_mask)))
            return_tuple = (batch_loss, log_probs, None, n_correct)
        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(**kwargs)
            return_tuple = (encoder_output, encoder_hidden, None, None)
        elif return_type == "decode":
            otputs, hidden, att_probs, att_vectors = self._decode(**kwargs)
            return_tuple = (outputs, hidden, att_probs, att_vectors)
        return tuple(return_tuple)


    def _encode_decode(
            self,
            src_gloss: Tensor,
            trg_gloss_input: Tensor,
            src_gloss_mask: Tensor,
            src_gloss_lengths: Tensor,
            trg_gloss_mask: Tensor = None,
            **kwargs,
    ) -> Tensor:
        encoder_output, encoder_hidden = self._encode(src_gloss, src_length=src_gloss_lengths, src_mask=src_gloss_mask)
        unroll_steps = trg_gloss_input.size(1)
        print(trg_gloss_input.shape)
        print(trg_gloss_mask.shape)
        print(src_gloss_mask.shape)
        return self._decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_gloss_mask,
                trg_input=trg_gloss_input,
                unroll_steps=unroll_steps,
                trg_mask=trg_gloss_mask,
        )

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor, **_kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        src_input = src if self.src_embed is None else self.src_embed(src)
        return self.encoder(src_input, src_length, src_mask, **_kwargs)

    def _decode(
            self,
            encoder_output: Tensor,
            encoder_hidden: Tensor,
            src_mask: Tensor,
            trg_input: Tensor,
            unroll_steps: int,
            decoder_hidden: Tensor = None,
            att_vector: Tensor=None,
            trg_mask: Tensor=None,
            **_kwargs,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.decoder(
                trg_embed=self.trg_embed(trg_input),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask = src_mask,
                unroll_steps=unroll_steps,
                hidden=decoder_hidden,
                prev_att_vector=att_vector,
                trg_mask=trg_mask,
                **_kwargs)

    def __rep__(self) -> str:
        return (f"{self.__class__.__name__}(\n"
                                f"\tencoder={self.encoder},\n"
                                                f"\tdecoder={self.decoder},\n"
                                                                f"\tsrc_embed={self.src_embed},\n"
                                                                                f"\ttrg_embed={self.trg_embed},\n"
                                                                                                f"\tloss_function={self.loss_function})")

    def log_parameters_list(self) -> None:
        model_paramters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total params %d", n_params)
        trainable_params = [n for (n, p) in self.named_parameters() if p.requires_grad]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))


def build_translation_model(cfg: dict = None,
                            src_vocab: Vocabulary=None,
                            trg_vocab: Vocabulary=None) -> TranslationModel:
    cfg = cfg['model']
    #logger.info("Building an encoder-decoder model for gloss translation")
    enc_cfg = cfg['encoder']
    dec_cfg = cfg['decoder']

    src_pad_index = src_vocab.pad_index
    trg_pad_index = trg_vocab.pad_index

    src_embed = Embeddings(**enc_cfg["embeddings"],
                    vocab_size=len(src_vocab),
                            padding_idx=src_pad_index,
                                )
    trg_embed = Embeddings(
                        **dec_cfg["embeddings"],
                                    vocab_size=len(trg_vocab),
                                                padding_idx=trg_pad_index,
                                                        )

    enc_dropout = enc_cfg.get("dropout", 0.0)
    enc_emb_dropout = enc_cfg["embeddings"].get("dropout", enc_dropout)
    emb_size = src_embed.embedding_dim
    encoder = TransformerEncoder(
            **enc_cfg,
            emb_size=emb_size,
            emb_dropout=enc_emb_dropout,
            pad_index=src_pad_index)

    dec_dropout = dec_cfg.get("dropout", 0.0)
    dec_emb_dropout = dec_cfg["embeddings"].get("dropout", dec_dropout)
    decoder = TransformerDecoderWord(
            **dec_cfg,
            encoder=encoder,
            vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
    
    model = TranslationModel(
            encoder=encoder,
            decoder=decoder,
            src_embed=src_embed,
            trg_embed=trg_embed,
            src_vocab=src_vocab,
            trg_vocab=trg_vocab)

    initialize_model(model, cfg, src_pad_index, trg_pad_index)
    return model
class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 gloss_output_layer: nn.Module,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 src_gloss_embed: Embeddings,
                 trg_gloss_embed: Embeddings,
                 src_gloss_vocab: Vocabulary,
                 trg_gloss_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 do_recognition: bool=True,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_gloss_vocab = src_gloss_vocab
        self.trg_gloss_vocab = trg_gloss_vocab
        self.bos_index = self.src_gloss_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_gloss_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_gloss_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD
        
        self.gloss_output_layer = gloss_output_layer
        self.do_recognition =do_recognition
        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in",True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise",False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None,
                src_input: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        #print(encoder_output)
        #raise EOFError
        if self.do_recognition:
            # Gloss Recognition Part (referene from https://github.com/neccam/slt/blob/master/signjoey/model.py)
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2)
            gloss_output = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_output = None 
        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:,:,-1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds,torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate*noise

        # Decode the target sequence
        skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
                                                 src_mask=src_mask, trg_input=trg_input,
                                                 trg_mask=trg_mask)

        #gloss_out = None

        return skel_out, gloss_output

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)

        return encode_output


    def decode(self, encoder_output: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)
        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                               src_mask=src_mask,trg_mask=trg_mask)

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module, recognition_loss_function: nn.Module, recognition_loss_weight: float=1.0) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, gloss_out  = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask)

        if self.do_recognition:
            assert gloss_out is not None
            # Calculate Recognition Loss 
            #print(batch.src_lengths)
            #print(batch.src_gloss_lengths)

            recognition_loss = (

                    recognition_loss_function(
                        gloss_out,
                        batch.src_gloss,
                        batch.src_lengths.long(),
                        batch.src_gloss_lengths,
                    )
                    * recognition_loss_weight
            )
        else:
            recognition_loss = None 
        # compute batch loss using skel_out and the batch target
        batch_loss = loss_function(skel_out, batch.trg)

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None
        
        if self.do_recognition:
            return batch_loss + recognition_loss, noise 
        # return batch loss = sum over all elements in batch that are not pad
        else:
            return batch_loss, noise

    def run_batch(self, batch: Batch, max_output_length: int, recognition_beam_size: int=3) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)
        #print("Encoder", encoder_output.shape)
        #print(batch.trg_input) 

        if self.do_recognition:
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2)
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                                    (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                                                    axis=-1,
                                                                )
            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                        inputs=tf_gloss_probabilities,
                        sequence_length=batch.src_lengths.cpu().detach().numpy(),
                        beam_width=recognition_beam_size,
                        top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append([x[0] for x in groupby(tmp_gloss_sequences[seq_idx])])
        else:
            decoded_gloss_sequences = None 
        print('decoded gloss', decoded_gloss_sequences)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
                encoder_output=encoder_output,
                src_mask=batch.src_mask,
                embed=self.trg_embed,
                decoder=self.decoder,
                trg_input=batch.trg_input,
                model=self)
        #print(stacked_output)
       # print("Greedy search", stacked_output.shape)
        #print(stacked_output)
        #raise EOFError
        return decoded_gloss_sequences,  stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                src_gloss_vocab: Vocabulary = None,
                trg_gloss_vocab: Vocabulary = None,
                do_recognition: bool = True) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = 0
    src_gloss_padding_idx = src_gloss_vocab.stoi[PAD_TOKEN]
    trg_gloss_padding_idx = trg_gloss_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1 ) * future_prediction + 1

    # Define source embedding
    src_gloss_embed = Embeddings(
       **cfg["encoder"]["embeddings"], vocab_size=len(src_gloss_vocab),
        padding_idx=src_gloss_padding_idx)
    
    trg_gloss_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(trg_gloss_vocab),
                padding_idx=src_gloss_padding_idx)

    src_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])
    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=src_linear.out_features,
                                 emb_dropout=enc_emb_dropout)

    ## Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.) # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    decoder = TransformerDecoder(
        **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
        emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
        trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)

    if do_recognition:
        gloss_output_layer = nn.Linear(encoder.output_size, len(src_gloss_vocab))
        if cfg['encoder'].get('freeze', False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None 
    # Define the model
    model = Model(encoder=encoder,
                  decoder=decoder,
                  gloss_output_layer=gloss_output_layer,
                  src_embed=src_linear,
                  trg_embed=trg_linear,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  src_gloss_embed=src_gloss_embed,
                  trg_gloss_embed=trg_gloss_embed,
                  src_gloss_vocab=src_gloss_vocab,
                  trg_gloss_vocab=trg_gloss_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
