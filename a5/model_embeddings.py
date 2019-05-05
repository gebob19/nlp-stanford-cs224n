#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
"""

import torch.nn as nn
from cnn import CNN
from highway import Highway

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.vocab = vocab
        self.embed_size = embed_size

        self.char_embedding = nn.Embedding(len(vocab.char2id), 50, padding_idx=vocab.char2id['<pad>'])

        self.cnn = CNN(self.embed_size)
        self.higway = Highway(self.embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, padded):
        x_emb = self.char_embedding(padded)
        max_slen, bs, max_wlen, _ = x_emb.shape
        x_emb = e_emb.view(max_slen * bs, max_wlen, self.embed_size)

        x_conv_out = self.cnn(x_emb)
        x_highway = self.higway(x_conv_out)
        word_emb = self.dropout(x_highway)
        word_emb = word_emb.view(max_slen, bs, self.embed_size)

        return word_emb


