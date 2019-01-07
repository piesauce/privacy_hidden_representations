# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:12:13 2018

@author: piesauce
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Classifier(nn.Module):
    """
    Implements a BiLSTM based text classifier that utilizes both word and character embeddings.
    Characters in each word are passed through an LSTM to generate an encoding.
    The character encoding is concatenated with the word embeddings for each word in the input
    and is fed through a BiLSTM to generate an intermediate representation which is
    then fed to a fully connected layer that performs the classification.
    """
    
    def __init__(self, alphabet_size, vocab_size, output_size, args):
        """
        Args:
            alphabet_size (int): Number of unique characters in the input
            vocab_size (int): Size of the input vocabulary
            output_size (int): Number of class labels
            args: Command-line arguments
        """
        super(Classifier, self).__init__()
       
        self.char_hidden_dim = args.char_hidden_dim
        
        self.char_embedding = nn.Embedding(alphabet_size, args.char_embed_dim)
        self.char_bilstm = nn.LSTM(args.char_embed_dim, self.char_hidden_dim, bidirectional=True)
        
        self.word_hidden_dim = args.word_hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, args.word_embed_dim)
        self.bilstm = nn.LSTM(args.word_embed_dim + self.char_hidden_dim * 2, self.word_hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(self.word_hidden_dim * 2,  args.fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.fc_dim, output_size)
        self.softmax = nn.Softmax()
    
    def forward(self, sentence):
        """
        Args:
            sentence (list): Input sentence, consisting of a tuple (sentence_c, sentence_w)
            sentence_c contains the character indices of the input sentence.
            sentence_w contains the word indices of the input sentence.
        """
        sentence_c, sentence_w = sentence
        c_lstm_hidden = []
        
        
        for token in sentence_c:
            token = torch.tensor(token)
            h_c = Variable(torch.zeros(2, 1, self.char_hidden_dim))
            c_c = Variable(torch.zeros(2, 1, self.char_hidden_dim))
            
            char_embed = self.char_embedding(token).view(len(token), 1, -1)
            _ , (hidden_state, cell_state) = self.char_bilstm(char_embed, (h_c, c_c))
            hidden_state = hidden_state.view(-1, self.char_hidden_dim * 2)
            c_lstm_hidden.append(hidden_state)
        c_lstm_hidden = torch.stack(c_lstm_hidden)
        
        
        word_embed = self.word_embedding(sentence_w).view(len(sentence_w), 1, -1)
        
        wc_embed = torch.cat((word_embed, c_lstm_hidden), 2)
        
        h_w = Variable(torch.zeros(2, 1, self.word_hidden_dim))
        c_w = Variable(torch.zeros(2, 1, self.word_hidden_dim))
        _ , (hidden_state, cell_state) = self.bilstm(wc_embed, (h_w, c_w))
        hidden_state = hidden_state.view(-1, self.word_hidden_dim * 2)
        
        fc_output = self.fc1(hidden_state)
        fc_output = self.relu(fc_output)
        fc_output = self.fc2(fc_output)
        fc_output = self.softmax(fc_output)
        return fc_output
    
   
        
        
        
        
        
        
        
        
      
            
        
        
        
        
        
      