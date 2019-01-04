# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:51:53 2018

@author: piesauce
"""

import os
from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from collections import defaultdict
import random


path = r'C:\Users\piesa\Documents\aDV\blogs1'


def get_contents(filename):
    """
    Extract and clean content from file
    """
    with open(os.path.join(path, filename), 'r', encoding='latin-1') as f:
        contents =[]
        for line in f:
            line = line.strip()
            if line and line[0] != '<':
                contents.append(line)
    return contents

def process_docs(examples):
    """
    Perform tokenization
    """
    processed_docs = []
    tokenizer = RegexpTokenizer(r'\w+')
    for ex in examples:
        doc = ex[3]
        processed_docs.append(tokenizer.tokenize(doc.lower()))
    return processed_docs
        
        
def train_LDA(documents):
    """
    Train an LDA (Latent Dirichlet allocation) model for topic modeling using Gensim
    """
    dct = Dictionary(documents)
    dct.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dct.doc2bow(text) for text in documents]
    model = ldamodel.LdaModel(corpus, num_topics=10, id2word=dct, passes=1)
    return model, corpus

def train_test_split(examples):
    """
    Generate train, dev, and test set using random splitting of examples with ratio - 80% train, 10% dev, 10% test
    """
    random.shuffle(examples)
    one_tenth = len(examples) // 10
    train, dev, test = examples[:one_tenth*8], examples[one_tenth*8:one_tenth*9], examples[one_tenth*9:]
    return train, dev, test

def balance_distributions(examples):
    """
    Balance distribution of examples so that they contain uniform distribution of private variables
    """
    balanced_dataset = []
    a_g = defaultdict(list)
    for ex in examples:
        a_g(ex[2]).append(ex)
    min_num = 10**10
    subcorpora = list(a_g.values())
    for subcorpus in subcorpora:
        if len(subcorpus) < min_num:
            min_num = len(subcorpus)
    for subcorpus in subcorpora:
        random.shuffle(subcorpus)
        balanced_dataset.extend(subcorpus[:min_num])
    return balanced_dataset


  
def preprocess_data():
    """
    Extract, clean, and generate example from dataset    
    """
X = []
for filename in os.listdir(path):
    user_info = filename.split('.')
    if user_info[2].startswith('1') or user_info[2].startswith('3'):
        userid = user_info[0]
        gender = user_info[1]
        age = user_info[2]
        docs = get_contents(filename)
        X+= [[userid, gender, age, doc] for doc in docs]
docs_processed = process_docs(X)
model, corpus = train_LDA(docs_processed)

new_dataset = []
for ex, doc in zip(X, corpus):
    topics = model.get_document_topics(doc)
    prob_list = [p for _, p in topics]
    if max(prob_list) > 0.8:
        Y = max(topics, key = lambda x: x[1])[0]
        Z = (ex[1], ex[2])
        new_ex = [ex[3], Y, Z]
        new_dataset.append(new_ex)
balanced_dataset = balance_distributions(new_dataset)
train, dev, test = train_test_split(new_dataset)
return train, dev, test



   


