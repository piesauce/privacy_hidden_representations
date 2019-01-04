# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:22:51 2018

@author: piesauce
"""
import os
from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from collections import defaultdict
import random


path = '/data/blogs/'


def get_contents(filename):
    """
    Extract data from file.
    """
    with open(os.path.join(path, filename), 'r', encoding='latin-1') as f:
        contents =[]
        for line in f:
            line = line.strip()
            if line and line[0] != '<':
                contents.append(line)
    return contents

def process_posts(examples):
    """
    Performs tokenization and normalization
    """
    processed_posts = []
    tokenizer = RegexpTokenizer(r'\w+')
    for ex in examples:
        post = ex[0]
        processed_posts.append(tokenizer.tokenize(post.lower()))
    return processed_posts
        
        
def train_LDA(posts):
    """
    Uses gensim to train an LDA model for topic modeling.
    """
    dct = Dictionary(posts)
    dct.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dct.doc2bow(text) for text in posts]
    model = ldamodel.LdaModel(corpus, num_topics=10, id2word=dct, passes=1)
    return model, corpus

def train_test_split(examples):
    """
    Random split of examples into train, dev, test sets - 80% train, 10% dev, 10% test
    """
    random.shuffle(examples)
    one_tenth = len(examples) // 10
    train, dev, test = examples[:one_tenth*8], examples[one_tenth*8:one_tenth*9], examples[one_tenth*9:]
    return train, dev, test

def balance_distributions(examples):
    """
    Maintains an even distribution of private variables.
    """
    balanced_dataset = []
    a_g = defaultdict(list)
    for ex in examples:
        a_g[tuple(ex.get_Z())].append(ex)
    min_num = 10**10
    subcorpora = list(a_g.values())
    for subcorpus in subcorpora:
        if len(subcorpus) < min_num:
            min_num = len(subcorpus)
    for subcorpus in subcorpora:
        random.shuffle(subcorpus)
        balanced_dataset.extend(subcorpus[:min_num])
    random.shuffle(balanced_dataset)
    return balanced_dataset

def gen_examples(examples, model, corpus):
    """
    Uses gensim to add labels to the examples.
    """
    new_dataset = []
    for ex, post in zip(examples, corpus):
        topics = model.get_document_topics(post)
        prob_list = [p for _, p in topics]
        if max(prob_list) > 0.8:
            Y = max(topics, key = lambda x: x[1])[0]
            Z = set()
            if ex[2].startswith('m'):
                gender = True
            if ex[3].startswith('1'):  #age is bucketed into <20 and >20
                age = True
            if gender:
                Z.add(0)
            if age:
                Z.add(1)
            new_ex = Example(ex[0], Y, Z)
            new_dataset.append(new_ex)
            
    balanced_dataset = balance_distributions(new_dataset)
    train, dev, test = train_test_split(balanced_dataset)

def preprocess_data():
    """
    Extract private variables, perform topic modeling to generate labels, and call gen_examples to generate examples
    """
    examples = []
    for filename in os.listdir(path):
        user_info = filename.split('.')
        if user_info[2].startswith('1') or user_info[2].startswith('3'):
            userid = user_info[0]
            gender = user_info[1]
            age = user_info[2]
            posts = get_contents(filename)
            examples += [[post, userid, gender, age] for post in posts]
    posts_processed = process_posts(examples)
    model, corpus = train_LDA(posts_processed)
    train, dev, test = gen_examples(examples, model, corpus)
    return train, dev, test

 

   



   


