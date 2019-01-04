# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:07:17 2018

@author: piesauce
"""

import os
from ast import literal_eval
import nltk.tokenize as tokenizer
import random
random.seed(2)
from example import Example

path = './data/trustpilot/'

def get_data(filename):
    """
    Retrieve data from file
    """
    data = []
    with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            ev = literal_eval(line)
            data.append(ev)
    return data

def calculate_age_bucket(birth_year, review_year):
    """
    Age is divided into two buckets - under 35 and over 35
    
    Args:
        birth_year (int): Birth year of reviewer
        review_year (int): Year the review was written
    """
    birth_year = int(birth_year)
    review_year = int(review_year.split('-')[0])
    if review_year - birth_year < 35:
        return True
    if review_year - birth_year >= 35:
        return False
    
def train_test_split(examples):
    """
    Random split of examples into train, dev, test sets - 80% train, 10% dev, 10% test
    """
    random.shuffle(examples)
    one_tenth = len(examples) // 10
    train, dev, test = examples[:one_tenth*8], examples[one_tenth*8:one_tenth*9], examples[one_tenth*9:]
    return train, dev, test

def gen_prefix(Z):
    """
    Generates prefix that represents implicit private data associated with the input
    """
    g = []
    a = []
    g_indicator = 'M' if 0 in Z else 'F'
    a_indicator = 'U' if 1 in Z else 'O'
    
    g.append('<g={}>'.format(g_indicator))
    a.append('<a={}>'.format(a_indicator))
    return g, a
    

def gen_examples(filename, args):
    """
    Generates train, dev, test examples from data.
    Performs tokenization, normalization, and extraction of private variables.
    """
    
    examples = []
    data = get_data(filename)
    for example in data:
        review_data = example['reviews'][0]
        if review_data['rating'] is None or review_data['text'] is None or review_data['date'] is None:
            continue
        if example['gender'] is None or example['birth_year'] is None:
            continue
        if review_data['title'] is None:
            review_data['title'] = ''
        X = review_data['title'].lower() + " " + " ".join(review_data['text']).lower()
        X_tokenized = tokenizer.word_tokenize(X)
        Y = review_data['rating']  #output label
        
        gender = example['gender'] == 'M'
        age = calculate_age_bucket(example['birth_year'], review_data['date'])
        Z = set()   # private data 
        if gender:
            Z.add(0)
        if age:
            Z.add(1)
        if args.use_prefix:
            g, a = gen_prefix(Z)  # add gender and age as prefix to the input
            X_tokenized = g + a + X_tokenized
        ex = Example(X_tokenized, Y, Z)   # inputs, labels, and private variables
        examples.append(ex)
    train, dev, test = train_test_split(examples)
    return train, dev, test



def preprocess_data(args, prefix):
    """
    Filters out reviews that do not contain age and gender information.
    Calls gen_examples() to generate training examples.
    """
    
    for filename in os.listdir(path):
        if filename.startswith(prefix):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f1:
                with open(os.path.join(path, filename + '_filtered'), 'w+', encoding='utf-8') as f2:
                    for line in f1:
                        if '\'gender\'' in line and '\'birth_year\'' in line:    #Retrieve examples that contain both gender and birth year
                            f2.write(line)
                                                           
    
    for filename in os.listdir(path):
        if filename.startswith(prefix) and filename.endswith('_filtered'):
            train, dev, test = gen_examples(filename, args)
    return train, dev, test
        
    
    

    
            
            
  
















