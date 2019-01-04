# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:43:06 2018

@author: piesauce
"""

import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape
import random
import NER
from example import Example



filename = './data/AG/newsspace200.xml'


def get_data():
    """
    Extract data from XML file
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    return root


def unescape_chars(X):
    return unescape(X.replace('\\', ' '))

def train_test_split(examples):
    """
    Random split of examples into train, dev, test sets - 80% train, 10% dev, 10% test
    """
    random.shuffle(examples)
    one_tenth = len(examples) // 10
    train, dev, test = examples[:one_tenth*8], examples[one_tenth*8:one_tenth*9], examples[one_tenth*9:]
    return train, dev, test

def gen_examples():
    """
    Generate train, dev, test examples from data.
    Extract news articles only belonging to topics 'World', 'Entertainment', 'Sports', and 'Business'.
    Retain only those examples that contain an instance of one of the top 5 most frequent named entities.
    """
    examples = []
    temp = []
    categories = ['World', 'Entertainment', 'Sports', 'Business']
    cat_label = {x:i for i, x in enumerate(categories)}
    tree_root = get_data()
    
    for child in tree_root:
        if child.tag == 'title' or child.tag == 'category' or child.tag == 'description':
            temp.append(child.text)
        if child.tag == 'pubdate':
            if len(temp) == 3:
                if temp[1] in categories:
                    if temp[0] is not None and temp[2] is not None:
                        X = temp[0] + " " + temp[2]
                        X_processed = unescape_chars(X)
                        Y = cat_label[temp[1]]
                        ex = Example(X_processed, Y)
                        examples.append(ex)
            temp = []
    new_examples = NER.ne_extract(examples, top=5)
    train, dev, test = train_test_split(new_examples)
    return train, dev, test                
                    

def preprocess_data():
    """
    Calls gen_examples() to generate training data
    """
    train, dev, test = gen_examples()
    return train, dev, test
    
                    
                
            
            
            
        
    
    
    
    
    
   


  
