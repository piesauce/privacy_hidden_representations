# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:16:06 2019

@author: piesauce
"""

import nltk.tokenize as tokenizer
from nltk import ne_chunk, pos_tag
from collections import defaultdict

def get_ne(examples):
    """
    Use NLTK to get named entities present in the examples
    """
    ne_overall = []
    for ex in examples:
        ne_per_example = []
        sent_tokenized = tokenizer.word_tokenize(ex.get_X())
        chunks = ne_chunk(pos_tag(sent_tokenized))
        for c in chunks:
            if hasattr(c, 'label'):
                tag = c.label()
                entities = "_".join([e[0] for e in c])
                ne_per_example.append((tag, entities))
        ne_overall.append(ne_per_example)
    return ne_overall
            
def construct_new_dataset(examples, named_entities, most_freq_entities, freq_entity_map):
    """
    Construct new dataset containing only those examples which contain instances of the 'top' most frequent entities
    """
    new_data = []
    for example, ne_instance in zip(examples, named_entities):
        Z = {freq_entity_map[e] for e in ne_instance if e in most_freq_entities}
        if len(Z) > 0:
            example.Z = Z
            new_data.append(example)
    return new_data
     
         
def ne_extract(examples, top=5):
    """
    Extract examples containing instances of 'top' most frequent entities
    """
    counts = defaultdict(int)
    named_entities = get_ne(examples)
    for i in named_entities:
        for j in range(len(i)):
            tag_instance = i[j]
            if tag_instance[0] == 'PERSON':
                counts[tag_instance] += 1
    
    most_freq = sorted(counts, key = lambda x: counts[x], reverse=True)[:top]
    freq_entity_map = {e: i for i, e in enumerate(most_freq)}
    return construct_new_dataset(examples, named_entities, most_freq, freq_entity_map)
    
