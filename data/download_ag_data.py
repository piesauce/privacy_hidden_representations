# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:42:19 2018

@author: piesauce
"""

import wget
import bz2
import os


local_path = './data/AG/'
url = 'http://www.di.unipi.it/~gulli/newsspace200.xml.bz'

wget.download(url, out= local_path)

for filename in os.listdir(local_path):
    if filename.endswith('.bz'):
        with open(os.path.join(local_path, filename), 'rb') as f1:
            with open(os.path.join(local_path, filename[:-3]), 'wb') as f2:
                f2.write(bz2.decompress(f1.read()))

