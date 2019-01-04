# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:51:21 2018

@author: piesauce
"""

import wget
import zipfile
import os


local_path = r'C:\Users\piesa\Documents\aDV\blogs'
url = 'http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip'
filename = 'blogs.zip'

wget.download(url, out= local_path)
zip_ref = zipfile.ZipFile(os.path.join(local_path, filename), 'r')
zip_ref.extractall()
zip_ref.close()


