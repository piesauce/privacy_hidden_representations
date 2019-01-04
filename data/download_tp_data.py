# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:05:05 2018

@author: piesauce
"""

import requests
from bs4 import BeautifulSoup as bs
import pathlib
import wget
import zipfile


url = 'https://bitbucket.org/lowlands/release/src/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/'



r = requests.get(url)
soup = bs(r.text, 'lxml')

file_names = []
urls = []


for i, link in enumerate(soup.findAll('a')):
    full_url =  link.get('href')
   
    if '.zip' in full_url:
        corr_url = url + full_url.split('.zip', 1)[0].rsplit('/', 1)[1] + '.zip'
        corr_raw_url = corr_url.replace('src', 'raw')
        urls.append(corr_raw_url)
        full_name = soup.select('a')[i].attrs['href']
        file_names.append(pathlib.Path(full_name).name.split('.zip', 1)[0] + '.zip') 
      
     
names_urls = zip(file_names, urls)

for file_name, link in names_urls:
    wget.download(link, out=file_name)
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall()
    zip_ref.close()
    














