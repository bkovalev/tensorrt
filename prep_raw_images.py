
# coding: utf-8

# In[1]:


import time
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from random import randint 
import pickle

INPUT_H = 224
INPUT_W = 224
NUM_PICS = 512
NUM_DIRS = 64

TARGET='/data/rawimages/'
DATA = '/data/imagenet_data/raw-data/train/'


# In[2]:


dirs= [f for f in listdir('/data/imagenet_data/raw-data/train/') if ("n" in f)]


# In[3]:


for t in range(NUM_DIRS):
    PATH = DATA + dirs[t] + "/"

    files= [f for f in listdir(PATH) if isfile(join(PATH, f))]

    start = time.time()

    img2 = np.empty(0)
    
    print(PATH)
    
    for k in range(NUM_PICS):
        name=files[k]
        full_name=PATH+name
        img = Image.open(full_name).convert('RGB')
        img = img.resize((INPUT_H,INPUT_W))
        img.save(TARGET + files[k] + '.bmp')
        
    end = time.time()
   
    
diff = (end-start)
print("time:" + str(diff))
print("image/sec = " + str((NUM_PICS * NUM_DIRS)/diff) )

