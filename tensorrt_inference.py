
# coding: utf-8

# In[1]:


import tensorrt as trt


# In[2]:


import pycuda.driver as cuda 
import pycuda.autoinit 
import numpy as np
import time


# In[3]:


from random import randint 
from PIL import Image


# In[4]:


from tensorrt.parsers import caffeparser


# In[5]:


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)


# In[6]:


INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['prob']
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
NUM_PICS = 1


# In[7]:


MODEL_PROTOTXT = '/machineLearning/installs/caffe/ResNet-50-deploy.prototxt'
CAFFE_MODEL = '/machineLearning/installs/caffe/ResNet-50-model.caffemodel'
DATA = '/data/imagenet_data/raw-data/train/n10565667/'
IMAGE_MEAN = '/machineLearning/installs/caffe/ResNet_mean.binaryproto'
TARGET ='/data/rawimages/'


# In[8]:


engine = trt.utils.caffe_to_trt_engine(G_LOGGER, MODEL_PROTOTXT, CAFFE_MODEL, NUM_PICS, 1 << 20, OUTPUT_LAYERS, trt.infer.DataType.FLOAT) 


# In[9]:


import os
file_list = [f for f in os.listdir(TARGET) if os.path.isfile(os.path.join(TARGET, f))]
images_trt = []
for f in file_list:
    limg = Image.open(os.path.join(TARGET, f))
    img = np.asarray(limg)
    img = img.astype(np.float32)
    images_trt.append(img)

num_batches = int(len(images_trt) / NUM_PICS)
print("Test Case: " + str(num_batches))


# In[10]:


runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()


# In[11]:


assert(engine.get_nb_bindings() == 2)
output = np.empty(OUTPUT_SIZE, dtype = np.float32)
counter = 0
stream = cuda.Stream()
d_input = cuda.mem_alloc(NUM_PICS * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(NUM_PICS * output.size * output.dtype.itemsize)
bindings = [int(d_input), int(d_output)]

start = time.time()
for img in images_trt:
    cuda.memcpy_htod_async(d_input, img, stream)
    context.enqueue(1, bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #print("Test Case: " + str(file_list[counter]))
    #print ("Prediction: " + str(np.argmax(output)) + str(counter))
    counter = counter + 1
    
stream.synchronize()
end = time.time()
diff = (end-start)
print("time:" + str(diff))
print("image/sec = " + str(num_batches/diff) )

