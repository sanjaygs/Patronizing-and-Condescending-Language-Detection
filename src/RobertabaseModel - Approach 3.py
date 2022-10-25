#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip install simpletransformers')
get_ipython().system('pip install tensorboardx')


# In[1]:


from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
from urllib import request
import pandas as pd
import logging
import torch
from collections import Counter
from ast import literal_eval
import sklearn


# In[2]:


# prepare logger
logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# check gpu
cuda_available = torch.cuda.is_available()

print('Cuda available? ',cuda_available)


# In[3]:


if cuda_available:
  import tensorflow as tf
  # Get the GPU device name.
  device_name = tf.test.gpu_device_name()
  # The device name should look like the following:
  if device_name == '/device:GPU:0':
      print('Found GPU at: {}'.format(device_name))
  else:
      raise SystemError('GPU device not found')


# # Reading and formatting the dataset

# In[4]:


from pathlib import Path
import re

def read_dataset(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = raw_text.split('\n')[4:]
#     ids_docs = []
#     keywords_docs = []
#     texts_docs = []
#     labels_docs = []
    ids = []
    keywords = []
    texts = []
    labels = []
    for line in raw_docs:
        
#         for line in doc.split('\n'):
        id_, keyword, text, label = line.split('\t')[0],line.split('\t')[2],line.split('\t')[4],line.split('\t')[5]
        ids.append(id_)
        keywords.append(keyword)
        texts.append(text)
        if(label=='0' or label=='1'):
            labels.append(0)
        else:
            labels.append(1)
#         labels.append(label)
#         token_docs.append(tokens)
#         tag_docs.append(tags)

    return ids, keywords, texts, labels


# # Importing Dataset into Google Colab

# In[5]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[6]:


IDs,Keywords,Texts,Labels = read_dataset('/content/gdrive/My Drive/Colab Notebooks/dontpatronizeme_pcl.tsv')


# # Splitting the Data into Training and Testing Sets

# In[7]:


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(Texts, Labels, test_size=.2)


# In[8]:


train_dataset = []
for i in range(len(train_texts)):
  train_dataset.append({'text':train_texts[i],'label':train_tags[i]})


# In[9]:


train_df = pd.DataFrame(train_dataset)


# In[10]:


val_dataset = []
for i in range(len(val_texts)):
  val_dataset.append({'text':val_texts[i],'label':val_tags[i]})


# In[11]:


val_df = pd.DataFrame(val_dataset)


# In[33]:


total_no_1s = len(train_df[train_df.label==1])


# In[37]:


training_set = pd.concat([train_df[train_df.label==1],train_df[train_df.label==0][3500:3500+total_no_1s*2]])


# In[38]:


total_no_1s


# In[39]:


len(training_set)


# # Making use of RoBERTa base Classification model.

# In[40]:


model_args = ClassificationArgs(num_train_epochs=10, 
                                      no_save=True, 
                                      no_cache=True, 
                                      overwrite_output_dir=True)
model = ClassificationModel("roberta", 
                                  'roberta-base', 
                                  args = model_args, 
                                  num_labels=2, 
                                  use_cuda=cuda_available)
# train model
model.train_model(training_set[['text', 'label']])


# In[41]:


val_total_no_1s = len(val_df[val_df.label==1])


# In[42]:


val_total_no_1s


# In[43]:


testing_set = pd.concat([val_df[val_df.label==1],val_df[val_df.label==0][:val_total_no_1s*2]])


# In[45]:


predictions, _ = model.predict(testing_set.text.tolist())


# In[46]:


len(predictions),len(testing_set.label)


# In[48]:


sklearn.metrics.precision_recall_fscore_support(testing_set.label,predictions,average=None)


# In[49]:


sklearn.metrics.precision_recall_fscore_support(testing_set.label,predictions,average='binary')


# # Using Confusion or error matrix to know the performance of the RoBERTa

# In[53]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(testing_set.label,predictions, labels=[0,1])


# In[56]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Not PCL','PCL'])


# In[57]:


disp.plot()


# # Classification Report: Precision, Recall, F1-score and Number of instances 

# In[59]:


from sklearn.metrics import classification_report
print(classification_report(testing_set.label,predictions,target_names=['Not PCL','PCL']))


# In[ ]:




