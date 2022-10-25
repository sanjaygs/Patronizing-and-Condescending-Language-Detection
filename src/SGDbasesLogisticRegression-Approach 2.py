#!/usr/bin/env python
# coding: utf-8

# In[113]:


import csv
import numpy as np
import math
from random import randrange
import sklearn


# In[2]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# # Reading and formatting the dataset

# In[3]:


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

    return ids, keywords, texts, labels


# In[4]:


IDs,Keywords,Texts,Labels = read_dataset('dontpatronizeme_pcl.tsv')


# # Defining the Keywords

# In[6]:


custom_Keywords = ['dis','able',
 'home','less',
 'hope',
 'need',
 'migr',dd
 'poor',
 'refuge',
 'vulner',
 'women',]


# In[7]:


custom_Keywords


# # Importing the Sentiment Analysis Outcome for each Statement

# In[23]:


import pickle
senti = pickle.load(open('sentimentmap','rb'))


# In[74]:


k=list(senti.values())
# print(senti(k[0]))
sen_label = []
ct=0
for i in range(8639):
    sen_label.append(k[i])
sen_label.append('NEGATIVE')
for i in range(8639,len(Texts)-1):
    print(i)
    sen_label.append(k[i])


# # Function that calculates the feature vector for each Statement

# In[79]:


def feature_extraction(sentence,labl):
    words=[]
    feature=[]
    pro = ["I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours", "them"]
    for i in sentence.split(' '):
        words.append(i.lower().replace(".",'').replace(",",'').replace("?",''))
    x1=0
    x2=0
    for j in words:
        for k in custom_Keywords:
            if k in j:
                x1+=1
        if j in pro:
            x2+=1
    if(labl=='NEGATIVE'):
        x3=0
    else:
        x3=1
    x4=len(i)
    x5=1
    feature = [x1,x2,x3,x4,x5]
    return feature


# In[81]:


features = []
for i in range(len(Texts)):
    features.append(feature_extraction(Texts[i],sen_label[i]))


# In[83]:


x1,x2,x3,x4,x5,x6=[],[],[],[],[],[]
for i in features:
    x1.append(i[0])
    x2.append(i[1])
    x3.append(i[2])
    x4.append(i[3])
    x5.append(i[4])


# In[85]:


# feature_extraction(Texts[0])


# In[86]:


len(features),len(Labels),len(IDs)


# In[87]:


rows = zip(IDs,x1,x2,x3,x4,Labels)


# # Writing the ID, Features and Class label to CSV file.

# In[88]:


with open('GangarekalveSomashekar-Sanjay-Project.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["ID","X1","X2","X3","X4","CLASS LABEL"])
    for row in rows:
        writer.writerow(row)


# In[89]:


k={}
with open('GangarekalveSomashekar-Sanjay-Project.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for i in reader:
        id_=i['ID']
        x1=i['X1']
        x2=i['X2']
        x3=i['X3']
        x4=i['X4']
        label=i['CLASS LABEL']
        for j in i:
            if j in k.keys():
                k[j].append((i[j]))
            else:
                k[j]=[]
                k[j].append((i[j]))


# # Extracting Feature Vector from the CSV file

# In[90]:


kk = np.array((list(k.values())[1:-1]),dtype=float)
X_tt = kk.T
X_train = X_tt.tolist()
(X_train)
for i in range(len(X_train)):
    X_train[i].append(1)


# # Extracting Class Labels from the CSV file

# In[92]:


yy = np.array((list(k.values())[-1]),dtype=float)
Y_tt = yy.T
Y_train = Y_tt.tolist()
# (Y_train)


# # Extracting IDs from the CSV file

# In[93]:


IDs = np.array((list(k.values())[0]))


# # Sigmoid Calculator

# In[94]:


def sigmoid(x):
    return 1/(1+math.exp(-x))


# # Entropy Loss Calculator

# In[95]:


def loss(weight,feature,label):
    return -1*((label*np.log(sigmoid(np.dot(feature,weight))))+((1-label)*np.log(1-sigmoid(np.dot(feature,weight)))))


# # SGD to obtain optimal Weight vector

# In[96]:


def SGD(no_epochs,weight,lr,x_train,y_train):
    epochs = 1
    while(epochs<=no_epochs):
        y_predict = []
        entropy_loss = []
#         print("Weight is ",weight)
        for i in range(len(x_train)):
            grad = []
            for j in range(len(x_train[i])):
                grad.append((sigmoid(np.dot(x_train[i],weight))-y_train[i])*x_train[i][j])
            entropy_loss.append(loss(weight,x_train[i],y_train[i]))
            for k in range(len(grad)):
                weight[k]-=round(grad[k]*lr,3)
            if(sigmoid(np.dot(x_train[i],weight))>=0.5):
                y_predict.append(1)
            else:
                y_predict.append(0)
        ct=0
        for i in range(len(y_predict)):
            if(y_predict[i]==y_train[i]):
                ct+=1
#         print("Accuracy is ",(ct/len(y_predict))*100,"%")
        
        epochs+=1
    return weight


# # Function to Predict class label using optimal weight vector obtained by SGD

# In[97]:


def predict(x_test,weight):
    value = sigmoid(np.dot(x_test,weight))
    if(value>=0.5):
        return 1
    else:
        return 0


# # Function to calculate Accuracy for Development Set

# In[98]:


def accuracy(y_actual,y_predict):
    ct=0
    for i in range(len(y_actual)):
        if(y_actual[i]==y_predict[i]):
            ct+=1
    return (ct*100)/len(y_actual)


# In[99]:


no_of_0 = 0
for i in Y_train:
    if(i==0):
        no_of_0+=1
# print(no_of_0)
no_of_1 = len(Y_train)-no_of_0
no_of_1


# # Splitting the Data into Training and Development Sets equally

# In[100]:


final_X_train = (X_train[:int((no_of_0)*0.8)]+X_train[no_of_0:no_of_0+int((no_of_1)*0.8)])


# In[101]:


final_Y_train = (Y_train[:int((no_of_0)*0.8)]+Y_train[no_of_0:no_of_0+int((no_of_1)*0.8)])


# In[102]:


X_test = X_train[int((no_of_0)*0.8):no_of_0]+X_train[no_of_0+int((no_of_1)*0.8):]


# In[103]:


Y_test = Y_train[int((no_of_0)*0.8):no_of_0]+Y_train[no_of_0+int((no_of_1)*0.8):]


# In[104]:


final_X_train = np.array(final_X_train,dtype=float)
final_Y_train = np.array(final_Y_train,dtype=float)
X_test = np.array(X_test,dtype=float)
Y_test = np.array(Y_test,dtype=float)


# In[124]:


weight = [0,0,1,0,0]
lr = 0.1
no_epochs = 250


# In[125]:


final_weights = SGD(no_epochs,weight,lr,final_X_train,final_Y_train)


# In[109]:


len(X_train[:int(len(X_train)*0.8)]),len(X_train[int(len(X_train)*0.8):])


# # Calculating the accuracy

# In[121]:


y_predict = []
for i in X_test:
    y_predict.append(predict(i,final_weights))


# In[122]:


accuracy(Y_test,y_predict)


# # Precision, Recall, F1 Score and number of instance for class 0 and 1 respectively

# In[123]:


sklearn.metrics.precision_recall_fscore_support(Y_test,y_predict,average=None)


# # Calculating Confusion matrix and plot the same

# In[115]:


import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test,y_predict, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Not PCL','PCL'])


# In[253]:


disp.plot()


# In[ ]:




