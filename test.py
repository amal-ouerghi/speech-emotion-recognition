import numpy
from scipy.io.wavfile import read
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
print("preparing the data ....")
data_0=[]
data_labels=[]
# labels={'a':numpy.array([1,0,0,0,0,0,0]),'d':numpy.array([0,1,0,0,0,0,0]),'f':numpy.array([0,0,1,0,0,0,0]), 'h':numpy.array([0,0,0,1,0,0,0]),'n':numpy.array([0,0,0,0,1,0,0]),'sa':numpy.array([0,0,0,0,0,1,0]),'su':numpy.array([0,0,0,0,0,0,1])}
labels={'a':0,'d':1,'f':2, 'h':3,'n':4,'sa':5,'su':6}
for fileName in os.listdir(os.getcwd()+"/all_data/"):
    a=read(os.getcwd()+"/all_data"+"/"+fileName)
    data_0.append(list(numpy.array(a[1],dtype=float).tolist()))
    if fileName[0]=='a':
        data_labels.append(labels['a'])
    elif fileName[0]=='d':
        data_labels.append(labels['d'])
    elif fileName[0]=='f':
        data_labels.append(labels['f'])
    elif fileName[0]=='h':
        data_labels.append(labels['h'])
    elif fileName[0]=='n':
        data_labels.append(labels['n'])
    elif fileName[1]=='a':
        data_labels.append(labels['sa'])
    elif fileName[1]=='u':
        data_labels.append(labels['su'])
print("reshaping the data ...")
data = list(numpy.zeros([len(data_0),len(max(data_0,key = lambda x: len(x)))]).tolist())
for i,j in enumerate(data_0):
    data[i][0:len(j)] = j
print(len(data))

print("splitting the data .....")
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set_labels, test_set_labes = train_test_split(data_labels, test_size=0.2, random_state=42)
print("training the model....")

svm_model_linear = SVC(kernel = 'linear', C = 1, gamma=0.0001).fit(train_set, train_set_labels)
print("testing the model ....")

svm_predictions = svm_model_linear.predict(test_set)
print(svm_predictions)
