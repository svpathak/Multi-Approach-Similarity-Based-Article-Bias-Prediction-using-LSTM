#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
import W2V2
from sklearn import metrics


# In[2]:


glove_path = 'glove.6B.100d.txt'


# In[3]:


X_train,y_train = W2V2.Sentence2Vec(filename='Train.xlsx',glovepath=glove_path, embedding_dim=100, max_length=120)


# In[4]:


X_test,y_test = W2V2.Sentence2Vec(filename='Test.xlsx',glovepath=glove_path)


# In[5]:


X_valid,y_valid = W2V2.Sentence2Vec(filename='Valid.xlsx',glovepath=glove_path)


# In[6]:


X_train.shape


# In[7]:


pd.Series(y_train).value_counts()


# In[8]:


pd.Series(y_valid).value_counts()


# In[9]:


pd.Series(y_test).value_counts()


# In[10]:


max_length = X_train.shape[1]


# In[11]:


from torch.autograd import Variable
class myLSTM(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=dimension,
                            hidden_size=max_length,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(max_length, 3)
        self.act = nn.Softmax()

    def forward(self, x, batch_size):
        h_0 = Variable(torch.zeros(2, batch_size, max_length))
        c_0 = Variable(torch.zeros(2, batch_size, max_length))
        x, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        #print(x.shape)
        x = self.fc(final_hidden_state[-1])
        #print(x.shape)
        x = self.act(x)
        return x.squeeze()
        


# In[12]:


def train(model, x, y, loss_fn, optimizer, batch_size=1024):
    model.train()
    x, y = torch.Tensor(x), torch.Tensor(y)
    y = y.type(torch.LongTensor)
    net_loss = 0
    
    for i in range(0, len(x), batch_size):
        i_end = i+batch_size
        x_batch = x[i:min(i_end, len(x))]
        y_batch = y[i:min(i_end, len(x))]
        
        pred = model(x_batch,x_batch.shape[0])
        loss = loss_fn(pred, y_batch)
        net_loss = net_loss+loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return net_loss

def test(model, xts, yts,batchsize):
    model.eval()
    with torch.no_grad():
        xts = torch.Tensor(xts)
        pred = model(xts,batchsize)
        loss = loss_fn(pred, torch.Tensor(yts).type(torch.LongTensor))
        yhat = np.argmax(pred, axis=1).numpy()
        acc = np.sum(yhat==yts)*100/yhat.shape[0]
        f1 = metrics.f1_score(yts,yhat,average=None)
        return acc, f1,loss


# In[ ]:


model = myLSTM(X_train.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
batch_size = 512
epochs = 8
train_loss = []
test_acc = []
train_acc = []
test_loss = []

for i in range(epochs):
    print("--- Epoch {} ---".format(i+1))
    epoch_loss = train(model, X_train, y_train, loss_fn, optimizer, batch_size)
    train_loss.append(epoch_loss)
    print("\tCross Entropy Loss (Training) : {} ".format(epoch_loss))

    acc, _, __ = test(model, X_train, y_train, batchsize=X_train.shape[0])
    print("\tTrain Accuracy : {:.2f} % ".format(acc))
    train_acc.append(acc)
    
    acc, _, tst_loss = test(model, X_test, y_test, batchsize=X_test.shape[0])
    print("\tTest Accuracy : {:.2f} % ".format(acc))
    test_acc.append(acc)
    test_loss.append(tst_loss)


# In[18]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), train_loss, label="train loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss vs epochs")
plt.legend(loc="best")
plt.show()


# In[17]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), test_loss, label="test loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss vs epochs")
plt.legend(loc="best")
plt.show()


# In[15]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), train_acc, label="train accuracy")
plt.plot(range(epochs), test_acc, label="test accuracy")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.title("Accuracy vs Epochs")
plt.legend(loc="best")
plt.show()


# In[16]:


acc, f1, _ = test(model, X_test, y_test, batchsize=X_test.shape[0])
print("Test Accuracy : {:.2f} % ".format(acc))
print("Test Class-wise F1 Score : \n{}".format(f1))


# In[ ]:




