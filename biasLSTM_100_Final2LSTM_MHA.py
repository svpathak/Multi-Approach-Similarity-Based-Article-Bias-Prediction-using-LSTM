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
import W2V3
from sklearn import metrics


# In[2]:


glove_path = 'glove.6B.100d.txt'


# In[3]:


X_head_train,X_body_train,y_train = W2V3.Sentence2Vec(filename='Train.xlsx',glovepath=glove_path, embedding_dim=100, max_length=120)



X_head_test,X_body_test,y_test = W2V3.Sentence2Vec(filename='Test.xlsx',glovepath=glove_path, embedding_dim=100, max_length=120)


# In[5]:


print(X_head_test.shape)
print(X_body_test.shape)


# In[6]:


len(X_body_test)


# In[7]:


pd.Series(y_train).value_counts()


# In[8]:


pd.Series(y_test).value_counts()


# In[9]:


max_length = X_body_train.shape[1]


# In[10]:


from torch.autograd import Variable
class myLSTM(nn.Module):

 

    def __init__(self, dimension, hidd_dim):
        super().__init__()
        self.hidd_dim = hidd_dim
        self.n_mha = 2
        self.embed_dim = hidd_dim
        self.lstm = nn.LSTM(input_size=dimension,
                            hidden_size=self.hidd_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.drop = nn.Dropout(p=0.5)
        self.mha = nn.MultiheadAttention(self.embed_dim,self.n_mha,dropout=0.4,batch_first=True,kdim=self.hidd_dim,vdim=hidd_dim)
        self.fc = nn.Linear(self.embed_dim, 3)
        self.act = nn.Softmax()

 

    def forward(self, head, body, batch_size):
        np.random.seed(1)
        h_0_head = Variable(torch.rand(2, batch_size, self.hidd_dim))
        c_0_head = Variable(torch.rand(2, batch_size, self.hidd_dim))
        h_0_body = Variable(torch.rand(2, batch_size, self.hidd_dim))
        c_0_body = Variable(torch.rand(2, batch_size, self.hidd_dim))
        
        head, (final_hidden_state_head, final_cell_state_head) = self.lstm(head, (h_0_head, c_0_head))
        body, (final_hidden_state_body, final_cell_state_body) = self.lstm(body, (h_0_body, c_0_body))
        
        h = final_hidden_state_head[-1].reshape(batch_size,1,-1)
        b = final_hidden_state_body[-1].reshape(batch_size,1,-1)
        mha, mha_wgts = self.mha(b,b,h)
        x = self.fc(mha).reshape(-1,3)
        #x = self.act(x)
        
        return x


            


# In[11]:


def train(model, head,body, y, loss_fn, optimizer, batch_size=1024):
    model.train()
    head,body, y = torch.Tensor(head),torch.Tensor(head), torch.Tensor(y)
    y = y.type(torch.LongTensor)
    net_loss = 0
    
    for i in range(0, len(head), batch_size):
        i_end = i+batch_size
        head_batch = head[i:min(i_end, len(head))]
        body_batch = body[i:min(i_end, len(body))]
        y_batch = y[i:min(i_end, len(head))]
        
        pred = model(head_batch,body_batch,head_batch.shape[0])
        loss = loss_fn(pred, y_batch)
        net_loss = net_loss+loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return net_loss

def test(model, head,body, yts,batchsize):
    model.eval()
    with torch.no_grad():
        head = torch.Tensor(head)
        body = torch.Tensor(body)
        pred = model(head,body,batchsize)
        loss = loss_fn(pred, torch.Tensor(yts).type(torch.LongTensor))
        yhat = np.argmax(pred, axis=1).numpy()
        acc = np.sum(yhat==yts)*100/yhat.shape[0]
        f1 = metrics.f1_score(yts,yhat,average=None)
        return acc, f1,loss


# In[12]:


model = myLSTM(X_body_train.shape[2], hidd_dim=100)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
batch_size = 256
epochs = 8
train_loss = []
test_acc = []
train_acc = []
test_loss = []

for i in range(epochs):
    print("--- Epoch {} ---".format(i+1))
    epoch_loss = train(model,X_head_train, X_body_train, y_train, loss_fn, optimizer, batch_size)
    train_loss.append(epoch_loss)
    print("\tCross Entropy Loss (Training) : {} ".format(epoch_loss))

    acc, _, __ = test(model, X_head_train, X_body_train, y_train, batchsize=X_body_train.shape[0])
    print("\tTrain Accuracy : {:.2f} % ".format(acc))
    train_acc.append(acc)
    
    acc, _, tst_loss = test(model, X_head_test, X_body_test, y_test, batchsize=X_body_test.shape[0])
    print("\tTest Accuracy : {:.2f} % ".format(acc))
    test_acc.append(acc)
    test_loss.append(tst_loss)


# In[13]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), train_loss, label="train loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Loss vs epochs")
plt.legend(loc="best")
plt.show()


# In[14]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), test_loss, label="test loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 15)
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


plt.figure(figsize=(7,7))
plt.plot(range(epochs), train_acc, label="train accuracy")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.title("Accuracy vs Epochs")
plt.legend(loc="best")
plt.show()


# In[17]:


plt.figure(figsize=(7,7))
plt.plot(range(epochs), test_acc, label="test accuracy")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.title("Accuracy vs Epochs")
plt.legend(loc="best")
plt.show()


# In[18]:


acc, f1, _ = test(model, X_head_test,X_body_test, y_test, batchsize=X_body_test.shape[0])
print("Test Accuracy : {:.2f} % ".format(acc))
print("Test Class-wise F1 Score : \n{}".format(f1))


# In[ ]:





# In[ ]:




