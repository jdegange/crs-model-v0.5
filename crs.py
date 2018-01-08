#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:24:37 2017

@author: chublet
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import string

#%% function for manually defining the weights
class manual_model(object):
    '''
    my own simple linear regression model using manually entered weights
    '''
    
    def __init__(self,allProfs):
        '''make sure to onehot or dummy encode allProfs before sending as argument!'''
        self.set_weights(allProfs)
        
    def set_weights(self,allProfs):
        '''
        manually input a model given the features in allProfs
        returns weights
        '''
        columns = allProfs.columns
        self.weights = {}
        for col in columns:
            self.weights[col] = float(input('Weight of ' + col + ': '))    
    
    def predict(self,profs):
        '''
        linear regression prediction based on input features in profs
        returns nparray of outputs
        '''
        weights = np.array([self.weights[col] for col in profs.columns])
        return np.dot(profs,weights)        


#%% load attributes from excel
def loadAttr(fname):
    '''
    loads attributes from csv file "fname"
    returns dictionary of attributes and their levels
    '''
    file= open(fname,'r')
    firstLine= True
    attr= {}
    for line in file:
        if firstLine:
            firstLine= False
            continue
        tmp= line.replace('\n','')
        tmp= tmp.replace('\r','')
        tmp= tmp.split(',')
        while '' in tmp: tmp.remove('')
        attr[tmp[0]]= tmp[1:]
    file.close()
    return attr

#%% load profiles from excel
def loadProf(fname):
    '''
    loads profiles from csv file "fname"
    returns dataframe of profiles with their attribute values
    '''
    file = open(fname,'r')
    
    # loop over all lines in the file
    firstLine,secondLine = True, True
    for line in file:
        
        # remove unwanted strings
        tmp= line.replace('\n','')
        tmp= tmp.replace('\r','')
        tmp= tmp.split(',')
        while '' in tmp: tmp.remove('')
        
        if firstLine: # first line is attribute keys
            keys = tmp
            allProfs = pd.DataFrame(None,columns=keys)
            firstLine = False
            continue
        elif secondLine: # second line: initialize dataframe
            allProfs = pd.DataFrame([tmp],columns=keys)
            secondLine = False
        else: # remaining lines: append to dataframe
            allProfs = allProfs.append(pd.DataFrame([tmp],columns=keys),ignore_index=True)
    return allProfs

#%% run the survey in command line
def runSurvey(allProfs,specify_coef=False):
    '''
    Runs the survey given attributes and their levels
    returns allProfs (dataframe of independent variables), y (encoded  1/-1/0 for max/min/other)
    '''
    if specify_coef: model = manual_model(pd.get_dummies(allProfs))
    y = np.zeros(N)
        
    # loop over choice sets
    print('BEGIN SURVEY\n')
    for setId in range(int(N/Nps)):
        
        # convert set into a dataframe for easy display
        profs = allProfs[setId*Nps:(setId+1)*Nps].reset_index(drop=True)
         
        # print for user and get input
        print('SET ' + str(setId+1) + ' of ' + str(N) + ':')        
        print(profs)
        if specify_coef:
            set_y = model.predict(pd.get_dummies(profs))
            altId = np.argmax(set_y)
        else:
            altId= int(input('Index of the most risky customer (' + '/'.join(map(str,profs.index)) + '): '))
        assert altId in profs.index, 'Customer index out of range'
        print('\n')
        
        # put answer into label array
        y[setId*Nps+altId]= 1
    
    return y

#%% request user input on number of profiles and profiles per set
def requestMetadata():
    '''
    Ask user to specify number of profiles and profiles per choice set
    '''
    print('SURVEY METADATA')
    N = int(input('Total number of profiles: '))
    Nps = int(input('Number of profiles per set: '))
    print('\n----------------------\n')
    assert N%Nps==0, 'Please ensure N%Nps==0'
    return N,Nps
    
#%% run survey

# request metadata request
N,Nps = requestMetadata()

# load attribute/level list
#fnameAttr = 'EntityRiskData_Levels_noCont_simplified_transposed' + '_N=' + str(N)
fnameAttr = 'attributes_fun' + '_N=' + str(N)
attr = loadAttr(fnameAttr + '.csv')
keys = attr.keys()

# load d-optimal design made by matlab and saved under the filename fnameAttr.csv
allProfs = loadProf(fnameAttr + '.csv')

# label the profiles via running the survey
y = runSurvey(allProfs,specify_coef=True)

#%% train/eval model
facOnehot = pd.get_dummies(allProfs,drop_first=True)

# train model
Xtrain, Xtest, ytrain, ytest = train_test_split(facOnehot,y,test_size=.2)
model= LogisticRegression(max_iter=1000)
model.fit(pd.get_dummies(Xtrain,drop_first=True),ytrain)

# evaluate model
ytestHat = model.predict(pd.get_dummies(Xtest))
accTest = sk.metrics.accuracy_score(ytest,ytestHat)

ytrainHat = model.predict(pd.get_dummies(Xtrain))
accTrain = sk.metrics.accuracy_score(ytrain,ytrainHat)

print(allProfs.assign(y=y))
print('Training accuracy: ' + str(accTrain))
print('Testing accuracy: ' + str(accTest))

#%% plot weights

# preliminary stuff
coef = np.squeeze(model.coef_)
colOnehot = list(facOnehot.columns)
numLvl= [len(attr[key]) for key in keys]
colors = plt.cm.tab20(range(max(numLvl)))

# load model weights and their corresponding names into two tables
tab = np.empty((max(numLvl),len(keys)),dtype=np.dtype('<U8'))
val = np.zeros((max(numLvl),len(keys)))
keys = list(attr.keys())
for k in range(len(keys)):
    levels = attr[keys[k]]
    cnt = 0
    for l in range(len(levels)):
        col = keys[k] + '_' + levels[l]
        if col in colOnehot:
            val[cnt,k] = coef[colOnehot.index(col)]
            tab[cnt,k] = levels[l]
            cnt += 1

# plot bar chart of the weights
width = .15
index = np.arange(len(keys))+.3
for row in range(val.shape[0]-1,-1,-1):
    plt.bar(index+row*width,val[row],width,color=colors[row])
    
# execute plotting
plt.table(cellText=tab,rowColours=colors,colLabels=keys)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xticks([])
plt.ylabel('weight')
plt.title('Trained model weights')
plt.show()





