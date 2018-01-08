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
    
    def __init__(self,profs):
        self.set_weights(profs)
        
    def set_weights(self,profs):
        '''
        manually input a model given the features in profs
        returns weights
        '''
        columns = profs.columns
        self.weights = {}
        for col in columns:
            self.weights[col] = float(input('Weight of ' + col + ': '))    
    
    def predict(self,setDf):
        '''
        linear regression prediction based on input features in setDf
        returns nparray of outputs
        '''
        weights = np.array([self.weights[col] for col in setDf.columns])
        return np.dot(setDf,weights)        


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
    firstLine = True
    secondLine = True
    profs = pd.DataFrame()
    for line in file:
        tmp= line.replace('\n','')
        tmp= tmp.replace('\r','')
        tmp= tmp.split(',')
        while '' in tmp: tmp.remove('')
        if firstLine:
            keys = tmp
            profs = pd.DataFrame(None,columns=keys)
            firstLine = False
            continue
        elif secondLine:
            profs = pd.DataFrame([tmp],columns=keys)
            secondLine = False
        else:
            profs = profs.append(pd.DataFrame([tmp],columns=keys),ignore_index=True)
    return profs
    

#%% make survey from attributes
def makeSampledSurvey(attr,N,Nps):
    '''
    makes a uniformly sampled survey based on the attributes given
    attr - attributes dictionary from loadAttr
    N - number of choice sets
    Nps - number of alternatives per set
    '''
      
    def makeFullFac(numLvl):
        '''
        recursive subfunction to make a matrix of all possible combinations of levels
        numLvl - list of the number of levels of each attribute
        returns matrix where each column is an attribute and each row is an alternative
        the matrix will have prod(numLvl) rows and len(numLvl) columns
        
        '''
        if len(numLvl)==1: return np.arange(numLvl[0]).reshape(numLvl[0],1)
        col2= makeFullFac(numLvl[1:])
        col1= np.repeat(range(numLvl[0]),col2.shape[0])
        col1= col1.reshape(len(col1),1)
        col2= np.tile(col2,(numLvl[0],1))
        fac= np.concatenate((col1,col2),axis=1)
        return fac
        
    # make full factorial of alternatives in matrix form
    keys= list(attr.keys())
    numLvl= [len(attr[key]) for key in keys]
    fac= makeFullFac(numLvl)

    # randomly draw a random subset from full factorial to make a fractional factorial
    numAlt = N*Nps
#    assert numAlt<fac.shape[0], 'N*Nps must be less than number of alternatives in full factorial'
    facId = []
    for setId in range(N):
        altId = []
        while len(altId)<Nps:
            tmp = int(np.random.rand()*fac.shape[0])
            if tmp not in altId: altId.append(tmp)
        facId.extend(altId)
        
    # convert fractional factorial from matrix to dataframe and return
    profs = pd.DataFrame({keys[k]:np.array(attr[keys[k]])[fac[facId,k]] for k in range(len(keys))})
    facFull = pd.DataFrame({keys[k]:np.array(attr[keys[k]])[fac[:,k]] for k in range(len(keys))})
    return profs, facFull


#%% run the survey in command line
def runSurvey(attr,setCoef=False):
    '''
    Runs the survey given attributes and their levels
    attr - attributes dictionary from loadATtr
    returns profs (dataframe of independent variables), y (1 if alternative chose, 0 otherwise)
    '''
    
    if setCoef: model = manual_model(pd.get_dummies(profs))
    y = np.zeros(N*Nps)
        
    # loop over choice sets
    
    print('BEGIN SURVEY\n')
    for setId in range(N):
        
        # convert set into a dataframe for easy display
        setDf = profs[setId*Nps:(setId+1)*Nps].reset_index(drop=True)
         
        # print for user and get input
        print('SET ' + str(setId+1) + ' of ' + str(N) + ':')        
        print(setDf)
        if setCoef:
            set_y = model.predict(pd.get_dummies(setDf))
            altId = np.argmax(set_y)
        else:
            altId= int(input('Index of the most risky customer (' + '/'.join(map(str,setDf.index)) + '): '))
        assert altId in setDf.index, 'Customer index out of range'
        print('\n')
        
        # put answer into label array
        y[setId*Nps+altId]= 1
    
    return profs,y

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
fnameAttr = 'EntityRiskData_Levels_noCont_simplified_transposed' + '_N=' + str(N)
fnameAttr = 'attributes_fun' + '_N=' + str(N)
attr = loadAttr(fnameAttr + '.csv')
keys = attr.keys()

# load d-optimal design made by matlab and saved under the filename fnameAttr.csv
profs = loadProf(fnameAttr + '.csv')

# label the profiles via running the survey
y = runSurvey(profs,setCoef=True)

#%% train/eval model
facOnehot = pd.get_dummies(profs,drop_first=True)

# train model
Xtrain, Xtest, ytrain, ytest = train_test_split(facOnehot,y,test_size=.2)
model= LogisticRegression(max_iter=1000)
model.fit(pd.get_dummies(Xtrain,drop_first=True),ytrain)

# evaluate model
ytestHat = model.predict(pd.get_dummies(Xtest))
accTest = sk.metrics.accuracy_score(ytest,ytestHat)

ytrainHat = model.predict(pd.get_dummies(Xtrain))
accTrain = sk.metrics.accuracy_score(ytrain,ytrainHat)

print(profs.assign(y=y))
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





