# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:35:55 2020

@author: Poorvahab
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def GetBestClassifier(count:int=5):

    dataset = pd.read_csv('vehicle_csv.csv')
    print('\n________ Dataset _________')
    print(dataset)
  #  bestmodel = GaussianNB()
   # modelno = 100
    for i in range(count):
       from sklearn.model_selection import train_test_split
    X= dataset[['COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO',
            'PR.AXIS_ASPECT_RATIO','MAX.LENGTH_ASPECT_RATIO','SCATTER_RATIO',
            'ELONGATEDNESS','PR.AXIS_RECTANGULARITY','MAX.LENGTH_RECTANGULARITY',
            'SCALED_VARIANCE_MAJOR','SCALED_VARIANCE_MINOR','SCALED_RADIUS_OF_GYRATION',
            'SKEWNESS_ABOUT_MAJOR','SKEWNESS_ABOUT_MINOR','KURTOSIS_ABOUT_MAJOR',
            'KURTOSIS_ABOUT_MINOR','HOLLOWS_RATIO']]
    Y= dataset['Class']
    Xtrain, Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3)
    
    guassian = GaussianNB()
    guassian.fit(Xtrain, Ytrain)
    Ypredict= guassian.predict(Xtest)
    #countOfMismatched : int = 0
    #for x in range(0,len(Ypredict)):
     #       if(Ypredict[x] != Ytest[x]):
     #           countOfMismatched +=1
     #           print(f'_____count of mismathed is {countOfMismatched} of total {len(Ypredict)}')
      #          print(countOfMismatched * 100 / len(Ypredict))
       #         e = countOfMismatched * 100 / len(Ypredict)
         #       if(e < modelno): 
          #        bestmodel = guassian
           #       modelno = e
 #   return(bestmodel)


    from sklearn import metrics
    accuracy= metrics.accuracy_score(Ytest, Ypredict)
    print('_____________________________ACCURACY____________________________')
    print(f'Your GaussianNB Accuracy is: {accuracy}')
    print('_________________PREDICTION BASED ON YOUR INPUTS_________________')
    print(guassian.predict([[109,55,102,169,51,6,241,27,26,165,265,870,247,84,10,11,184,183]]))
