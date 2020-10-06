# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:35:55 2020

@author: Azizi
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
def GetBestClassifier(count=5):

    dataset = pd.read_csv('vehicle_csv.csv')
    print('\n________ Dataset _________')
    print(dataset)
    bestmodel = GaussianNB()
    modelno = 100
    X= dataset[['COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO',
            'PR.AXIS_ASPECT_RATIO','MAX.LENGTH_ASPECT_RATIO','SCATTER_RATIO',
            'ELONGATEDNESS','PR.AXIS_RECTANGULARITY','MAX.LENGTH_RECTANGULARITY',
            'SCALED_VARIANCE_MAJOR','SCALED_VARIANCE_MINOR','SCALED_RADIUS_OF_GYRATION',
            'SKEWNESS_ABOUT_MAJOR','SKEWNESS_ABOUT_MINOR','KURTOSIS_ABOUT_MAJOR',
            'KURTOSIS_ABOUT_MINOR','HOLLOWS_RATIO']]
    Y= dataset['Class']
    for i in range(count):          
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3)       
        gaussian = GaussianNB()
        gaussian.fit(Xtrain, Ytrain)
        Ypredict= gaussian.predict(Xtest)
        Ytrainpredict= gaussian.predict(Xtrain)
        accuracytrain= metrics.accuracy_score(Ytrain, Ytrainpredict)
        accuracytest= metrics.accuracy_score(Ytest, Ypredict)
        print('_____________________________Train ACCURACY____________________________')
        print(f'Your RandomForest Accuracy is: {accuracytrain}')
        print('_____________________________Test ACCURACY____________________________')
        print(f'Your RandomForest Accuracy is: {accuracytest}')
        countofMismatched= 0
        for x in range(0,len(Ypredict)):
            if(Ypredict[x]!= Ytest.tolist()[x]):
                 countofMismatched +=1
            e = countofMismatched * 100/len(Ypredict)
            if e < modelno: 
             bestmodel = gaussian
             modelno = e
    return(bestmodel)
gaussian1=GetBestClassifier(10)
print('_________________PREDICTION BASED ON YOUR INPUTS_________________')
print(gaussian1.predict([[109,55,102,169,51,6,241,27,26,165,265,870,247,84,10,11,184,183]]))
