import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.metrics import confusion_matrix



def k_fold(x, y,algo,n_splits=5,shuffle=True,random_state=4,average='binary'):
    precision = []
    recall = []
    F1 = []
    accuracy = []
    specificity=[]
    
    cv = KFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    
    for train_index, test_index in cv.split(x):
        X_train, X_test, Y_train, Y_test = x[train_index], x[test_index], y[train_index], y[test_index]
        algo.fit(X_train, Y_train)
        
        precision.append(precision_score(Y_test,algo.predict(X_test),average=average))
        recall.append(recall_score(Y_test,algo.predict(X_test),average=average))
        F1.append(f1_score(Y_test,algo.predict(X_test),average=average))
        accuracy.append(algo.score(X_test,Y_test))

        tn, fp, fn, tp = confusion_matrix(Y_test,algo.predict(X_test)).ravel()
        specificity.append(tn / (tn+fp))

    return np.array([np.array(accuracy).mean(),np.array(recall).mean(),np.array(specificity).mean(),np.array(precision).mean(),np.array(F1).mean()])

algo={

}


def k_fold_results(x,y,algo=algo):
    df=pd.DataFrame({
        
        },index=['accuracy','recall','specificity','precision','F1'])

    for item in algo.items():
        df[item[0]]=k_fold(x,y,item[1])
    
    
    return df