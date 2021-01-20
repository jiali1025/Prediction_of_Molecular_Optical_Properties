import numpy as np
import pandas as pd
import os
import pickle

from sklearn import metrics
from sklearn.metrics import classification_report


def acc_cv ( x , nfold ):
    score_acc = [ ]
    score_AUC = [ ]
    score_F1 = [ ]

    for i in range ( nfold ):
        y_true = x [ 'Ytest' + str ( i ) ]
        y_pred = x [ 'Ypred' + str ( i ) ]
        acc = metrics.accuracy_score ( y_true , y_pred )

        if 0 < sum ( y_true ) < len ( y_true ):
            auc = metrics.roc_auc_score ( y_true , y_pred )
        else:
            auc = 1.0

        clrpt = classification_report ( y_true , y_pred , output_dict = True )
        d = clrpt [ "weighted avg" ]
        f1 = d [ 'f1-score' ]

        score_acc.append ( acc )
        score_AUC.append ( auc )
        score_F1.append ( f1 )

    score_acc.append ( str ( np.mean ( score_acc ) ) + '+-' + str ( np.std ( score_acc ) ) )
    score_AUC.append ( str ( np.mean ( score_AUC ) ) + '+-' + str ( np.std ( score_AUC ) ) )
    score_F1.append ( str ( np.mean ( score_F1 ) ) + '+-' + str ( np.std ( score_F1 ) ) )

    score = np.concatenate ( (np.reshape ( score_acc , (-1 , 1) ) , np.reshape ( score_AUC , (-1 , 1) ) ,
                              np.reshape ( score_F1 , (-1 , 1) )) , axis = 1 )
    score = pd.DataFrame ( score )

    score.columns = [ 'ACC' , 'AUC' , 'F1' ]

    return score


if __name__ == '__main__':

    dataLocation = '../data'

    resultSaveLocations = [ '../results/ori_result/' , '../results/mul_result/' ]
    nfold = 10

    for resultSaveLocation in resultSaveLocations:

        resultSavePath = os.path.join ( resultSaveLocation , 'Ensem' )

        if not os.path.exists ( resultSavePath ):
            os.makedirs ( resultSavePath )

        with open ( os.path.join ( resultSavePath , 'ensem.pkl' ) , 'rb' ) as file:
            data = pickle.load ( file )

        score = acc_cv ( data , nfold )
        score.to_csv ( os.path.join ( resultSavePath , 'score.csv' ) , index = False )

        pass
