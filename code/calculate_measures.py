import numpy as np
import pandas as pd
import os

from sklearn import metrics
from sklearn.metrics import classification_report


def auc_f1_cv ( result_path , method , type , nfold ):
    save_path = (os.path.join ( result_path , method ))

    score_AUC = [ ]
    score_F1 = [ ]

    for i in range ( nfold ):
        y = pd.read_csv ( os.path.join ( save_path , type + '_' + str ( i ) + '_resultsY.csv' ) )
        y_test = y [ 'Ytest' ]
        y_pred = y [ 'Ypred' ]

        if 0 < sum ( y_test ) < len ( y_test ):
            auc = metrics.roc_auc_score ( y_test , y_pred )
        else:
            auc = 1.0

        clrpt = classification_report ( y_test , y_pred , output_dict = True )
        d = clrpt [ "weighted avg" ]
        f1 = d [ 'f1-score' ]

        score_AUC.append ( auc )
        score_F1.append ( f1 )

    score_AUC.append ( str ( np.mean ( score_AUC ) ) + '+-' + str ( np.std ( score_AUC ) ) )
    score_F1.append ( str ( np.mean ( score_F1 ) ) + '+-' + str ( np.std ( score_F1 ) ) )

    score = np.concatenate ( (np.reshape ( score_AUC , (-1 , 1) ) , np.reshape ( score_F1 , (-1 , 1) )) , axis = 1 )
    score = pd.DataFrame ( score )

    score.columns = [ 'AUC' , 'F1' ]
    score.to_csv ( os.path.join ( save_path , type + '_AUCF1.csv' ) )


if __name__ == '__main__':

    resultSaveLocations = [ '../results/ori_result/' , '../results/mul_result/' ]
    nfold = 10

    for resultSaveLocation in resultSaveLocations:

        auc_f1_cv ( resultSaveLocation , 'LR' , 'morg' , nfold )
        auc_f1_cv ( resultSaveLocation , 'LR' , 'rd' , nfold )
        auc_f1_cv ( resultSaveLocation , 'LR' , 'AP' , nfold )
        auc_f1_cv ( resultSaveLocation , 'LR' , 'torsion' , nfold )

        auc_f1_cv ( resultSaveLocation , 'KNN' , 'morg' , nfold )
        auc_f1_cv ( resultSaveLocation , 'KNN' , 'rd' , nfold )
        auc_f1_cv ( resultSaveLocation , 'KNN' , 'AP' , nfold )
        auc_f1_cv ( resultSaveLocation , 'KNN' , 'torsion' , nfold )

        auc_f1_cv ( resultSaveLocation , 'GB' , 'morg' , nfold )
        auc_f1_cv ( resultSaveLocation , 'GB' , 'rd' , nfold )
        auc_f1_cv ( resultSaveLocation , 'GB' , 'AP' , nfold )
        auc_f1_cv ( resultSaveLocation , 'GB' , 'torsion' , nfold )

        auc_f1_cv ( resultSaveLocation , 'RF' , 'morg' , nfold )
        auc_f1_cv ( resultSaveLocation , 'RF' , 'rd' , nfold )
        auc_f1_cv ( resultSaveLocation , 'RF' , 'AP' , nfold )
        auc_f1_cv ( resultSaveLocation , 'RF' , 'torsion' , nfold )

        auc_f1_cv ( resultSaveLocation , 'NN' , 'morg' , nfold )
        auc_f1_cv ( resultSaveLocation , 'NN' , 'rd' , nfold )
        auc_f1_cv ( resultSaveLocation , 'NN' , 'AP' , nfold )
        auc_f1_cv ( resultSaveLocation , 'NN' , 'torsion' , nfold )

        if resultSaveLocation == '../results/ori_result/':
            auc_f1_cv ( resultSaveLocation , 'LR' , 'descriptors' , nfold )
            auc_f1_cv ( resultSaveLocation , 'KNN' , 'descriptors' , nfold )
            auc_f1_cv ( resultSaveLocation , 'GB' , 'descriptors' , nfold )
            auc_f1_cv ( resultSaveLocation , 'RF' , 'descriptors' , nfold )
            auc_f1_cv ( resultSaveLocation , 'NN' , 'descriptors' , nfold )

        pass
