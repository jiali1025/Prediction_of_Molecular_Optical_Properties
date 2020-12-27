# general and data handling
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

def data_cv(nfold,type,data_path):

    train_x = []
    test_x = []
    train_y = []
    test_y = []
    val_train_x = []
    val_train_y = []
    val_test_x = []
    val_test_y = []
    for i in range(nfold) :

        train_data = pd.read_csv(os.path.join(data_path , type + '_train_' + str(i) + '.csv'))
        test_data = pd.read_csv(os.path.join(data_path , type + '_test_' + str(i) + '.csv'))

        train_val, test_val = train_test_split( train_data , test_size=0.15 , random_state=1 )

        columns = train_data.columns.tolist( )

        train_x_i = np.array(train_data[columns[1 :]])
        train_y_i = np.array(train_data[columns[0]])
        test_x_i = np.array(test_data[columns[1 :]])
        test_y_i = np.array(test_data[columns[0]])
        val_train_x_i = np.array(train_val[columns[1 :]])
        val_train_y_i = np.array(train_val[columns[0]])
        val_test_x_i = np.array(test_val[columns[1 :]])
        val_test_y_i = np.array(test_val[columns[0]])

        train_x.append(train_x_i)
        test_x.append(test_x_i)
        train_y.append(train_y_i)
        test_y.append(test_y_i)
        val_train_x.append(val_train_x_i)
        val_train_y.append(val_train_y_i)
        val_test_x.append(val_test_x_i)
        val_test_y.append(val_test_y_i)

    return train_x , test_x , train_y , test_y , val_train_x , val_train_y , val_test_x , val_test_y


def lr_cv(x_train10,y_train10,x_test10,y_test10,type,save_path,nfold):
    score_train_ = []
    score_test_ = []
    for i in range(nfold) :
        x_train = x_train10[i]
        y_train = y_train10[i]
        x_test = x_test10[i]
        y_test = y_test10[i]

        lr_fit = LogisticRegressionCV(cv=5 , solver='lbfgs' , max_iter=1000 , multi_class="ovr" , n_jobs=-1).fit(
            x_train , y_train)

        score_train = lr_fit.score(x_train , y_train)
        print('Accuracy of logistic regression classifier on train set:' , score_train)

        score_test = lr_fit.score(x_test , y_test)
        print('Accuracy of logistic regression classifier on test set:' , score_test)

        y_pred = lr_fit.predict(x_test)

        result_y = np.concatenate((np.reshape(y_test , (-1 , 1)) , np.reshape(y_pred , (-1 , 1))) , axis=1)
        result_y = pd.DataFrame(result_y)

        result_y.columns = ['Ytest' , 'Ypred']
        result_y.to_csv(os.path.join(save_path , type + '_' + str(i) + '_resultsY.csv'))

        score_train_.append(score_train)
        score_test_.append(score_test)

    score_train_.append(str(np.mean(score_train_)) + '+-' + str(np.std(score_train_)))
    score_test_.append(str(np.mean(score_test_)) + '+-' + str(np.std(score_test_)))

    score = np.concatenate((np.reshape(score_train_ , (-1 , 1)) , np.reshape(score_test_ , (-1 , 1))) , axis=1)
    score = pd.DataFrame(score)

    score.columns = ['Train' , 'Test']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def knn_cv(x_train10,y_train10,x_test10,y_test10,type,save_path,nfold):
    knn = KNeighborsClassifier()
    grid_params = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['distance'], 'metric': ['euclidean']}
    knn_cv = GridSearchCV(knn, grid_params, cv=5, verbose = 1, n_jobs = -1)
    score_train_ = []
    score_test_ = []
    for i in range(nfold) :
        x_train = x_train10[i]
        y_train = y_train10[i]
        x_test = x_test10[i]
        y_test = y_test10[i]

        knn_cv.fit(x_train , y_train)

        knn_ = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
        knn_fp = knn_.fit(x_train , y_train)

        score_train = knn_fp.score(x_train , y_train)
        print('Accuracy of logistic regression classifier on train set:' , score_train)

        score_test = knn_fp.score(x_test , y_test)
        print('Accuracy of logistic regression classifier on test set:' , score_test)

        y_pred = knn_fp.predict(x_test)

        result_y = np.concatenate((np.reshape(y_test , (-1 , 1)) , np.reshape(y_pred , (-1 , 1))) , axis=1)
        result_y = pd.DataFrame(result_y)

        result_y.columns = ['Ytest' , 'Ypred']
        result_y.to_csv(os.path.join(save_path , type + '_' + str(i) + '_resultsY.csv'))

        score_train_.append(score_train)
        score_test_.append(score_test)

    score_train_.append(str(np.mean(score_train_)) + '+-' + str(np.std(score_train_)))
    score_test_.append(str(np.mean(score_test_)) + '+-' + str(np.std(score_test_)))

    score = np.concatenate((np.reshape(score_train_ , (-1 , 1)) , np.reshape(score_test_ , (-1 , 1))) , axis=1)
    score = pd.DataFrame(score)

    score.columns = ['Train' , 'Test']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def gb_cv(x_train10,y_train10,x_test10,y_test10,type,save_path,nfold):
    params_gb = {
        "loss" : ["deviance"] ,
        "learning_rate" : [1.0] ,
        "min_samples_split" : np.linspace(0.1 , 0.5 , 12) ,
        "min_samples_leaf" : np.linspace(0.1 , 0.5 , 12) ,
        "max_depth" : [3 , 5 , 7] ,
        "max_features" : ["sqrt"] ,
        "criterion" : ["friedman_mse"] ,
        "subsample" : [0.5 , 0.75 , 0.95] ,
        "n_estimators" : [100]
    }
    method_cv = GridSearchCV(GradientBoostingClassifier( ) , params_gb , cv=5 , n_jobs=-1 , verbose=1)
    score_train_ = []
    score_test_ = []
    for i in range(nfold) :
        x_train = x_train10[i]
        y_train = y_train10[i]
        x_test = x_test10[i]
        y_test = y_test10[i]

        method_cv.fit(x_train , y_train)

        method_ = GradientBoostingClassifier(loss = "deviance",
                                             learning_rate=1.0,
                                             min_samples_split=method_cv.best_params_["min_samples_split"],
                                             min_samples_leaf=method_cv.best_params_["min_samples_leaf"],
                                             max_depth=method_cv.best_params_["max_depth"] ,
                                             max_features="sqrt",
                                             criterion="friedman_mse",
                                             subsample=method_cv.best_params_["subsample"],
                                             n_estimators=100)
        method_fp = method_.fit(x_train , y_train)

        score_train = method_fp.score(x_train , y_train)
        print('Accuracy of logistic regression classifier on train set:' , score_train)

        score_test = method_fp.score(x_test , y_test)
        print('Accuracy of logistic regression classifier on test set:' , score_test)

        y_pred = method_fp.predict(x_test)

        result_y = np.concatenate((np.reshape(y_test , (-1 , 1)) , np.reshape(y_pred , (-1 , 1))) , axis=1)
        result_y = pd.DataFrame(result_y)

        result_y.columns = ['Ytest' , 'Ypred']
        result_y.to_csv(os.path.join(save_path , type + '_' + str(i) + '_resultsY.csv'))

        score_train_.append(score_train)
        score_test_.append(score_test)

    score_train_.append(str(np.mean(score_train_)) + '+-' + str(np.std(score_train_)))
    score_test_.append(str(np.mean(score_test_)) + '+-' + str(np.std(score_test_)))

    score = np.concatenate((np.reshape(score_train_ , (-1 , 1)) , np.reshape(score_test_ , (-1 , 1))) , axis=1)
    score = pd.DataFrame(score)

    score.columns = ['Train' , 'Test']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def rf_cv(x_train10,y_train10,x_test10,y_test10,type,save_path,nfold):
    rfr = RandomForestClassifier( )
    grid_params = {'min_samples_split' : list((2 , 4 , 6)) , 'n_estimators' : list((10 , 50 , 100 , 200 , 500))}
    method_cv = GridSearchCV(rfr , grid_params , cv=5 , verbose=1 , n_jobs=-1)
    score_train_ = []
    score_test_ = []
    for i in range(nfold) :
        x_train = x_train10[i]
        y_train = y_train10[i]
        x_test = x_test10[i]
        y_test = y_test10[i]

        method_cv.fit(x_train , y_train)

        method_ = RandomForestClassifier(min_samples_split=method_cv.best_params_['min_samples_split'],
                                         n_estimators=method_cv.best_params_['n_estimators'])
        method_fp = method_.fit(x_train , y_train)

        score_train = method_fp.score(x_train , y_train)
        print('Accuracy of logistic regression classifier on train set:' , score_train)

        score_test = method_fp.score(x_test , y_test)
        print('Accuracy of logistic regression classifier on test set:' , score_test)

        y_pred = method_fp.predict(x_test)

        result_y = np.concatenate((np.reshape(y_test , (-1 , 1)) , np.reshape(y_pred , (-1 , 1))) , axis=1)
        result_y = pd.DataFrame(result_y)

        result_y.columns = ['Ytest' , 'Ypred']
        result_y.to_csv(os.path.join(save_path , type + '_' + str(i) + '_resultsY.csv'))

        score_train_.append(score_train)
        score_test_.append(score_test)

    score_train_.append(str(np.mean(score_train_)) + '+-' + str(np.std(score_train_)))
    score_test_.append(str(np.mean(score_test_)) + '+-' + str(np.std(score_test_)))

    score = np.concatenate((np.reshape(score_train_ , (-1 , 1)) , np.reshape(score_test_ , (-1 , 1))) , axis=1)
    score = pd.DataFrame(score)

    score.columns = ['Train' , 'Test']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def nn_cv(x_train10,y_train10,x_test10,y_test10,type,save_path,nfold):
    mlpr = MLPClassifier( )
    param_grid = {'learning_rate_init' : np.logspace(-5 , -1 , 5) , 'alpha' : np.logspace(-5 , 3 , 5)}
    method_cv = GridSearchCV(mlpr , param_grid , cv=5 , n_jobs=-1 , verbose=1)

    score_train_ = []
    score_test_ = []
    for i in range(nfold) :
        x_train = x_train10[i]
        y_train = y_train10[i]
        x_test = x_test10[i]
        y_test = y_test10[i]

        method_cv.fit(x_train , y_train)

        method_ = MLPClassifier(learning_rate_init=method_cv.best_params_['learning_rate_init'],
                               alpha=method_cv.best_params_['alpha'])
        method_fp = method_.fit(x_train , y_train)

        score_train = method_fp.score(x_train , y_train)
        print('Accuracy of logistic regression classifier on train set:' , score_train)

        score_test = method_fp.score(x_test , y_test)
        print('Accuracy of logistic regression classifier on test set:' , score_test)

        y_pred = method_fp.predict(x_test)

        result_y = np.concatenate((np.reshape(y_test , (-1 , 1)) , np.reshape(y_pred , (-1 , 1))) , axis=1)
        result_y = pd.DataFrame(result_y)

        result_y.columns = ['Ytest' , 'Ypred']
        result_y.to_csv(os.path.join(save_path , type + '_' + str(i) + '_resultsY.csv'))

        score_train_.append(score_train)
        score_test_.append(score_test)

    score_train_.append(str(np.mean(score_train_)) + '+-' + str(np.std(score_train_)))
    score_test_.append(str(np.mean(score_test_)) + '+-' + str(np.std(score_test_)))

    score = np.concatenate((np.reshape(score_train_ , (-1 , 1)) , np.reshape(score_test_ , (-1 , 1))) , axis=1)
    score = pd.DataFrame(score)

    score.columns = ['Train' , 'Test']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

if __name__ == '__main__':
    dataLocation = '../data'

    data_path = (os.path.join(dataLocation,'ori_data'))

    resultSaveLocation = '../results/ori_result/'
    if not os.path.exists(resultSaveLocation):
        os.makedirs(resultSaveLocation)

    nfold = 10

    x_morg_train10, x_morg_test10, y_morg_train10, y_morg_test10, x_morg_train_val, \
    x_morg_val, y_morg_train_val, y_morg_val  = data_cv(nfold,'morg',data_path)

    x_rd_train10, x_rd_test10, y_rd_train10, y_rd_test10, x_rd_train_val, \
    x_rd_val, y_rd_train_val, y_rd_val = data_cv(nfold,'rd',data_path)

    x_AP_train10, x_AP_test10, y_AP_train10, y_AP_test10, x_AP_train_val, \
    x_AP_val, y_AP_train_val, y_AP_val = data_cv(nfold,'AP',data_path)

    x_torsion_train10, x_torsion_test10, y_torsion_train10, y_torsion_test10, x_torsion_train_val, \
    x_torsion_val, y_torsion_train_val, y_torsion_val =data_cv(nfold,'torsion',data_path)


    print('Logistic regression')

    save_path = (os.path.join(resultSaveLocation,'LR'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_cv(x_morg_train10,y_morg_train10,x_morg_test10,y_morg_test10,'morg',save_path,nfold)
    lr_cv(x_rd_train10,y_rd_train10,x_rd_test10,y_rd_test10,'rd',save_path,nfold)
    lr_cv(x_AP_train10,y_AP_train10,x_AP_test10,y_AP_test10,'AP',save_path,nfold)
    lr_cv(x_torsion_train10,y_torsion_train10,x_torsion_test10,y_torsion_test10,'torsion',save_path,nfold)

    print('K-nearest neighbor')

    save_path = (os.path.join(resultSaveLocation,'KNN'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    knn_cv(x_morg_train10,y_morg_train10,x_morg_test10,y_morg_test10,'morg',save_path,nfold)
    knn_cv(x_rd_train10,y_rd_train10,x_rd_test10,y_rd_test10,'rd',save_path,nfold)
    knn_cv(x_AP_train10,y_AP_train10,x_AP_test10,y_AP_test10,'AP',save_path,nfold)
    knn_cv(x_torsion_train10,y_torsion_train10,x_torsion_test10,y_torsion_test10,'torsion',save_path,nfold)

    print('Gradient boosting')

    save_path = (os.path.join(resultSaveLocation,'GB'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gb_cv(x_morg_train10,y_morg_train10,x_morg_test10,y_morg_test10,'morg',save_path,nfold)
    gb_cv(x_rd_train10,y_rd_train10,x_rd_test10,y_rd_test10,'rd',save_path,nfold)
    gb_cv(x_AP_train10,y_AP_train10,x_AP_test10,y_AP_test10,'AP',save_path,nfold)
    gb_cv(x_torsion_train10,y_torsion_train10,x_torsion_test10,y_torsion_test10,'torsion',save_path,nfold)


    print('Random Forest')

    save_path = (os.path.join(resultSaveLocation,'RF'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rf_cv(x_morg_train10,y_morg_train10,x_morg_test10,y_morg_test10,'morg',save_path,nfold)
    rf_cv(x_rd_train10,y_rd_train10,x_rd_test10,y_rd_test10,'rd',save_path,nfold)
    rf_cv(x_AP_train10,y_AP_train10,x_AP_test10,y_AP_test10,'AP',save_path,nfold)
    rf_cv(x_torsion_train10,y_torsion_train10,x_torsion_test10,y_torsion_test10,'torsion',save_path,nfold)

    print('Neural network')

    save_path = (os.path.join(resultSaveLocation,'NN'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    nn_cv(x_morg_train10,y_morg_train10,x_morg_test10,y_morg_test10,'morg',save_path,nfold)
    nn_cv(x_rd_train10,y_rd_train10,x_rd_test10,y_rd_test10,'rd',save_path,nfold)
    nn_cv(x_AP_train10,y_AP_train10,x_AP_test10,y_AP_test10,'AP',save_path,nfold)
    nn_cv(x_torsion_train10,y_torsion_train10,x_torsion_test10,y_torsion_test10,'torsion',save_path,nfold)

    pass


