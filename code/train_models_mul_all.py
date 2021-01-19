import numpy as np
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

def data_cv(type,data_path1,data_path2):

    train_data_ = pd.read_csv(os.path.join(data_path1 , type + '_data.csv'))
    train_data_des = pd.read_csv(os.path.join(data_path2 , 'descriptors' + '_data.csv'))

    columns_des = train_data_des.columns.tolist( )
    train_x_des = train_data_des[columns_des[1 :]]

    train_data = pd.merge(train_data_ , train_x_des , left_index=True , right_index=True , sort=False)

    train_val, test_val = train_test_split( train_data , test_size=0.15 , random_state=1 )

    columns = train_data.columns.tolist( )

    train_x = np.array(train_data[columns[1 :]])
    train_y = np.array(train_data[columns[0]])
    val_train_x= np.array(train_val[columns[1 :]])
    val_train_y = np.array(train_val[columns[0]])
    val_test_x = np.array(test_val[columns[1 :]])
    val_test_y = np.array(test_val[columns[0]])

    return train_x , train_y , val_train_x , val_train_y , val_test_x , val_test_y

def lr_cv(x_train,y_train,type,save_path):

    lr_fit = LogisticRegressionCV(cv=5 , solver='lbfgs' , max_iter=1000 , multi_class="ovr" , n_jobs=-1).fit(x_train , y_train)

    score_train = lr_fit.score(x_train , y_train)
    print('Accuracy of logistic regression classifier on train set:' , score_train)

    savemodel = os.path.join(save_path , type + '_classifier_all.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(lr_fit , pickle_file)

    score = pd.DataFrame([score_train])
    score.columns = ['Train']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def knn_cv(x_train,y_train, type,save_path):
    knn = KNeighborsClassifier()
    grid_params = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['distance'], 'metric': ['euclidean']}
    knn_cv = GridSearchCV(knn, grid_params, cv=5, verbose = 1, n_jobs = -1)

    knn_cv.fit(x_train , y_train)

    knn_ = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
    knn_fp = knn_.fit(x_train , y_train)

    score_train = knn_fp.score(x_train , y_train)
    print('Accuracy of logistic regression classifier on train set:' , score_train)

    savemodel = os.path.join(save_path , type + '_classifier_all.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(knn_fp , pickle_file)

    score = pd.DataFrame([score_train])
    score.columns = ['Train']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def gb_cv(x_train,y_train,type,save_path):
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

    savemodel = os.path.join(save_path , type + '_classifier_all.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(method_fp , pickle_file)

    score = pd.DataFrame([score_train])
    score.columns = ['Train']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def rf_cv(x_train,y_train,type,save_path):
    rfr = RandomForestClassifier( )
    grid_params = {'min_samples_split' : list((2 , 4 , 6)) , 'n_estimators' : list((10 , 50 , 100 , 200 , 500))}
    method_cv = GridSearchCV(rfr , grid_params , cv=5 , verbose=1 , n_jobs=-1)

    method_cv.fit(x_train , y_train)

    method_ = RandomForestClassifier(min_samples_split=method_cv.best_params_['min_samples_split'],
                                     n_estimators=method_cv.best_params_['n_estimators'])
    method_fp = method_.fit(x_train , y_train)

    score_train = method_fp.score(x_train , y_train)
    print('Accuracy of logistic regression classifier on train set:' , score_train)

    savemodel = os.path.join(save_path , type + '_classifier_all.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(method_fp , pickle_file)

    score = pd.DataFrame([score_train])
    score.columns = ['Train']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

def nn_cv(x_train,y_train,type,save_path):
    mlpr = MLPClassifier( )
    param_grid = {'learning_rate_init' : np.logspace(-5 , -1 , 5) , 'alpha' : np.logspace(-5 , 3 , 5)}
    method_cv = GridSearchCV(mlpr , param_grid , cv=5 , n_jobs=-1 , verbose=1)

    method_cv.fit(x_train , y_train)

    method_ = MLPClassifier(learning_rate_init=method_cv.best_params_['learning_rate_init'],
                           alpha=method_cv.best_params_['alpha'])
    method_fp = method_.fit(x_train , y_train)

    score_train = method_fp.score(x_train , y_train)
    print('Accuracy of logistic regression classifier on train set:' , score_train)

    savemodel = os.path.join(save_path , type + '_classifier_all.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(method_fp , pickle_file)

    score = pd.DataFrame([score_train])
    score.columns = ['Train']
    score.to_csv(os.path.join(save_path , type + '_score.csv'))

if __name__ == '__main__' :
    dataLocation = '../data'

    data_path1 = (os.path.join ( dataLocation , 'pca_data_all' ))
    data_path2 = (os.path.join ( dataLocation , 'ss_data_all' ))

    resultSaveLocation = '../results/mul_all_result/'
    if not os.path.exists ( resultSaveLocation ) :
        os.makedirs ( resultSaveLocation )

    x_morg_train, y_morg_train, x_morg_train_val, \
    x_morg_val, y_morg_train_val, y_morg_val  = data_cv('morg',data_path1,data_path2)

    x_rd_train, y_rd_train, x_rd_train_val, \
    x_rd_val, y_rd_train_val, y_rd_val = data_cv('rd',data_path1,data_path2)

    x_AP_train, y_AP_train, x_AP_train_val, \
    x_AP_val, y_AP_train_val, y_AP_val = data_cv('AP',data_path1,data_path2)

    x_torsion_train, y_torsion_train, x_torsion_train_val, \
    x_torsion_val, y_torsion_train_val, y_torsion_val =data_cv('torsion',data_path1,data_path2)

    print ( 'Logistic regression' )

    save_path = (os.path.join(resultSaveLocation,'LR'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_cv(x_morg_train,y_morg_train,'morg',save_path)
    lr_cv(x_rd_train,y_rd_train,'rd',save_path)
    lr_cv(x_AP_train,y_AP_train,'AP',save_path)
    lr_cv(x_torsion_train,y_torsion_train,'torsion',save_path)

    print ( 'K-nearest neighbor' )

    save_path = (os.path.join(resultSaveLocation,'KNN'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    knn_cv(x_morg_train,y_morg_train,'morg',save_path)
    knn_cv(x_rd_train,y_rd_train,'rd',save_path)
    knn_cv(x_AP_train,y_AP_train,'AP',save_path)
    knn_cv(x_torsion_train,y_torsion_train,'torsion',save_path)

    print ( 'Gradient boosting' )

    save_path = (os.path.join(resultSaveLocation,'GB'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gb_cv(x_morg_train,y_morg_train,'morg',save_path)
    gb_cv(x_rd_train,y_rd_train,'rd',save_path)
    gb_cv(x_AP_train,y_AP_train,'AP',save_path)
    gb_cv(x_torsion_train,y_torsion_train,'torsion',save_path)

    print ( 'Random Forest' )

    save_path = (os.path.join(resultSaveLocation,'RF'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rf_cv(x_morg_train,y_morg_train,'morg',save_path)
    rf_cv(x_rd_train,y_rd_train,'rd',save_path)
    rf_cv(x_AP_train,y_AP_train,'AP',save_path)
    rf_cv(x_torsion_train,y_torsion_train,'torsion',save_path)

    print ( 'Neural network' )

    save_path = (os.path.join(resultSaveLocation,'NN'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    nn_cv(x_morg_train,y_morg_train,'morg',save_path)
    nn_cv(x_rd_train,y_rd_train,'rd',save_path)
    nn_cv(x_AP_train,y_AP_train,'AP',save_path)
    nn_cv(x_torsion_train,y_torsion_train,'torsion',save_path)

    pass


