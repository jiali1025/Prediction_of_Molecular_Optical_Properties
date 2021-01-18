# general and data handling
import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings ( 'ignore' )

SS = StandardScaler ( )
pca = PCA ( )


def data_cv ( nfold , data , type , inds_path , save_path_ori , save_path_ss , save_path_pca ):
    ratio_score = [ ]
    for i in range ( nfold ):

        train = pd.read_csv ( os.path.join ( inds_path , 'train_ind_' + str ( i ) + '.csv' ) )
        test = pd.read_csv ( os.path.join ( inds_path , 'test_ind_' + str ( i ) + '.csv' ) )
        train = np.array ( train [ 'train_ind' ] )
        test = np.array ( test [ 'test_ind' ] )

        columns = data.columns.tolist ( )
        data_x = np.array ( data [ columns [ 1: ] ] )
        data_y = np.array ( data [ columns [ 0 ] ] )

        train_x = data_x [ train ]
        train_y = data_y [ train ]
        test_x = data_x [ test ]
        test_y = data_y [ test ]

        SS.fit ( train_x )
        train_x_ss = SS.transform ( train_x )
        test_x_ss = SS.transform ( test_x )

        pca.fit ( train_x_ss )
        train_x_pca = pca.transform ( train_x_ss )
        test_x_pca = pca.transform ( test_x_ss )

        train_x = pd.DataFrame ( train_x )
        train_y = pd.DataFrame ( train_y )
        test_x = pd.DataFrame ( test_x )
        test_y = pd.DataFrame ( test_y )

        train_ori = pd.merge ( train_y , train_x , left_index = True , right_index = True , sort = False )
        train_ori.columns = columns
        test_ori = pd.merge ( test_y , test_x , left_index = True , right_index = True , sort = False )
        test_ori.columns = columns

        train_ori.to_csv ( os.path.join ( save_path_ori , type + '_train_' + str ( i ) + '.csv' ) , index = False )
        test_ori.to_csv ( os.path.join ( save_path_ori , type + '_test_' + str ( i ) + '.csv' ) , index = False )

        train_x_ss = pd.DataFrame ( train_x_ss )
        test_x_ss = pd.DataFrame ( test_x_ss )

        train_ss = pd.merge ( train_y , train_x_ss , left_index = True , right_index = True , sort = False )
        train_ss.columns = columns
        test_ss = pd.merge ( test_y , test_x_ss , left_index = True , right_index = True , sort = False )
        test_ss.columns = columns

        train_ss.to_csv ( os.path.join ( save_path_ss , type + '_train_' + str ( i ) + '.csv' ) , index = False )
        test_ss.to_csv ( os.path.join ( save_path_ss , type + '_test_' + str ( i ) + '.csv' ) , index = False )

        if type == 'descriptors':
            columns_pca = columns
        else:
            columns_pca = list ( range ( train_x_pca.shape [ 1 ] ) )
            columns_pca = [ 'y' ] + columns_pca
        train_x_pca = pd.DataFrame ( train_x_pca )
        test_x_pca = pd.DataFrame ( test_x_pca )

        train_pca = pd.merge ( train_y , train_x_pca , left_index = True , right_index = True , sort = False )
        train_pca.columns = columns_pca
        test_pca = pd.merge ( test_y , test_x_pca , left_index = True , right_index = True , sort = False )
        test_pca.columns = columns_pca

        train_pca.to_csv ( os.path.join ( save_path_pca , type + '_train_' + str ( i ) + '.csv' ) , index = False )
        test_pca.to_csv ( os.path.join ( save_path_pca , type + '_test_' + str ( i ) + '.csv' ) , index = False )

        ratio = pca.explained_variance_ratio_
        ratio_sum = sum ( ratio )

        ratio = pd.DataFrame ( np.reshape ( ratio , (1 , -1) ) )
        ratio = pd.DataFrame ( ratio )
        ratio.columns = columns_pca [ 1: ]
        ratio.to_csv ( os.path.join ( save_path_pca , type + '_ratio_' + str ( i ) + '.csv' ) , index = False )

        ratio_score.append ( [ i , ratio_sum ] )

    ratio_score = pd.DataFrame ( ratio_score , columns = [ 'nfold' , 'score' ] )
    ratio_score.to_csv ( os.path.join ( save_path_pca , type + '_pca' + '.csv' ) , index = False )


if __name__ == '__main__':

    dataLocation = '../data'

    inds_path = os.path.join ( dataLocation , 'cv_inds' )

    save_path_ori = (os.path.join ( dataLocation , 'ori_data' ))
    if not os.path.exists ( save_path_ori ):
        os.makedirs ( save_path_ori )

    save_path_ss = (os.path.join ( dataLocation , 'scaler_data' ))
    if not os.path.exists ( save_path_ss ):
        os.makedirs ( save_path_ss )

    save_path_pca = (os.path.join ( dataLocation , 'pca_data' ))
    if not os.path.exists ( save_path_pca ):
        os.makedirs ( save_path_pca )

    morg_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_morg.csv' ) )
    rd_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_rd.csv' ) )
    AP_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_AP.csv' ) )
    torsion_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_torsion.csv' ) )
    descriptors_sample = pd.read_csv ( os.path.join ( dataLocation , 'descriptors.csv' ) )

    descriptors_item = pd.read_csv ( os.path.join ( dataLocation , 'descriptors_sec.csv' ) )
    descriptors_sec = [ 'y' ] + descriptors_item [ 'descriptors' ].to_list ( )
    descriptors_data = descriptors_sample [ descriptors_sec ]

    nfold = 10

    data_cv ( nfold , morg_sample , 'morg' , inds_path , save_path_ori , save_path_ss , save_path_pca )
    data_cv ( nfold , rd_sample , 'rd' , inds_path , save_path_ori , save_path_ss , save_path_pca )
    data_cv ( nfold , AP_sample , 'AP' , inds_path , save_path_ori , save_path_ss , save_path_pca )
    data_cv ( nfold , torsion_sample , 'torsion' , inds_path , save_path_ori , save_path_ss , save_path_pca )
    data_cv ( nfold , descriptors_data , 'descriptors' , inds_path , save_path_ori , save_path_ss , save_path_pca )
