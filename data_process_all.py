# general and data handling
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

SS = StandardScaler()
pca = PCA()

def data_cv(data,type,save_path_ori, save_path_ss, save_path_pca):

    columns = data.columns.tolist( )
    data_x = np.array(data[columns[1 :]] )
    data_y = np.array(data[columns[0]])

    SS.fit(data_x)
    data_x_ss = SS.transform(data_x )

    pca.fit(data_x_ss)
    data_x_pca = pca.transform(data_x_ss)

    savemodel = os.path.join(save_path_ss , type + '_ss.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(SS , pickle_file)

    savemodel = os.path.join(save_path_pca , type + '_pca.pkl')
    with open(savemodel , 'wb') as pickle_file :
        pickle.dump(pca , pickle_file)

    data_x = pd.DataFrame( data_x )
    data_y = pd.DataFrame(data_y)

    data_ori = pd.merge( data_y , data_x , left_index=True , right_index=True , sort=False )
    data_ori.columns = columns

    data_ori.to_csv( os.path.join( save_path_ori , type + '_data.csv' ) , index=False )

    data_x_ss = pd.DataFrame( data_x_ss )

    data_ss = pd.merge( data_y , data_x_ss , left_index=True , right_index=True , sort=False )
    data_ss.columns = columns

    data_ss.to_csv( os.path.join( save_path_ss , type + '_data.csv' ) , index=False )

    if type == 'descriptors':
        columns_pca = columns
    else:
        columns_pca = list( range( data_x_pca.shape[1] ) )
        columns_pca = ['y'] + columns_pca
    data_x_pca = pd.DataFrame( data_x_pca )

    data_pca = pd.merge(data_y , data_x_pca , left_index=True , right_index=True , sort=False )
    data_pca.columns = columns_pca

    data_pca.to_csv( os.path.join( save_path_pca , type + '_data.csv' ) , index=False )

    ratio = pca.explained_variance_ratio_
    ratio_sum = sum( ratio )

    ratio = pd.DataFrame( np.reshape( ratio , (1 , -1) ) )
    ratio = pd.DataFrame( ratio )
    ratio.columns = columns_pca[1:]
    ratio.to_csv( os.path.join( save_path_pca , type + '_ratio.csv' ) , index=False )

    ratio_score = pd.DataFrame( [ratio_sum] , columns=['score'] )
    ratio_score.to_csv( os.path.join( save_path_pca , type + '_pca.csv' ) , index=False )

if __name__ == '__main__':

    dataLocation = '../data'

    save_path_ori = (os.path.join ( dataLocation , 'ori_data_all' ))
    if not os.path.exists ( save_path_ori ) :
        os.makedirs ( save_path_ori )

    save_path_ss = (os.path.join ( dataLocation , 'scaler_data_all' ))
    if not os.path.exists ( save_path_ss ) :
        os.makedirs ( save_path_ss )

    save_path_pca = (os.path.join ( dataLocation , 'pca_data_all' ))
    if not os.path.exists ( save_path_pca ) :
        os.makedirs ( save_path_pca )

    morg_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_morg.csv' ) )
    rd_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_rd.csv' ) )
    AP_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_AP.csv' ) )
    torsion_sample = pd.read_csv ( os.path.join ( dataLocation , 'x_torsion.csv' ) )
    descriptors_sample = pd.read_csv ( os.path.join ( dataLocation , 'descriptors.csv' ) )

    descriptors_item = pd.read_csv ( os.path.join ( dataLocation , 'descriptors_sec.csv' ) )
    descriptors_sec = [ 'y' ] + descriptors_item [ 'descriptors' ].to_list ( )
    descriptors_data = descriptors_sample [ descriptors_sec ]

    data_cv(morg_sample, 'morg' ,save_path_ori , save_path_ss, save_path_pca )
    data_cv(rd_sample, 'rd' , save_path_ori , save_path_ss, save_path_pca )
    data_cv(AP_sample, 'AP' , save_path_ori , save_path_ss, save_path_pca )
    data_cv(torsion_sample, 'torsion' , save_path_ori , save_path_ss, save_path_pca )
    data_cv(descriptors_data, 'descriptors' , save_path_ori , save_path_ss, save_path_pca )
