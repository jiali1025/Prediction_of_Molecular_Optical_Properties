import os
import pickle
import numpy as np
import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import warnings
warnings.filterwarnings('ignore')

def clean_data(data,name):

    dataClean = data.drop_duplicates(['smiles'])
    dataClean.reset_index(drop=True , inplace=True)
    dataClean.to_csv(dataLocation + '2-Clean' + name, index=False)
    print(dataClean['smiles'].nunique())
    print(dataClean['smiles'].count())
    print(dataClean.shape[0])

    dataDuplicate = data[data.duplicated()]
    dataDuplicate['ind'] = dataDuplicate.index
    dataDuplicate.reset_index(drop=True , inplace=True)
    dataDuplicate.to_csv(dataLocation + '3-Duplicate' + name, index=False)
    print(dataDuplicate.shape[0])

    return dataClean

def mol_data(data,name):

    columns = list(range(2048))

    data["mol"] = [Chem.MolFromSmiles(x) for x in data["smiles"]]

    data_na = data[data.isnull().T.any()]
    data_na.to_csv(dataLocation + '4_Null'+ name, index=False)

    data.dropna(axis=0 , how='any' , inplace=True)
    data.reset_index(drop=True , inplace=True)
    data.to_csv(dataLocation + '5_Dropna' + name , index=False)

    data["morg_fp"] = [Chem.GetMorganFingerprintAsBitVect(m , 2 , nBits=2048) for m in data['mol']]

    morg_fp = [Chem.GetMorganFingerprintAsBitVect(m , 2 , nBits=2048) for m in data['mol']]

    morg_fp_np = []
    for fp in morg_fp :
        arr = np.zeros((1 ,))
        DataStructs.ConvertToNumpyArray(fp , arr)
        morg_fp_np.append(arr)

    data["rd_fp"] = [Chem.RDKFingerprint(m) for m in data["mol"]]

    rd_fp = [Chem.RDKFingerprint(m) for m in data["mol"]]

    rd_fp_np = []
    for fp in rd_fp :
        arr = np.zeros((1 ,))
        DataStructs.ConvertToNumpyArray(fp , arr)
        rd_fp_np.append(arr)

    data["AP_fp"] = [Chem.GetHashedAtomPairFingerprintAsBitVect(m) for m in data["mol"]]

    AP_fp = [Chem.GetHashedAtomPairFingerprintAsBitVect(m) for m in data["mol"]]

    AP_fp_np = []
    for fp in AP_fp :
        arr = np.zeros((1 ,))
        DataStructs.ConvertToNumpyArray(fp , arr)
        AP_fp_np.append(arr)

    data["torsion_fp"] = [Chem.GetHashedTopologicalTorsionFingerprintAsBitVect(m) for m in data["mol"]]

    torsion_fp = [Chem.GetHashedTopologicalTorsionFingerprintAsBitVect(m) for m in data["mol"]]

    torsion_fp_np = []
    for fp in torsion_fp :
        arr = np.zeros((1 ,))
        DataStructs.ConvertToNumpyArray(fp , arr)
        torsion_fp_np.append(arr)

    x_morg = morg_fp_np
    x_rd = rd_fp_np
    x_AP = AP_fp_np
    x_torsion = torsion_fp_np

    x_morg = np.array(x_morg)
    x_rd = np.array(x_rd)
    x_AP = np.array(x_AP)
    x_torsion = np.array(x_torsion)

    x_morg = pd.DataFrame(x_morg,columns=columns)
    x_rd = pd.DataFrame(x_rd , columns=columns)
    x_AP = pd.DataFrame(x_AP , columns=columns)
    x_torsion = pd.DataFrame(x_torsion , columns=columns)

    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descriptors = pd.DataFrame([desc_calc.CalcDescriptors(mol) for mol in data['mol']])
    descriptors.columns = descs

    descriptors_item = pd.read_csv('./Dataset/' + '7_descriptors_sec.csv')
    descriptors_sec = descriptors_item['descriptors'].to_list()
    x_descriptors = descriptors[descriptors_sec]

    return x_morg, x_rd, x_AP, x_torsion, x_descriptors

def data_cv(data_x,type,save_path_ss, save_path_pca):

    columns = data_x.columns.tolist( )
    data_x = np.array(data_x)

    savemodel = os.path.join(save_path_ss , type + '_ss.pkl')
    with open(savemodel , 'rb') as file:
        SS = pickle.load(file)

    data_x_ss = SS.transform(data_x)

    savemodel = os.path.join(save_path_pca , type + '_pca.pkl')
    with open(savemodel , 'rb') as file :
        pca = pickle.load(file)

    data_x_pca = pca.transform(data_x)

    data_ori = pd.DataFrame( data_x )
    data_ori.columns = columns

    data_ss = pd.DataFrame( data_x_ss )
    data_ss.columns = columns

    if type == 'descriptors':
        columns_pca = columns
    else:
        columns_pca = list( range( data_x_pca.shape[1] ) )

    data_pca = pd.DataFrame( data_x_pca )
    data_pca.columns = columns_pca

    return data_ori,data_ss,data_pca

def test_model(x,save_path,type1,type2,type3):

    save_path = (os.path.join(save_path , type3))
    savemodel = os.path.join(save_path , type1 + '_classifier_all.pkl')
    with open(savemodel , 'rb') as file :
        test_fit = pickle.load(file)
    y_pred = test_fit.predict(x)
    y_pred = np.reshape(y_pred , (-1 , 1))
    y_pred = pd.DataFrame(y_pred , columns=[type1 + '_' + type2 + '_'+ type3])

    return y_pred

def test(data,type1,type2,save_path):

    columns = data.columns.tolist()
    x = np.array(data[columns])

    lr_pred = test_model(x , save_path , type1 , type2 , 'LR')
    knn_pred = test_model(x , save_path , type1 , type2 , 'KNN')
    gb_pred = test_model(x , save_path , type1 , type2 , 'GB')
    rf_pred = test_model(x , save_path , type1 , type2 , 'RF')
    nn_pred = test_model(x , save_path , type1 , type2 , 'NN')

    results = pd.concat((lr_pred , knn_pred , gb_pred, rf_pred, nn_pred),axis=1)

    return results


if __name__ == '__main__':

    dataLocation = './TestDataset/'
    save_path_ss = './Dataset/Data_10F_SS_0425_all'
    save_path_pca = './Dataset/Data_10F_PCA_0425_all'

    resultSaveLocationORI = './results_10f_ORI_0425_all/'
    resultSaveLocationMUL = './results_10f_MUL_0425_all/'

    # AIEgens_dataset_file = "SimilarStructuresData.csv"
    # AIEgens_dataset_file = "Test2.csv"
    AIEgens_dataset_file = 'test0627.csv'
    AIEdataset1 = pd.read_csv(dataLocation + AIEgens_dataset_file, header=None, names=['molecular','smiles'])
    AIEdataset1.to_csv(dataLocation + '1-Formation' + AIEgens_dataset_file,index=False)

    dataset = clean_data(AIEdataset1,AIEgens_dataset_file)

    x_morg , x_rd , x_AP , x_torsion, x_descriptors = mol_data(dataset,AIEgens_dataset_file)

    data_ori_morg , data_ss_morg , data_pca_morg = data_cv(x_morg,'morg',save_path_ss, save_path_pca)
    data_ori_rd , data_ss_rd , data_pca_rd = data_cv(x_rd , 'rd' , save_path_ss , save_path_pca)
    data_ori_AP , data_ss_AP , data_pca_AP = data_cv(x_AP , 'AP' , save_path_ss , save_path_pca)
    data_ori_torsion , data_ss_torsion , data_pca_torsion = data_cv(x_torsion , 'torsion' , save_path_ss , save_path_pca)
    data_ori_descriptors , data_ss_descriptors , data_pca_descriptors = data_cv(x_descriptors , 'descriptors' , save_path_ss , save_path_pca)

    results_ori_morg = test(data_ori_morg, 'morg' ,'ORI', resultSaveLocationORI)
    results_ori_rd = test(data_ori_rd , 'rd' , 'ORI' , resultSaveLocationORI)
    results_ori_AP = test(data_ori_AP , 'AP' , 'ORI' , resultSaveLocationORI)
    results_ori_torsion = test(data_ori_torsion , 'torsion' , 'ORI' , resultSaveLocationORI)
    results_ori_descriptors = test( data_ori_descriptors , 'descriptors' , 'ORI' , resultSaveLocationORI )
    results_ori = pd.concat((results_ori_morg , results_ori_rd , results_ori_AP , results_ori_torsion, results_ori_descriptors) , axis=1)
    results_ori['Final_ORI'] = round(results_ori.sum(axis=1)/results_ori.shape[1]).astype(int)

    results_mul_morg = test(pd.merge(data_pca_morg , data_ss_descriptors , left_index=True , right_index=True , sort=False) ,
                            'morg' , 'MUL' , resultSaveLocationMUL)
    results_mul_rd = test(pd.merge(data_pca_rd , data_ss_descriptors , left_index=True , right_index=True , sort=False) ,
                          'rd' , 'MUL' , resultSaveLocationMUL)
    results_mul_AP = test(pd.merge(data_pca_AP , data_ss_descriptors , left_index=True , right_index=True , sort=False) ,
                          'AP' , 'MUL' , resultSaveLocationMUL)
    results_mul_torsion = test(pd.merge(data_pca_torsion, data_ss_descriptors , left_index=True , right_index=True , sort=False) ,
                               'torsion' , 'MUL' , resultSaveLocationMUL)
    results_mul = pd.concat((results_mul_morg , results_mul_rd , results_mul_AP , results_mul_torsion) , axis=1)
    results_mul['Final_MUL'] = round( results_mul.sum( axis=1 ) / results_mul.shape[1] ).astype( int )

    results_all = pd.concat((dataset['molecular'], results_ori, results_mul) , axis=1)

    columns = results_all.columns.tolist()
    results = pd.DataFrame(results_all[columns[1:]].values.T, index=results_all[columns[1:]].columns, columns=results_all[columns[0]])

    results.to_csv(os.path.join(dataLocation, 'resultsFinal' + AIEgens_dataset_file))


    pass


