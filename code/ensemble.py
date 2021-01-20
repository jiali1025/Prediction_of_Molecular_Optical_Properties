import pandas as pd
import os
import pickle

def ensem_cv ( result_path , methods , types , nfold ) :
    result = pd.DataFrame ( [ ] )
    result_ = { }
    for i in range ( nfold ) :
        num = 0
        for j in range ( len ( methods ) ) :
            for k in range ( len ( types ) ) :
                num += 1
                if j == 0 and k == 0 :
                    y0 = pd.read_csv (
                        os.path.join ( result_path , methods [ j ] , types [ k ] + '_' + str ( i ) + '_resultsY.csv' ) )
                else :
                    y = pd.read_csv (
                        os.path.join ( result_path , methods [ j ] , types [ k ] + '_' + str ( i ) + '_resultsY.csv' ) )
                    y0 [ 'Ypred' ] = y0 [ 'Ypred' ] + y [ 'Ypred' ]
        result [ 'Ypred' + str ( i ) ] = round ( y0 [ 'Ypred' ] / num ).astype ( int )
        result [ 'Ytest' + str ( i ) ] = y0 [ 'Ytest' ]
        result_ [ 'Ypred' + str ( i ) ] = round ( y0 [ 'Ypred' ] / num ).astype ( int )
        result_ [ 'Ytest' + str ( i ) ] = y0 [ 'Ytest' ]

    return result , result_

if __name__ == '__main__' :

    dataLocation = '../data'

    resultSaveLocations = [ '../results/ori_result/' , '../results/mul_result/' ]

    nfold = 10
    methods = [ 'LR' , 'KNN' , 'GB' , 'RF' , 'NN' ]
    types = [ 'morg' , 'rd' , 'AP' , 'torsion' ]

    for resultSaveLocation in resultSaveLocations :

        resultSavePath = os.path.join ( resultSaveLocation , 'Ensem' )

        if not os.path.exists ( resultSavePath ) :
            os.makedirs ( resultSavePath )

        res , res_ = ensem_cv ( resultSaveLocation , methods , types , nfold )

        res.to_csv ( os.path.join ( resultSavePath , 'ensem.csv' ) , index = False )

        savefile = os.path.join ( resultSavePath , 'ensem.pkl' )
        with open ( savefile , 'wb' ) as pickle_file :
            pickle.dump ( res_ , pickle_file )

        pass
