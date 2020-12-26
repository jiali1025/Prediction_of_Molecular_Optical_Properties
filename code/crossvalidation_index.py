# general and data handling
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold

dataLocation = '../data'

save_path = (os.path.join(dataLocation,'cv_inds'))
if not os.path.exists(save_path):
    os.makedirs(save_path)

data = pd.read_csv(os.path.join(dataLocation, 'data.csv'))
data = np.array(data)

nfold = 10
kf = KFold(n_splits=nfold, shuffle = True, random_state=1)

i = 0
for train, test in kf.split(data) :
    train_ind = pd.DataFrame(train)
    train_ind.columns = ['train_ind']
    train_ind.to_csv(os.path.join(save_path , 'train_ind_' + str(i) + '.csv') , index=False)

    test_ind = pd.DataFrame(test)
    test_ind.columns = ['test_ind']
    test_ind.to_csv(os.path.join(save_path , 'test_ind_' + str(i) + '.csv') , index=False)
    i += 1






