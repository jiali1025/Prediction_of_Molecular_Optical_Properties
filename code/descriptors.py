import numpy as np
import pandas as pd
import os

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import warnings
warnings.filterwarnings('ignore')

dataLocation = '../data'

dataFile = "data.csv"
data = pd.read_csv(os.path.join(dataLocation, dataFile))

data["mol"] = [Chem.MolFromSmiles(x) for x in data["smiles"]]

descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
descriptors = pd.DataFrame([desc_calc.CalcDescriptors(mol) for mol in data['mol']])
descriptors.columns = descs

y = data['AIE character'].values

y = pd.DataFrame(np.reshape(y, (-1 , 1)))
y.columns = ['y']

descriptors = pd.merge(y, descriptors,left_index=True,right_index=True,sort=False)
descriptors.to_csv(os.path.join(dataLocation, 'descriptors.csv'),index=False)


pass


