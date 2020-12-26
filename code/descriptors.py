import numpy as np
import pandas as pd
import os

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import warnings
warnings.filterwarnings('ignore')

dataLocation = '../data'

dataFile = "data.csv"
data = pd.read_csv(os.path.join(dataLocation, dataFile))

data["mol"] = [Chem.MolFromSmiles(x) for x in data["smiles"]]

y = data['AIE character'].values
y = pd.DataFrame(np.reshape(y, (-1 , 1)))
y.columns = ['y']

descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
descriptors = pd.DataFrame([desc_calc.CalcDescriptors(mol) for mol in data['mol']])
descriptors.columns = descs
descriptors = pd.merge(y, descriptors,left_index=True,right_index=True,sort=False)
descriptors.to_csv(os.path.join(dataLocation, 'descriptors.csv'),index=False)

columns = ['y'] + list(range(2048))

morg_fp = [Chem.GetMorganFingerprintAsBitVect(m, 2, nBits = 2048) for m in data['mol']]
morg_fp_np = []
for fp in morg_fp:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  morg_fp_np.append(arr)
x_morg = morg_fp_np
x_morg = np.array(x_morg)
x_morg = pd.DataFrame(x_morg)
x_morg_sample = pd.merge(y, x_morg,left_index=True,right_index=True,sort=False)
x_morg_sample.columns = columns
x_morg_sample.to_csv(os.path.join(dataLocation, 'x_morg.csv'),index=False)

rd_fp = [Chem.RDKFingerprint(m) for m in data["mol"]]
rd_fp_np = []
for fp in rd_fp:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  rd_fp_np.append(arr)
x_rd = rd_fp_np
x_rd = np.array(x_rd)
x_rd = pd.DataFrame(x_rd)
x_rd_sample = pd.merge(y, x_rd,left_index=True,right_index=True,sort=False)
x_rd_sample.columns = columns
x_rd_sample.to_csv(os.path.join(dataLocation, 'x_rd.csv'),index=False)

AP_fp = [Chem.GetHashedAtomPairFingerprintAsBitVect(m) for m in data["mol"]]
AP_fp_np = []
for fp in AP_fp:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  AP_fp_np.append(arr)
x_AP = AP_fp_np
x_AP = np.array(x_AP)
x_AP = pd.DataFrame(x_AP)
x_AP_sample = pd.merge(y, x_AP,left_index=True,right_index=True,sort=False)
x_AP_sample.columns = columns
x_AP_sample.to_csv(os.path.join(dataLocation, 'x_AP.csv'),index=False)

torsion_fp = [Chem.GetHashedTopologicalTorsionFingerprintAsBitVect(m) for m in data["mol"]]
torsion_fp_np = []
for fp in torsion_fp:
  arr = np.zeros((1,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  torsion_fp_np.append(arr)
x_torsion = torsion_fp_np
x_torsion = np.array(x_torsion)
x_torsion = pd.DataFrame(x_torsion)
x_torsion_sample = pd.merge(y, x_torsion,left_index=True,right_index=True,sort=False)
x_torsion_sample.columns = columns
x_torsion_sample.to_csv(os.path.join(dataLocation, 'x_torsion.csv'),index=False)

pass


