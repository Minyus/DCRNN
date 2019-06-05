#%%
from pathlib import Path
import numpy as np

#%%

pred_raw_file = 'data/dcrnn_predictions.npz'
pred = np.load(pred_raw_file, allow_pickle=True)

#%%
pred_tensor = pred['predictions']
# truth_tensor = pred['groundtruth']

#%%
pred_arr2d = pred_tensor[:,-1,:]

pred_arr2d
np.savetxt('data/dcrnn_pred_arr2d.csv', pred_arr2d, delimiter=',')


