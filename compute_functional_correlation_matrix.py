import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from region_connectivity import RegionConnectivity
import datetime

analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'
data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

response_filepaths = glob.glob(os.path.join(data_dir, 'region_responses') + '/' + '*.pkl')

cmats_z = []
for resp_fp in response_filepaths:
    region_responses = pd.read_pickle(resp_fp)
    correlation_matrix = np.corrcoef(np.vstack(region_responses.to_numpy()))
    # set diag to 0
    np.fill_diagonal(correlation_matrix, 0)
    # fischer z transform (arctanh) and append
    cmats_z.append(np.arctanh(correlation_matrix))

cmats_z = np.stack(cmats_z, axis=2)

# Make pd Dataframe
mean_cmat = np.mean(cmats_z, axis=2)
CorrelationMatrix_Functional = pd.DataFrame(data=mean_cmat, index=region_responses.index, columns=region_responses.index)

# Save
d = datetime.datetime.today()
datestring ='{:02d}'.format(d.year)+'{:02d}'.format(d.month)+'{:02d}'.format(d.day)
save_path = os.path.join(data_dir, 'functional_connectivity', 'CorrelationMatrix_Functional_{}.pkl'.format(datestring))
CorrelationMatrix_Functional.to_pickle(save_path)
