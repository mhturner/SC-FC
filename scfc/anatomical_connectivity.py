"""
Turner, Mann, Clandinin: structural connectivity utils and functions.

https://github.com/mhturner/SC-FC

References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python
"""

import numpy as np
import pandas as pd
import os

from . import bridge


def getAtlasConnectivity(include_inds, name_list, atlas_id, metric='cellcount'):
    """
    .

    metric: 'cellcount', 'tbar', 'weighted_tbar'
    """
    data_dir = bridge.getUserConfiguration()['data_dir']
    if atlas_id == 'branson':
        cellcount_full = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_branson_{}_matrix.csv'.format(metric)), header=0).to_numpy()[:, 1:]
        cellcount_full = pd.DataFrame(data=cellcount_full, index=np.arange(1, 1000), columns=np.arange(1, 1000))
    elif atlas_id == 'ito':
        cellcount_full = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_ito_{}_matrix.csv'.format(metric)), header=0).to_numpy()[:, 1:]
        cellcount_full = pd.DataFrame(data=cellcount_full, index=np.arange(1, 87), columns=np.arange(1, 87))

    # filter and sort cellcount_full by include_inds
    cellcount_filtered = pd.DataFrame(data=np.zeros((len(include_inds), len(include_inds))), index=name_list, columns=name_list)

    for s_ind, src in enumerate(include_inds):
        for t_ind, trg in enumerate(include_inds):
            cellcount_filtered.iloc[s_ind, t_ind] = cellcount_full.loc[src, trg]

    return cellcount_filtered


def getRoiCompleteness(neuprint_client, name_list):
    """
    Return roi completness measures for Ito atlas regions.

    neuprint_client
    name_list: list of Ito regions to return completeness for
    """
    # How many synapses belong to traced neurons
    completeness_neuprint = neuprint_client.fetch_roi_completeness()
    completeness_neuprint.index = completeness_neuprint['roi']

    completeness = np.zeros((len(name_list), 2))
    for r_ind, roi in enumerate(name_list):
        current_rois = completeness_neuprint.loc[bridge.ito_to_neuprint(roi), :]
        completeness[r_ind, 0] = current_rois['roipre'].sum() / current_rois['totalpre'].sum()
        completeness[r_ind, 1] = current_rois['roipost'].sum() / current_rois['totalpost'].sum()

    roi_completeness = pd.DataFrame(data=completeness, index=name_list, columns=['frac_pre', 'frac_post'])
    roi_completeness['completeness'] = roi_completeness['frac_pre'] * roi_completeness['frac_post']

    return roi_completeness


class AnatomicalConnectivity():
    """Anatomical Connectivity class."""

    def __init__(self, data_dir, neuprint_client=None, mapping=None):
        """
        Initialize AnatomicalConnectivity object.

        :data_dir:
        :neuprint_client:
        :mapping:
        """
        self.data_dir = data_dir
        self.mapping = mapping

        self.rois = list(self.mapping.keys())
        self.rois.sort()
        self.upper_inds = np.triu_indices(len(self.rois), k=1) # k=1 excludes main diagonal
        self.lower_inds = np.tril_indices(len(self.rois), k=-1) # k=-1 excludes main diagonal

    def getConnectivityMatrix(self, type, symmetrize=False, diag=None, computed_date=None):
        """
        Retrieve computed connectivity matrix.

        Computed using compute_connectivity_matrix.py

        :type: str, one of ['CellCount', 'ConnectivityWeight', 'WeightedSynapseCount', 'TBars', 'CommonInputFraction']
        :symmetrize: bool, symmetrize connectivity matrix?
        :diag: value to fill diagonal with, if None, then fills with measured value
        :computed_date: str, specifies precomputed connectivity file to pull

        Return square dataframe connectivity matrix
        """
        if computed_date is None:
            computed_date = '20210114'

        if type == 'CellCount':
            """
            """
            WeakConnections = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'WeakConnections_computed_{}.pkl'.format(computed_date)))
            MediumConnections = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'MediumConnections_computed_{}.pkl'.format(computed_date)))
            StrongConnections = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'StrongConnections_computed_{}.pkl'.format(computed_date)))
            conn_mat = WeakConnections + MediumConnections + StrongConnections

        elif type == 'ConnectivityWeight':
            """
            """
            conn_mat = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'Connectivity_computed_{}.pkl'.format(computed_date)))

        elif type == 'WeightedSynapseCount':
            """
            """
            conn_mat = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'WeightedSynapseNumber_computed_{}.pkl'.format(computed_date)))

        elif type == 'TBars':
            """
            """
            conn_mat = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'TBars_computed_{}.pkl'.format(computed_date)))

        tmp_mat = conn_mat.to_numpy().copy()
        # set diagonal value
        if diag is not None:
            np.fill_diagonal(tmp_mat, diag)

        if symmetrize:
            return pd.DataFrame(data=(tmp_mat + tmp_mat.T)/2, index=conn_mat.index, columns=conn_mat.index)
        else:
            return pd.DataFrame(data=tmp_mat, index=conn_mat.index, columns=conn_mat.index)

    def getAdjacency(self, type, do_log=False, thresh=None):
        """
        Retrieve adjacency data.

        :type: connectivity metric, as in getConnectivityMatrix()
        :do_log: bool, do log transform on connectivity data
        :thresh: as quantile of connection strength, minimum value before log transforming, default (None) is 0

        Returns:
            adjacency: array
            keep_inds: subset of upper_inds (i.e. those that are above threshold)
        """
        ConnectivityMatrix = self.getConnectivityMatrix(type=type, symmetrize=True)
        if thresh is None:
            thresh_value = 0
        else: # thresh as a quantile of all anatomical connections
            thresh_value = np.quantile(ConnectivityMatrix.to_numpy()[self.upper_inds], thresh)

        if do_log:
            keep_inds = np.where(ConnectivityMatrix.to_numpy()[self.upper_inds] > thresh_value) # for log-transforming anatomical connectivity, toss zero values
            adjacency = np.log10(ConnectivityMatrix.to_numpy().copy()[self.upper_inds][keep_inds])
        else:
            keep_inds = None
            adjacency = ConnectivityMatrix.to_numpy().copy()[self.upper_inds]

        return adjacency, keep_inds
