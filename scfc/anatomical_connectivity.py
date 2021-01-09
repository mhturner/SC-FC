"""
Turner, Mann, Clandinin: structural connectivity utils and functions.

https://github.com/mhturner/SC-FC

References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python
"""

from neuprint import (fetch_neurons, NeuronCriteria)
import numpy as np
import pandas as pd
import os


def getRoiCompleteness(neuprint_client, mapping):
    """
    Return roi completness measures.

    neuprint_client
    mapping: mapping dict to bridge hemibrain regions to atlas regions
    returns roi_completeness (dataframe)
    """
    rois = list(mapping.keys())
    rois.sort()

    # How many synapses belong to traced neurons
    completeness_df = neuprint_client.fetch_roi_completeness()
    completeness_df['frac_pre'] = completeness_df['roipre'] / completeness_df['totalpre']
    completeness_df['frac_post'] = completeness_df['roipost'] / completeness_df['totalpost']

    roi_completeness = pd.DataFrame(data=np.zeros((len(rois), 2)), index=rois, columns=['frac_pre', 'frac_post'])
    for r in rois:
        grouped_rois = mapping[r]
        comp_pre = []
        comp_post = []
        for gr in grouped_rois:
            ind = np.where(completeness_df['roi'] == gr)[0]
            if len(ind) > 0:
                new_pre = completeness_df['frac_pre'][ind].to_numpy()
                new_post = completeness_df['frac_post'][ind].to_numpy()
                comp_pre.append(new_pre)
                comp_post.append(new_post)

        if len(comp_pre) > 0:
            roi_completeness.loc[[r], ['frac_pre']] = np.array(comp_pre).mean()
            roi_completeness.loc[[r], ['frac_post']] = np.array(comp_post).mean()

    roi_completeness['completeness'] = roi_completeness['frac_pre'] * roi_completeness['frac_post']

    return roi_completeness


def getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan'):
    """
    Return neuprint's precomputed connectivity matrix.

    neuprint_client
    mapping: mapping dict to bridge hemibrain regions to atlas regions
    metric: connectivity metric, defined by neuprint data. Can be 'count' or 'weight'
    diagonal: value to fill along diag

    returns ConnectivityMatrix (square dataframe)
    """
    conn_df = neuprint_client.fetch_roi_connectivity()
    rois = list(mapping.keys())
    rois.sort()

    ConnectivityMatrix = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]
            total = 0
            for sour in sources:
                for targ in targets:
                    new = conn_df[np.logical_and(conn_df.from_roi==sour, conn_df.to_roi==targ)][metric].to_numpy()
                    if len(new) > 0:
                        total += new

            if roi_source == roi_target: # on the diagonal
                if diagonal == 'zero':
                    ConnectivityMatrix.loc[[roi_source], [roi_target]] = 0
                elif diagonal == 'nan':
                    ConnectivityMatrix.loc[[roi_source], [roi_target]] = np.nan
                elif diagonal == 'compute':
                    ConnectivityMatrix.loc[[roi_source], [roi_target]] = total
            else: # off the diagonal
                ConnectivityMatrix.loc[[roi_source], [roi_target]] = total

    return ConnectivityMatrix


def getPrimaryInput(roiInfo):
    """
    Find region that provides the most inputs to region.

    roiInfo: neuprint roi info dataframe, for a single roi
    """
    inputs = dict.fromkeys(roiInfo.keys(), 0)
    for key in inputs:
        inputs[key] = roiInfo[key].get('post', 0)

    primary_input = max(inputs, key=inputs.get)
    return primary_input


def computeSynapseConnectivityDistribution(neuprint_client, mapping):
    """
    Get distribution of synapse connectivity for each connection from region A to region B.

    neuprint_client
    mapping: mapping dict to bridge hemibrain regions to atlas regions
    """
    rois = list(mapping.keys())
    rois.sort()

    SynapseCount = pd.DataFrame(columns=['source', 'target', 'cell', 'num_tbars', 'weighted_tbars'])

    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]

            for s_ind, sour in enumerate(sources): # this multiple sources/targets is necessary for collapsing rois based on mapping
                for targ in targets:
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced'))

                    # weighted synapses, for each cell going from sour -> targ:
                    #       := output tbars (presynapses) in targ * (input (post)synapses in sour)/(total (post)synapses onto that cell)
                    weighted_synapses = [Neur.roiInfo[x][targ]['pre'] * (Neur.roiInfo[x][sour]['post'] / Neur.loc[x, 'post']) for x in range(len(Neur))]

                    tbars = [Neur.roiInfo[x][targ]['pre'] for x in range(len(Neur))]
                    for cell in range(len(tbars)):
                        SynapseCount.loc[len(SynapseCount)] = [sour, targ, cell, tbars[cell], weighted_synapses[cell]]

    return SynapseCount


def computeConnectivityMatrix(neuprint_client, mapping):
    """
    Compute region connectivity matrix, for various metrics.

    neuprint_client
    mapping: mapping dict to bridge hemibrain regions to atlas regions
    """
    rois = list(mapping.keys())
    rois.sort()

    WeakConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    MediumConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    StrongConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    Connectivity = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    WeightedSynapseNumber = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    TBars = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)

    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]

            weak_neurons = 0
            medium_neurons = 0
            strong_neurons = 0
            summed_connectivity = 0
            weighted_synapse_number = 0
            tbars = 0
            for s_ind, sour in enumerate(sources): # this multiple sources/targets is necessary for collapsing rois based on mapping
                for targ in targets:
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced', cropped=False))

                    outputs_in_targ = np.array([x[targ]['pre'] for x in Neur.roiInfo]) # neurons with Tbar output in target
                    inputs_in_sour = np.array([x[sour]['post'] for x in Neur.roiInfo]) # neuron with PSD input in source

                    n_weak = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour<3))
                    n_medium = np.sum(np.logical_and(outputs_in_targ>0, np.logical_and(inputs_in_sour>=3, inputs_in_sour<10)))
                    n_strong = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour>=10))

                    # Connection strength for each cell := sqrt(input PSDs in source x output tbars in target)
                    conn_strengths = [np.sqrt(x[targ]['pre'] * x[sour]['post']) for x in Neur.roiInfo]

                    # weighted synapses, for each cell going from sour -> targ:
                    #       := output tbars (presynapses) in targ * (input (post)synapses in sour)/(total (post)synapses onto that cell)
                    weighted_synapses = [Neur.roiInfo[x][targ]['pre'] * (Neur.roiInfo[x][sour]['post'] / Neur.loc[x, 'post']) for x in range(len(Neur))]

                    new_tbars = [Neur.roiInfo[x][targ]['pre'] for x in range(len(Neur))]

                    if Neur.roiInfo.shape[0] > 0:
                        summed_connectivity += np.sum(conn_strengths)
                        weighted_synapse_number += np.sum(weighted_synapses)
                        weak_neurons += n_weak
                        medium_neurons += n_medium
                        strong_neurons += n_strong
                        tbars += np.sum(new_tbars)

            WeakConnections.loc[[roi_source], [roi_target]] = weak_neurons
            MediumConnections.loc[[roi_source], [roi_target]] = medium_neurons
            StrongConnections.loc[[roi_source], [roi_target]] = strong_neurons

            Connectivity.loc[[roi_source], [roi_target]] = summed_connectivity
            WeightedSynapseNumber.loc[[roi_source], [roi_target]] = weighted_synapse_number
            TBars.loc[[roi_source], [roi_target]] = tbars

    return WeakConnections, MediumConnections, StrongConnections, Connectivity, WeightedSynapseNumber, TBars


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

        if neuprint_client is not None:
            # get reconstruction completeness matrix
            roi_completeness = getRoiCompleteness(neuprint_client, self.mapping)
            self.CompletenessMatrix = pd.DataFrame(data=np.outer(roi_completeness['frac_post'], roi_completeness['frac_pre']), index=roi_completeness.index, columns=roi_completeness.index)

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
            computed_date = '20200909'

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

        elif type == 'CommonInputFraction':
            """
            """
            conn_mat = pd.read_pickle(os.path.join(self.data_dir, 'connectome_connectivity', 'CommonInputFraction_computed_{}.pkl'.format(computed_date)))

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
