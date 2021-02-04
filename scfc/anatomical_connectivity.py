"""
Turner, Mann, Clandinin: structural connectivity utils and functions.

https://github.com/mhturner/SC-FC
mhturner@stanford.edu

References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python
"""

import numpy as np
import pandas as pd
import os
from neuprint import (fetch_neurons, NeuronCriteria)

from . import bridge


def getAtlasConnectivity(include_inds, name_list, atlas_id, metric='cellcount'):
    """
    Load .csv of region-to-region structural connectivity, computed from hemibrain_2_atlas.r.

    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    :atlas_id:
    :metric: 'cellcount', 'tbar', 'weighted_tbar'
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


def computeConnectivityMatrix(neuprint_client, mapping):
    """
    Compute region connectivity matrix from neuprint tags, for various metrics.

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

    body_ids = [] # keep list of all cells connecting among regions

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
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced', cropped=False)) # only take uncropped neurons

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

                    # body_ids
                    body_ids.append(Neur.bodyId.values)

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

    body_ids = np.unique(np.hstack(body_ids)) # don't double count cells that contribute to multiple connections

    return WeakConnections, MediumConnections, StrongConnections, Connectivity, WeightedSynapseNumber, TBars, body_ids
