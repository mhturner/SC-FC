from neuprint import (Client, fetch_neurons, NeuronCriteria)
import numpy as np
import pandas as pd

def getRoiMapping(neuprint_client):
    """
    Exclude L-lateralized ROIs, since those are mostly not in the volume or sometimes only partially so
    Exclude optic lobes
    Combine MB lobes into medial and ventral, rename other MB rois
    """
    conn_df = neuprint_client.fetch_roi_connectivity()

    neuprint_rois = np.unique(conn_df['from_roi'])
    neuprint_rois = list(neuprint_rois[np.where(['(L)' not in x for x in neuprint_rois])])

    # could include: BU(L), ATL(L)

    # remove: 'PRW', 'GNG', 'FLA(R), SAD'

    rois_to_remove = ('LO(R)', 'LOP(R)', 'ME(R)', 'AME(R)',
                      "b'L(R)", 'bL(R)', 'gL(R)', "a'L(R)",
                      'aL(R)', 'PED(R)', 'CA(R)', 'AB(R)')
    equivalent_rois = [x for x in neuprint_rois if x not in rois_to_remove]
    # keys of mapping are roi name in connectivity matrix, values are lists of corresponding roi names in neuprint data
    mapping = {}
    for er in equivalent_rois:
        mapping[er] = [er]

    mapping['MBML(R)'] = ["b'L(R)", 'bL(R)', 'gL(R)']
    mapping['MBVL(R)'] = ["a'L(R)", 'aL(R)']
    mapping['MBPED(R)'] = ['PED(R)']
    mapping['MBCA(R)'] = ['CA(R)']
    mapping['FB'] = ['AB(R)', 'FB']

    return mapping


def getRoiCompleteness(neuprint_client, mapping):
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
            roi_completeness.loc[[r],['frac_pre']] = np.array(comp_pre).mean()
            roi_completeness.loc[[r],['frac_post']] = np.array(comp_post).mean()

    roi_completeness['completeness'] = roi_completeness['frac_pre'] * roi_completeness['frac_post']

    return roi_completeness


def getPrecomputedConnectivityMatrix(neuprint_client, mapping, metric='count', diagonal='nan'):
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

def computeConnectivityMatrix(neuprint_client, mapping):
    """
    This takes like 20 minutes
    """
    rois = list(mapping.keys())
    rois.sort()

    WeakConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    MediumConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    StrongConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    Connectivity = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]

            weak_neurons = 0
            medium_neurons = 0
            strong_neurons = 0
            summed_connectivity = 0
            for s_ind, sour in enumerate(sources): # this multiple sources/targets is necessary for collapsing rois based on mapping
                for targ in targets:
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced'))

                    outputs_in_targ = np.array([x[targ]['pre'] for x in Neur.roiInfo]) # neurons with Tbar output in target
                    inputs_in_sour = np.array([x[sour]['post'] for x in Neur.roiInfo]) # neuron with PSD input in source

                    n_weak = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour<3))
                    n_medium = np.sum(np.logical_and(outputs_in_targ>0, np.logical_and(inputs_in_sour>=3, inputs_in_sour<10)))
                    n_strong = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour>=10))

                    # Connection strength for each cell := sqrt(input PSDs in source x output tbars in target)
                    conn_strengths = [np.sqrt(x[targ]['pre'] * x[sour]['post']) for x in Neur.roiInfo]

                    if Neur.roiInfo.shape[0] > 0:
                        summed_connectivity += np.sum(conn_strengths)
                        weak_neurons += n_weak
                        medium_neurons += n_medium
                        strong_neurons += n_strong

            WeakConnections.loc[[roi_source], [roi_target]] = weak_neurons
            MediumConnections.loc[[roi_source], [roi_target]] = medium_neurons
            StrongConnections.loc[[roi_source], [roi_target]] = strong_neurons

            Connectivity.loc[[roi_source], [roi_target]] = summed_connectivity


    return WeakConnections, MediumConnections, StrongConnections, Connectivity
