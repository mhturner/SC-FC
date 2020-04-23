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

def computeConnectivityMatrix(neuprint_client, mapping, neuron_count_threshold=[1]):
    """
    This takes like 20 minutes
    :neuron_count_threshold: list of thresholds - minumum number of pre & post synapses on a cell to count it as a connection
        NeuronCount will be have the same length as threshold list, one df for each count threshold
    """
    rois = list(mapping.keys())
    rois.sort()

    completeness_df = neuprint_client.fetch_roi_completeness()

    ConnectivityMatrix = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    SynapseCount = pd.DataFrame(data=np.zeros((len(rois), 2)), index=rois, columns=['total_synapses','assigned_synapses'])
    NeuronCount = []
    for thresh in neuron_count_threshold:
        NeuronCount.append(pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois))

    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]
            total_conn_score = 0
            total_neurons = np.zeros(len(neuron_count_threshold))
            total_synapses = 0
            assigned_synapses = 0
            for s_ind, sour in enumerate(sources): # this multiple sources/targets is necessary for collapsing rois based on mapping
                for targ in targets:
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, regex=True, status='Traced'))

                    # get the number of neurons from source to target with over :neuron_count_threshold: input & output synapses in rois"
                    n_neurons = np.zeros(len(neuron_count_threshold))
                    for t_ind, thresh in enumerate(neuron_count_threshold):
                        outputs_in_targ = np.array([x[targ]['pre'] for x in Neur.roiInfo])
                        inputs_in_sour = np.array([x[sour]['post'] for x in Neur.roiInfo])
                        n_neurons[t_ind] = np.sum(np.logical_and(outputs_in_targ>thresh, inputs_in_sour>thresh))

                    scaled_output = []
                    for n in range(Neur.shape[0]):
                        # For each cell that connects source -> target. Compute the scaled_output:
                        # scaled_output = (Number of output synapses in target) x (number of input synapses in source / total input synapses)
                        scaled_output.append(Neur.roiInfo[n][targ]['pre'] * (Neur.roiInfo[n][sour]['post']/Neur.post[n]))

                    if len(scaled_output) > 0:
                        new_score = np.sum(scaled_output)
                        total_conn_score += new_score
                        total_neurons += n_neurons # list of counts for each threshold value

                    if s_ind==0:
                        # get the presynapses in target
                        new_total_synapses = completeness_df[completeness_df.roi==targ]['totalpre'].to_numpy()
                        new_assigned_synapses = completeness_df[completeness_df.roi==targ]['roipre'].to_numpy()
                        total_synapses += new_total_synapses
                        assigned_synapses += new_assigned_synapses

            ConnectivityMatrix.loc[[roi_source], [roi_target]] = total_conn_score

            for t_ind, thresh in enumerate(neuron_count_threshold):
                NeuronCount[t_ind].loc[[roi_source], [roi_target]] = total_neurons[t_ind]

            SynapseCount.loc[[roi_target], ['total_synapses']] = total_synapses
            SynapseCount.loc[[roi_target], ['assigned_synapses']] = assigned_synapses


    return ConnectivityMatrix, NeuronCount, SynapseCount
