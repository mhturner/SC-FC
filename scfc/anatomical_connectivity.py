from neuprint import (Client, fetch_neurons, NeuronCriteria)
import numpy as np
import pandas as pd
import os
import networkx as nx

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

def getPrimaryInput(roiInfo):
    inputs = dict.fromkeys(roiInfo.keys(), 0)
    for key in inputs:
        inputs[key] = roiInfo[key].get('post', 0)

    primary_input = max(inputs, key=inputs.get)
    return primary_input

def computeConnectivityMatrix(neuprint_client, mapping):
    """

    """
    rois = list(mapping.keys())
    rois.sort()

    WeakConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    MediumConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    StrongConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    Connectivity = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    WeightedSynapseNumber = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)

    # CommonInputFraction: each row is fraction of total input cells that also project to region in [col]
    CommonInputFraction = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]

            weak_neurons = 0
            medium_neurons = 0
            strong_neurons = 0
            summed_connectivity = 0
            weighted_synapse_number = 0
            total_cells_to_a = 0
            shared_cells_to_ab = 0
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

                    # weighted synapses, for each cell going from sour -> targ:
                    #       := output tbars (presynapses) in targ * (input (post)synapses in sour)/(total (post)synapses onto that cell)
                    weighted_synapses = [Neur.roiInfo[x][targ]['pre'] * (Neur.roiInfo[x][sour]['post'] / Neur.loc[x, 'post']) for x in range(len(Neur))]

                    if Neur.roiInfo.shape[0] > 0:
                        summed_connectivity += np.sum(conn_strengths)
                        weighted_synapse_number += np.sum(weighted_synapses)
                        weak_neurons += n_weak
                        medium_neurons += n_medium
                        strong_neurons += n_strong

                    # Common input fraction
                    Neur_a, _ = fetch_neurons(NeuronCriteria(outputRois=sour, status='Traced')) # row in comon input matrix
                    Neur_b, _ = fetch_neurons(NeuronCriteria(outputRois=targ, status='Traced'))

                    Neur_a['primary_input'] = [getPrimaryInput(x) for x in Neur_a.roiInfo] # top input region for each cell
                    Neur_b['primary_input'] = [getPrimaryInput(x) for x in Neur_b.roiInfo]

                    a_from_elsewhere = Neur_a[~Neur_a['primary_input'].isin([sour, targ])] # only cells whose top input is somewhere other than source or target
                    b_from_elsewhere = Neur_b[~Neur_b['primary_input'].isin([sour, targ])]

                    total_cells_to_a += len(a_from_elsewhere.bodyId)
                    shared_cells_to_ab += len(np.intersect1d(a_from_elsewhere.bodyId, b_from_elsewhere.bodyId))


            WeakConnections.loc[[roi_source], [roi_target]] = weak_neurons
            MediumConnections.loc[[roi_source], [roi_target]] = medium_neurons
            StrongConnections.loc[[roi_source], [roi_target]] = strong_neurons

            Connectivity.loc[[roi_source], [roi_target]] = summed_connectivity
            WeightedSynapseNumber.loc[[roi_source], [roi_target]] = weighted_synapse_number

            CommonInputFraction.loc[[roi_source], [roi_target]] = shared_cells_to_ab / total_cells_to_a


    return WeakConnections, MediumConnections, StrongConnections, Connectivity, WeightedSynapseNumber, CommonInputFraction


class AnatomicalConnectivity():
    def __init__(self, data_dir, neuprint_client=None, mapping=None):
        """
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
        if computed_date is None:
            computed_date = '20200812'

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

    def getTwoStepConnectivity(self, OneStepConnectivity, symmetrize=False):
        """
        """
        A = OneStepConnectivity.to_numpy().copy()
        two_steps = np.zeros_like(A)
        for source in range(OneStepConnectivity.shape[0]):
            for target in range(OneStepConnectivity.shape[1]):
                if source != target:
                    conns = [np.sqrt(A[source, x] * A[x, target]) for x in range(OneStepConnectivity.shape[0]) if x not in (source, target)]
                    two_steps[source, target] = np.nansum(conns)

        if symmetrize:
            return pd.DataFrame(data=(two_steps + two_steps.T)/2, index=OneStepConnectivity.index, columns=OneStepConnectivity.index)
        else:
            return pd.DataFrame(data=two_steps, index=OneStepConnectivity.index, columns=OneStepConnectivity.index)

    def getShortestPathLength(self, type='CellCount'):
        anat_connect = self.getConnectivityMatrix(type, diag=None).to_numpy()
        G_anat = nx.from_numpy_matrix(anat_connect)

        for e in G_anat.edges:
            G_anat.edges[e]['distance'] = 1/G_anat.edges[e]['weight']

        sp_anat = np.zeros_like(anat_connect)
        for row in range(anat_connect.shape[0]):
            for col in range(anat_connect.shape[1]):
                path_len = nx.algorithms.shortest_path_length(G_anat, source=row, target=col, weight='distance')
                sp_anat[row, col] = path_len

        ShortestPath = pd.DataFrame(data=sp_anat, index=self.rois, columns=self.rois)

        return ShortestPath

    def getAdjacency(self, type, do_log=False, thresh=None):
        ConnectivityMatrix = self.getConnectivityMatrix(type=type, symmetrize=True)
        if thresh is None:
            thresh_value = 0
        else: #thresh as a quantile of all anatomical connections
            thresh_value = np.quantile(ConnectivityMatrix.to_numpy()[self.upper_inds], thresh)

        if do_log:
            keep_inds = np.where(ConnectivityMatrix.to_numpy()[self.upper_inds] > thresh_value) # for log-transforming anatomical connectivity, toss zero values
            adjacency = np.log10(ConnectivityMatrix.to_numpy().copy()[self.upper_inds][keep_inds])
        else:
            keep_inds = None
            adjacency = ConnectivityMatrix.to_numpy().copy()[self.upper_inds]

        return adjacency, keep_inds
