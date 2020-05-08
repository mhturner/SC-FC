from neuprint import (Client, fetch_neurons, NeuronCriteria)
import numpy as np
import pandas as pd
import os

def getRoiMapping():
    """
    Include ROIs that are at least 50% in the EM dataset
    Toss stuff in the optic lobes since those aren't in the functional dataset
    Mapping smooshes some anatomical ROI groups into single functional ROIs. E.g. MB lobes
    """
    # keys of mapping are roi names to use in analysis, based on functional data atlas
    #   values are lists of corresponding roi names in neuprint data
    mapping = {'AL(R)':['AL(R)'], # 83% in EM volume
                'AOTU(R)':['AOTU(R)'],
                'ATL(R)': ['ATL(R)'],
                'ATL(L)': ['ATL(L)'],
                'AVLP(R)': ['AVLP(R)'],
                'BU(R)': ['BU(R)'],
                'BU(L)': ['BU(L)'], # 52% in EM volume
                'CAN(R)': ['CAN(R)'], # 68% in volume
                'CRE(R)': ['CRE(R)'],
                'CRE(L)': ['CRE(L)'], #% 90% in vol
                'EB': ['EB'],
                'EPA(R)': ['EPA(R)'],
                'FB': ['AB(R)', 'AB(L)', 'FB'],
                'GOR(R)': ['GOR(R)'],
                'GOR(L)': ['GOR(L)'], # ~60% in volume
                # 'IB(R)': ['IB'], # This is lateralized in functional data but not EM. Can we smoosh IB_R and IB_L in fxnal?
                'ICL(R)': ['ICL(R)'],
                'LAL(R)': ['LAL(R)'],
                'LH(R)': ['LH(R)'],
                'MBCA(R)': ['CA(R)'],
                'MBML(R)': ["b'L(R)", 'bL(R)', 'gL(R)'],
                'MBML(L)': ["b'L(L)", 'bL(L)', 'gL(L)'], # ~50-80% in volume
                'MBPED(R)': ['PED(R)'],
                'MBVL(R)': ["a'L(R)", 'aL(R)'],
                'NO': ['NO'],
                'PB': ['PB'],
                'PLP(R)': ['PLP(R)'],
                'PVLP(R)': ['PVLP(R)'],
                'SCL(R)': ['SCL(R)'],
                'SIP(R)': ['SIP(R)'],
                'SLP(R)': ['SLP(R)'],
                'SMP(R)': ['SMP(R)'],
                'SMP(L)': ['SMP(L)'],
                'SPS(R)': ['SPS(R)'],
                'VES(R)': ['VES(R)'], # 84% in vol
                'WED(R)': ['WED(R)']}

    return mapping

def prepFullCorrrelationMatrix():
    analysis_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/hemibrain_analysis/roi_connectivity'

    fn_cmat = 'full_cmat.txt'
    fn_names = 'Original_Index_panda_full.csv'

    # cmat = pd.read_csv(os.path.join(analysis_dir, 'data', fn_cmat), sep=' ', header=0)
    cmat = np.loadtxt(os.path.join(analysis_dir, 'data', fn_cmat), delimiter=' ')

    # roi_names = pd.read_csv(os.path.join(analysis_dir, 'data', fn_names), sep=',', header=0)
    roi_names = pd.read_csv(os.path.join(analysis_dir, 'data', fn_names), sep=',', header=0)
    pull_inds = np.where([type(x) is not str for x in roi_names.name.to_numpy()])[0]

    new_roi_names = roi_names.name.to_numpy()
    new_roi_names = [x.replace('_R','(R)') for x in new_roi_names if type(x) is str]
    new_roi_names = [x.replace('_L','(L)') for x in new_roi_names if type(x) is str]
    new_roi_names = [x.replace('_', '') for x in new_roi_names if type(x) is str]

    cmat = np.delete(cmat, pull_inds, axis=0)
    cmat = np.delete(cmat, pull_inds, axis=1)

    CorrelationMatrix = pd.DataFrame(data=cmat, index=new_roi_names, columns=new_roi_names)

    return CorrelationMatrix


def filterCorrelationMatrix(CorrelationMatrix_Full, mapping):
    rois = list(mapping.keys())
    rois.sort()
    mat = np.zeros(shape=(len(rois), len(rois)))
    CorrelationMatrix_Filtered = pd.DataFrame(data=mat, index=rois, columns=rois)
    for r_ind, r in enumerate(rois):
        for c_ind, c in enumerate(rois):
            CorrelationMatrix_Filtered.loc[[r], [c]] = CorrelationMatrix_Full.loc[r, c]

    return CorrelationMatrix_Filtered

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
