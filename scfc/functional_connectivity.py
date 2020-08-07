import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy import signal

def getRoiMapping():
    """
    Include ROIs that are at least 50% in the EM dataset
    Toss stuff in the optic lobes since those aren't in the functional dataset
    Mapping smooshes some anatomical ROI groups into single functional ROIs. E.g. MB lobes
    """
    # keys of mapping are roi names to use in analysis, based on functional data atlas
    #   values are lists of corresponding roi names in neuprint data
    mapping =  {'AL(R)':['AL(R)'], # 83% in EM volume
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
                'IB': ['IB'], # This is lateralized in functional data but not EM. Smoosh IB_R and IB_L together in fxnal to match, see loadAtlasData
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
np
def loadAtlasData(atlas_path, roinames_path, mapping=None):
    roi_names = pd.read_csv(roinames_path, sep=',', header=0).name.to_numpy()
    mask_brain = np.asarray(np.squeeze(nib.load(atlas_path).get_fdata()), 'uint8')

    # cut out nan regions (tracts))
    pull_inds = np.where([type(x) is str for x in roi_names])[0]
    delete_inds = np.where([type(x) is not str for x in roi_names])[0]

    # filter region namesfraction
    roi_names = np.array([x for x in roi_names if type(x) is str]) # cut out nan regions from roi names

    # convert names to match display format
    roi_names = [x.replace('_R','(R)') for x in roi_names]
    roi_names = [x.replace('_L','(L)') for x in roi_names]
    roi_names = [x.replace('_', '') for x in roi_names]

    # filter mask brain regions
    roi_mask = []
    roi_size = []
    for r_ind, r in enumerate(roi_names):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == (pull_inds[r_ind] + 1) # mask values start at 1, not 0
        roi_mask.append(new_roi_mask) #bool
        roi_size.append(np.sum(new_roi_mask>0))

    # combine IB(R) and IB(L) in fxnal atlas
    ibr_ind = np.where(np.array(roi_names)=='IB(R)')[0][0]
    ibl_ind = np.where(np.array(roi_names)=='IB(L)')[0][0]
    # merge into IB(R) slot
    roi_mask[ibr_ind] = np.logical_or(roi_mask[ibr_ind], roi_mask[ibl_ind])
    roi_size[ibr_ind] = roi_size[ibr_ind] + roi_size[ibl_ind]
    roi_names[ibr_ind] = 'IB'
    # remove IB(L) slot
    roi_mask.pop(ibl_ind);
    roi_size.pop(ibl_ind);
    roi_names.pop(ibl_ind);

    if mapping is not None: #filter atlas data to only include rois in mapping, sort by sorted mapping rois
        rois = list(mapping.keys())
        rois.sort()
        pull_inds = []
        for r_ind, r in enumerate(rois):
            pull_inds.append(np.where(np.array(roi_names) == r)[0][0])

        roi_mask = [roi_mask[x] for x in pull_inds]
        roi_size = [roi_size[x] for x in pull_inds]

    return roi_mask, roi_size

def filterRegionResponse(region_response, cutoff=None, fs=None):
    """
    region_response is np array
    cutoff in Hz
    fs in Hz
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered

def trimRegionResponse(file_id, region_response, start_include=100, end_include=None):
    """
    file_id is string
    region_response is np array
        either:
            nrois x frames (region responses)
            1 x frames (binary behavior response)
    """

    # Key: brain file id
    # Val: time inds to include
    brains_to_trim = {'2018-10-19_1': np.array(list(range(100,900)) + list(range(1100,2000))), # transient dropout spikes
                      '2017-11-08_1': np.array(list(range(100,1900)) + list(range(2000,4000))), # baseline shift
                      '2018-10-20_1': np.array(list(range(100,1000)))} # dropout halfway through

    if file_id in brains_to_trim.keys():
        include_inds = brains_to_trim[file_id]
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, include_inds]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[include_inds]
    else: # use default start / end
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, start_include:end_include]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[start_include:end_include]

    return region_response_trimmed

def getProcessedRegionResponse(resp_fp, cutoff=None, fs=None):
    file_id = resp_fp.split('/')[-1].replace('.pkl', '')
    region_responses = pd.read_pickle(resp_fp)

    resp = filterRegionResponse(region_responses.to_numpy(), cutoff=cutoff, fs=fs)
    resp = trimRegionResponse(file_id, resp)

    region_responses_processed = pd.DataFrame(data=resp, index=region_responses.index)
    return region_responses_processed

def getBehavingBinary(motion_filepath):
    motion_df = pd.read_csv(motion_filepath, sep='\t')
    num_frames = 2000 # imaging frames
    total_time = motion_df.loc[0, 'Total length'] #total time, synced to imaging start/stop
    starts = (num_frames*(motion_df.loc[:,'Start (s)'].to_numpy() / total_time)).astype(int) # in imaging frames
    stops = (num_frames*(motion_df.loc[:,'Stop (s)'].to_numpy() / total_time)).astype(int) # in imaging frames

    is_behaving = np.zeros(num_frames)
    for st_ind, start in enumerate(starts):
        is_behaving[start : stops[st_ind]] = 1

    return is_behaving

def getFunctionalConnectivity(response_filepaths, cutoff=None, fs=None):
    cmats_z = []
    for resp_fp in response_filepaths:
        region_responses_processed = getProcessedRegionResponse(resp_fp, cutoff=cutoff, fs=fs)

        correlation_matrix = np.corrcoef(region_responses_processed)
        # set diag to 0
        np.fill_diagonal(correlation_matrix, 0)
        # fischer z transform (arctanh) and append
        new_cmat_z = np.arctanh(correlation_matrix)
        cmats_z.append(new_cmat_z)

    cmats = np.stack(cmats_z, axis=2) # population cmats, z transformed

    # Make mean pd Dataframe
    mean_cmat = np.mean(cmats, axis=2)
    np.fill_diagonal(mean_cmat, np.nan)
    CorrelationMatrix_Functional = pd.DataFrame(data=mean_cmat, index=region_responses_processed.index, columns=region_responses_processed.index)

    return CorrelationMatrix_Functional, cmats

def computeRegionResponses(brain, region_masks):
    """
    brain is xyzt
    region_masks is list of xyz masks to use
    """
    region_responses = []
    for r_ind, mask in enumerate(region_masks):
        region_responses.append(np.mean(brain[mask, :], axis=0))

    return np.vstack(region_responses)
