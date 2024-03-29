import numpy as np

# helper function for last piece of work
def Helper_dataorg(arrayData, uni_units):
    # a list to hold all data
    sptArray_all = []

    for unit in uni_units:
        # first we need to get spike count from all trials
        sc = []
        for trial in arrayData:
            sc.append(np.sum(np.all(np.equal(trial['response'][:, 0:2], unit), axis=1)))

        # initialize np array
        spt_array = -1000 * np.ones((arrayData.__len__(), np.max(sc)))
        for idx, trial in enumerate(arrayData):
            # spike time for this unit in the trial
            trial_data = trial['response']
            spt = trial_data[np.all(np.equal(trial_data[:, 0:2], unit), axis=1), 2]
            # put the spikes into the array
            spt_array[idx, 0:spt.__len__()] = spt

        sptArray_all.append(spt_array)

    return sptArray_all


def Helper_spt2psth(spt, bin_size, bin_range, axis=0):
    # calculate psth from spike time data
    num_bins = int((bin_range[1] - bin_range[0]) / bin_size)
    # if spt has dimension of 1, do not need to check axis
    if len(spt.shape) == 1:
        psth, t0 = np.histogram(spt, num_bins, bin_range)
        return psth, t0
    elif len(spt.shape) == 2:
        # need to decide which axis to operate on
        if axis == 0:
            spt_binned = np.zeros((num_bins, spt.shape[1]), np.int64)
            for i in range(spt.shape[1]):
                psth, t0 = np.histogram(spt[:, i], num_bins, bin_range)
                spt_binned[:, i] = psth
        elif axis == 1:
            spt_binned = np.zeros((spt.shape[0], num_bins), np.int64)
            for i in range(spt.shape[0]):
                psth, t0 = np.histogram(spt[i, :], num_bins, bin_range)
                spt_binned[i, :] = psth
        else:
            raise ValueError('axis not understood')
        return spt_binned, t0
    else:
        raise ValueError('data dimension cannot be processed')


# function to find spikes in one trial
def findspikes(Vm0, thresh=-0.03, sampling_freq=20000):
    """
    find spike times in a whole cell recording
    :param vm: membrane potential traces
    :param thresh: threshold to detect spikes
    :param sampling_freq: sampling frequency of vm
    :return: numpy array of spike times, in second; list of indexes for the 1st data points in each spike; list of
        tuples for indexes of each spike region
    """
    # find regions contain spike
    Vm0_th_idx = np.where(Vm0 > thresh)[0]  # indexes above threshold
    Vm0_th_val = Vm0[Vm0_th_idx]  # values above threshold
    # get indexes of the regions
    spk_region = []
    Vm0_th_idx_diff = np.diff(Vm0_th_idx)
    idx_start = 0
    # the last spike is missing from the code
    for i in np.where(Vm0_th_idx_diff > 1)[0]:
        spk_region.append(tuple(Vm0_th_idx[idx_start:i + 1]))
        idx_start = i + 1
    # append the last spike
    spk_region.append(Vm0_th_idx[idx_start:])

    # spike times
    spt0_idx = []
    spt0 = np.array([])
    if Vm0_th_idx.size:
        spt0_idx = [i[0] for i in spk_region]
        spt0 = np.array(spt0_idx) / sampling_freq
    return spt0, spt0_idx, spk_region


# function to find spikes in one trial
def findspikes_v2(Vm0, thresh=-0.03, sampling_freq=20000):
    """
    a better, easier to understand version
    find spike times in a whole cell recording
    :param vm: membrane potential traces in one trial
    :param thresh: threshold to detect spikes
    :param sampling_freq: sampling frequency of vm
    :return: numpy array of spike times, in second; list of indexes for the 1st data points in each spike; list of
        tuples for indexes of each spike region
    """
    # find regions contain spike
    Vm0_th_idx = np.where(Vm0 > thresh)[0]  # indexes above threshold
    # now we just need to loop over those index, comparing consecutive values to see if the difference is larger than 1
    # compare current and previous
    pre_pos = 0
    # use list to hold the result
    spk_region = []
    # if nothing found, we can directly return
    if len(Vm0_th_idx) == 0:
        return [], [], spk_region
    # another list to hold indexes of current spike regions
    # the very first index obviously belongs to the first spike
    curr_spk = [Vm0_th_idx[pre_pos]]
    for curr_pos in range(1, len(Vm0_th_idx)):
        # check the difference of the two index values
        if Vm0_th_idx[curr_pos] - Vm0_th_idx[pre_pos] > 1:
            # start of a new spike
            # we have finished with current spike, append it into result and start a new one
            spk_region.append(curr_spk.copy())
            curr_spk = []
        else:
            # index pointed by curr_pos belong to the same spike
            curr_spk.append(Vm0_th_idx[curr_pos])
        # update pre_pos
        pre_pos = curr_pos
    # append last spike
    spk_region.append(curr_spk.copy())

    # spike times
    spt0_idx = []
    spt0 = np.array([])
    if Vm0_th_idx.size:
        spt0_idx = [i[0] for i in spk_region]
        spt0 = np.array(spt0_idx) / sampling_freq
    return spt0, spt0_idx, spk_region

