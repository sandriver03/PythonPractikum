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
