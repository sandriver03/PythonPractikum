import pickle
import numpy as np
import matplotlib.pyplot as plt

# load data from pickle file
with open('MembranePotential.pkl', 'rb') as fh:
    data, sampling_freq = pickle.load(fh)

# plot the membrane potential from one trial
plt.plot(data[:, 0])
plt.plot(data[:, 1])
plt.show()

# if you are using pt4agg backend, to have multiple line on the same figure
# you need to do the following:
fig, axes = plt.subplots()
# axes.hold()
axes.plot(data[:, 0], 'b')  # plot function accept additional parameters to specify line properties
axes.plot(data[:, 1], 'r')
plt.show()

# now lets get spike time
# say, use -30 mV as threshold to decide spike time
# get all the points higher than -30mV (-0.03V) in first recording
thresh = -0.03
Vm0 = data[:, 0]   # first recording
Vm0_th_idx = np.where(Vm0 > thresh)[0]   # indexes above threshold
Vm0_th_val = Vm0[Vm0_th_idx]    # values above threshold

fig, axes = plt.subplots()
# axes.hold()
axes.plot(Vm0)
axes.plot(Vm0 > thresh, 'r')   # can you scale this plot so it looks better?
plt.legend(('original', 'thresholded'))
plt.show()

#
#
#
#
#
#
#
#
#
# as you can see, we have 'blocks', and each block is one spike. We only need one time point for
# each spike. how can you separate these blocks?
# check numpy.diff function
spk_region = []
Vm0_th_idx_diff = np.diff(Vm0_th_idx)
idx_start = 0
for i in np.where(Vm0_th_idx_diff > 1)[0]:
    spk_region.append(tuple(Vm0_th_idx[idx_start:i+1]))
    idx_start = i + 1
# TODO: is any spike missing from the above code?
# if we use the time when Vm cross -30mV as spike time, we just need to take the first index in each
# spike region we identified in spk_region, i.e. each blocks
# of cause we can also find the peak of the spike. How to do it?
spt0 = [i[0] for i in spk_region]

fig, axes = plt.subplots()
# axes.hold()
axes.plot(Vm0)
axes.plot(spt0, Vm0[spt0], '.r')   # can you scale this plot so it looks better?
plt.show()

# note that the unit of spt0 is not time, but number of samples. To get actual spike time, we need to
# convert the unit into time. We only need to know the sampling frequency, i.e. how long in time is one
# sample represent. In this dataset, sampling frequency is provided as sampling_freq. Its unit is Hz
spt0 = np.array(spt0)/sampling_freq

# ToDo: can you write a function to extract spike time in Vm recordings, as those provided in data?
# TODO: can you also try to calculate spike width and spike amplitude, and inter-spike interval? try to plot them as well


"""
raster plot
"""
# raster plot is the representation of spikes as function of time
# here is an example of raster plot for a single trial, which we calculated previously
plt.vlines(spt0, 0, 1)  # you can also try to plot with plt.plot() function
plt.show()

# most of time, we also want to indicate when in time the stimulus was present. in the example data, we do not
# have such information. for illustration purpose, let's assume the stimulus was present during 2 - 3 second
fig, axes = plt.subplots()
# axes.hold()
axes.vlines(spt0, 0, 1)
axes.axvspan(2, 3, alpha=0.1, color='b')  # you can try to change the arguments to get different effect
plt.show()

# TODO: can you plot the rasters for all the trials in the data set?


"""
peri-stimulus histogram (PSTH)
"""
# find spike times for all trials
spt_all = []
for i in range(data.shape[-1]):
    spt_all.append(findspikes(data[:, i]))

# now lets calculate PSTH. simply put, it is calculating the average firing rate in each time bin
# first we need to decide bin size
bin_size = 0.05  # 50 ms; the unit in spt is second
# use numpy.histogram to bin the spike times
# now only calculate the psth for first 4 second
bin_range = [0, 4]
num_bins = int((bin_range[1]-bin_range[0])/bin_size)
# one trial, also get time points
spt0_binned, t0 = np.histogram(spt_all[0], num_bins, bin_range)
# all trials
spt_binned = np.zeros((num_bins, spt_all.__len__()), np.int64)
for i in range(spt_all.__len__()):
    spt_binned[:, i] = np.histogram(spt_all[i], num_bins, bin_range)[0]
# calculate psth from all trials
psth = spt_binned.mean(1)
# here what is the unit of the psth? do you need to convert the unit to something else?

# plot psth
# you can use either simple plot or bar plot; it is a personal choice
fig = plt.figure()
ax = plt.subplot(211)
t0 = np.linspace(bin_range[0]+bin_size/2, bin_range[1]-bin_size/2, num_bins)
ax.plot(t0, psth)
plt.xlabel('time (second)')
plt.ylabel('??')
# again, lets assume stimulus lasts from 2 - 3 second. can you label it on the figure?
ax = plt.subplot(212)
t0 = np.linspace(bin_range[0], bin_range[1]-bin_size, num_bins)  # why time vector is different?
ax.bar(t0, psth, bin_size)
# put labels
plt.show()

# TODO: try different bin size and/or different time range to see what happens
# TODO: can you generate a similar figure as shown in the slides?


"""
tuning curve
"""
# tuning curve is simply the average firing rate of neuron in different stimulus condition, plotted
# as a figure
# use data from 'tenIntensities.pkl'
# in this data set, a single neuron was stimulated with stimuli at 10 different intensities, and the
# responses are recorded
# load data from the pickle file
# we already imported pickle module
with open('tenIntensities.pkl', 'rb') as fh:
    data = pickle.load(fh)

# we can check what is inside the data
print(data)

# as you can see, the data does not include membrane potentials, but just spike times
# the response to each stimulus is recorded as a list in the dictionary; the list again
# contains multiple lists, each list is the spike time in one trial
# first, to inspect the behavior of the cell, let's make a raster plot as well as a psth plot
# TODO

# let's calculate the average firing rate for each stimulus condition. naturally, we only interested
# in spikes occurred after stimulus onset
# TODO

# now since we have the average firing rates (maybe the variance as well), we can plot them as a
# function of stimulus intensity. this plot is the tuning curve in response to stimulus intensity
# for this neuron
# TODO


"""
spike train correlation
"""
# sometimes you are interested in how different neurons response to the same stimulus in the same trial.
# this is done by calculating the spike train correlations
# generally speaking, you must have multiple simultaneously recorded neurons, otherwise the correlation
# analysis does not make sense
# all our previous data does not contain simultaneously recorded cells. we need the data stored in
# 'arrayData.pkl'. for the description of the data, check the exercise_1, question 2
# load data
with open('arrayDATA.pkl', 'rb') as fh:
    data = pickle.load(fh)

# our data is in 'arrayData'
data = data['arrayData']
# data is a list containing 2300 element, which correspond to 2300 trials performed in the experiment
# this data is very hard to analyze. generally speaking, at the start of any data analysis, you need to
# prepare the data in an easy to use structure
# there are many ways to organize the data so it is easy to use. here, what we are going to do
# is to organize the data into a form resemble unit-trial-spiketimes, i.e. for each identified unit, store
# the spike times for every spike in each trial. trials with no spike will be an empty list
# the following code is not very efficient, but it is (hopefully) easier to read. you can try to write
# more efficient code. generally speaking, using numpy functions is more efficient compared with using python
# functions
# we need the stimulus orientation in each trial so we can sort the data accordingly later on
ori = np.array([ele['ori'] for ele in data]).ravel()
uni_ori = np.unique(ori)
# find all the units identified in the data
# each electrode+unit pair is a unique presumed neuron
# first get a numpy array with all electrode-unit pairs from all trials
# electrode is in index 0, and unit is in index 1
# to get a unique row in numpy array, check the numpy.unique function
# for example, to get unique units in each trial, say trail 0
u_unit_trial0 = np.unique(data[0]['response'][:, 0:2], axis=0)
# list of unique units in each trial
u_unit_all = [np.unique(ele['response'][:, 0:2], axis=0) for ele in data]
# however, we need to get unique units in all trials
# let's also keep track of if the unit is presumably noise
is_noise = []   # bool, if the unit is noise
uni_units = []  # list of unique units
for ele in u_unit_all:
    ele = ele.tolist()
    for row in ele: # loop over all rows
        if row not in uni_units:
            uni_units.append(row)
            if row[1] in (0, 255):
                is_noise.append(True)
            else:
                is_noise.append(False)
# convert them in to numpy array
is_noise = np.array(is_noise)
uni_units = np.array(uni_units)

# now let's sort the spike times for each identified unit. we can skip the noise for now
# TODO:
# if you are interested, you can also do the same analysis to the noise (e.g. plot rasters and
# psths) and see what it looks like

# for the unit-trial-spiketimes data, there are many options: for each unit, storing as a
# N_trial-by-Bin_time array of 0s and 1s probably is easiest for later analysis, but this requares
# very large arrays: in our case, if we use time resolution of 1ms, since each trial is 3s, we will
# have 2300-by-3000 array for each unit, and we have 227 of these units in total, or 106 units that
# are not noise
# if we use int8 data type, this data will take 8*2300*3000*102/8/1000/1000 = 703.8 MB of RAM
# of cause we can use much larger time bin size at the expense of loosing time resolution
# SIDENOTE: actually most of the entry in this array will be 0 - in this case you can use a data type
# named sparse matrix. we are not going to discuss it, but if you are interested you can look it up
# yourself

# alternatively, we can save the data as N_trial-by-N_spike array, then our data will be in size of
# 2300-by-MaxNumberSpikes size, which is much smaller
# however in this case, we need to know the maximum number of spikes coming from a given unit in one
# trial, so we can initialize our array properly
# remember, change the size of numpy array is very costly, so if possible you should avoid doing it.
# here we are going to pre-allocate the size for each array
# as an example, lets look at our first non-noise unit, [20, 1]
# first find the maximum spike count in one trial, for all trials
# how to find the spike count for a given unit in one trial?
# response from trial 1
neural_data_trial1 = data[0]['response']
# we can compare the unit code, [20, 1] to the first 2 columns in the response using np.equal function
res_comp = np.equal(neural_data_trial1[:, 0:2], [20, 1])
# this will compare the 1st column of neural_data_trial1 to 20, and 2nd column to 1, and return a bool
# array of same size as neural_data_trial1
# if we find a row with both columns are true, then we find a match, i.e. this row is exactly [20, 1]
# we can check it now - if both columns are true, when we sum them up we will get value of 2 (remember
# False is 0 and True is 1)
idx_from_unit = res_comp.sum(axis=1) == 2
print(neural_data_trial1[idx_from_unit, :])
# the spike count in this trial is just the number of indexes we found
sc = np.sum(idx_from_unit)
# calculate sc (spike count) for all trials, for unit [20, 1]
# I will use some different functions, but the end result will be the same
sc_all = []
for trial in data:
    sc_all.append(np.sum(np.all(np.equal(trial['response'][:, 0:2], [20, 1]), axis=1)))
# now, the maximum of the spike count tells us what is the correct array size to use
# we can start generate our numpy array now to hold all the spikes from unit [20, 1], for all trials
spikes_unit0 = -1000*np.ones((data.__len__(), np.max(sc_all)))
# I use -1000 to indicate non-spikes. all values larger than this will be a spike in this array
# now we can go over the entire data set and put spikes from this unit in to the array
for idx, trial in enumerate(data):
    # spike time for this unit in the trial
    trial_data = trial['response']
    spt = trial_data[np.all(np.equal(trial_data[:, 0:2], [20, 1]), axis=1), 2]
    # put the spikes into the array
    spikes_unit0[idx, 0:spt.__len__()] = spt

# now we can do the same to all the non-noise unit
# TODO: get the spike time for all non-noise unit from all trials

# spike times from all non-noise unit
from SpikeTrains_Helper import Helper_dataorg
# lets only work on non-noise unit, first get these units
non_noise_units = uni_units[np.logical_not(is_noise)]
spikes_nnu = Helper_dataorg(data, non_noise_units)

# lets calculate PSTH for a couple of units for all trials
# let's check the activity from electrode 20, 58, 67, 78
# how can you get the spike times from these electrode? remember that each elements in spikes_nnu
# contains data from corresponding unit in non_noise_units. as a result, we can use non_noise_units
# to index spikes_nnu
from SpikeTrains_Helper import Helper_spt2psth
# first decide about binsize and range for psth. let's use 50ms bin size, and range of [-1, 3].
# remember that stimulus starts at 0, and the recording starts before the stimulus onset
bin_size = .05
bin_range = [-1, 3]
num_bins = int((bin_range[1]-bin_range[0])/bin_size)

# psth from unit 1, electrode 20
psth_20_1 = Helper_spt2psth(spikes_nnu[0], bin_size, bin_range, axis=1)
# psth from unit 1, electrode 58
psth_58_1 = Helper_spt2psth(spikes_nnu[52], bin_size, bin_range, axis=1)
# psth from unit 1, electrode 67
psth_67_1 = Helper_spt2psth(spikes_nnu[61], bin_size, bin_range, axis=1)
# psth from unit1, electrode 78
psth_78_1 = Helper_spt2psth(spikes_nnu[69], bin_size, bin_range, axis=1)

# these psths are tuple with 2 elements, 1st is the real psth, and 2nd is the time vector
# TODO: can you plot these psths?
# TODO: can you calculate PSTHs for different stimulus from a given unit as well?

# spike count correlation (with pearson correlation)
# how do you get spike count in each trial, for each unit?
# lets only consider the spikes after stimulus onset
sc_20_1 = (spikes_nnu[0] > 0).sum(1)
sc_58_1 = (spikes_nnu[52] > 0).sum(1)
sc_67_1 = (spikes_nnu[61] > 0).sum(1)
sc_78_1 = (spikes_nnu[69] > 0).sum(1)
# to calculate pearson correlation coefficient, use scipy.stats.pearsonr function
import scipy.stats as sst
# calculate pearson correlation coefficient between unit 20_1 and 58_1
pce, p = sst.pearsonr(sc_20_1, sc_58_1)
# TODO: explore the spike count correlation in the dataset. does the correlation depends on electrode?

# spike time correlation (with cross correlation)
# to calculate spike time correlation or cross correlation, you can use either numpy.correlate or
#  scipy.signal.correlate
# as an example, lets calculate the correlation of unit 20_1 with itself. in this case, the correlation
# is called auto-correlation
# in psth_20_1, we already have the spike train in form of 0s and 1s. we simply do the cross correlation
# with spike trains coming from the same trial
# e.g. xcr of trial 1:
xcr_1 = np.correlate(psth_20_1[0][0], psth_20_1[0][0], 'same')
# we can plot it out, remember our bin_size is .05s:
t = np.linspace(-2, 2, 80)   # why the time is from -2s to 2s?
plt.plot(t, xcr_1)
# TODO: calculate the average xcr for all trials and plot it




# function to find spikes in one trial
def findspikes(Vm0, thresh=-0.03, sampling_freq=20000):
    """
    find spike times in a whole cell recording
    :param vm: membrane potential traces
    :param thresh: threshold to detect spikes
    :param sampling_freq: sampling frequency of vm
    :return: numpy array of spike times, in second
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


result = []
for idx in range(data.shape[1]):
    result.append(findspikes(data[:, idx]))
spk_wd = []
spk_isi = []
for info in result:
    spk_wd.append([len(spk_re) for spk_re in info[2]])
    spk_isi.append(np.diff(info[0]))


# cross-correlation on a sine wave
import math
x = np.linspace(0, 10*math.pi, 10000)
y = np.sin(x)

xcr_y = np.correlate(y, y, mode='same')