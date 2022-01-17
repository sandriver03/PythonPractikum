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
# if we can get the spikes in 1 trial, then we can apply the method to every trials and get all the spikes
# TODO: how should we detect each spike in 1 trial? say, use -30 mV as threshold to decide spike time


# note the unit of spike times you got. Is it time, or number of samples? To get actual spike time, we need to
# convert the unit into time. We only need to know the sampling frequency, i.e. how long in time is one
# sample represent. In this dataset, sampling frequency is provided as sampling_freq. Its unit is Hz


# ToDo: can you write a function to extract spike time in Vm recordings, as those provided in data?
# TODO: can you also try to calculate spike width and spike amplitude, and inter-spike interval? try to plot them as
#  well


"""
raster plot
"""
# raster plot is the representation of spikes as function of time
# here is an example of raster plot for a single trial, with randomly generated spike times
# 10 spikes evenly distributed in [0, 1]
spt0 = np.random.rand(10, )
spt0.sort()
plt.vlines(spt0, 0, 1)  # you can also try to plot with plt.plot() function
plt.show()

# most of time, we also want to indicate when in time the stimulus was present. in the example data, we do not
# have such information. for illustration purpose, let's assume the stimulus was present during .2 - .3 second
fig, axes = plt.subplots()
# axes.hold()
axes.vlines(spt0, 0, 1)
axes.axvspan(.2, .3, alpha=0.1, color='b')  # you can try to change the arguments to get different effect
plt.show()

# TODO: can you plot the rasters for all the trials in the data set?


"""
peri-stimulus histogram (PSTH)
"""
# again, just as an example, randomly generate spikes from 20 trials
# here the first dimension is trial, you can use different arrangement
spt_all = np.random.rand(20, 10)
spt_all.sort(axis=1)

# now lets calculate PSTH. simply put, it is calculating the average firing rate in each time bin
# first we need to decide bin size
bin_size = 0.05  # 50 ms; the unit in spt is second
# use numpy.histogram to bin the spike times
# now only calculate the psth for 1 second, because that is the time range of the simulated data
bin_range = [0, 1]
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

# TODO: get all the spikes from the recording, and use them to calculate the psth and plot it
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
