import numpy as np
import matplotlib.pyplot as plt


"""
summing sine waves to get a square wave
"""
# sum sine waves to get a square wave
# the general form of sine wave is y = A*sin(2*pi*f*t), where A is the amplitude, f is the
# frequency and t is time
t = np.linspace(0, 10, 1000)

# sine wave with frequency of 1
sin_1 = np.sin(2*np.pi*1*t)
# sine wave with frequency of 3, with 1/3 amplitude
sin_3 = np.sin(2*np.pi*3*t)/3
# sine wave with frequency of 5, with 1/5 amplitude
sin_5 = np.sin(2*np.pi*5*t)/5
# sine wave with frequency of 7, with 1/7 amplitude
sin_7 = np.sin(2*np.pi*7*t)/7
# sine wave with frequency of 9, with 1/9 amplitude
sin_9 = np.sin(2*np.pi*9*t)/9

# lets plot the sum of these sine waves
plt.plot(t, sin_1 + sin_3 + sin_5 + sin_7 + sin_9)


"""
frequency information in noisy signal
"""
# sum of 10s sine waves of 2 frequencies, here 5 Hz and 7 Hz
t = np.linspace(0, 10, 1000)
sig = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*7*t)
# add white noise to the signal
sig_n = sig + 3*np.random.randn(sig.shape[0])
# plot the signal with and without noise
plt.figure(2)
ax = plt.subplot(211)
ax.plot(t, sig)
ax = plt.subplot(212)
ax.plot(t, sig_n, 'r')
# as you can see, in time domain the two frequencies is much harder to see in the
# noisy signal
# but what if we check these two signal from frequency domain?
# we do this by performing Fourier transform on the signals
# we are going to use scipy.fftpack.fft to do Fourier transform. you can read
# the documentation to understand the details
import scipy.fftpack as sfft

# in short, the fft algorithm will give you both the positive and negative frequency
# components of a signal. negative frequencies are useful when dealing with a complex
# signal; for us, since most times we are working on real valued signals, we can ignore
# the negative frequencies. the first [1, n//2] data points of the fft output is the real
# frequencies and the [n//2+1:] data points is the negative frequencies, where n is the
# length of the output
# the frequency resolution (i.e. frequency each data point represents) depends on the
# sampling frequency of the signal. the fft algorithm calculates frequencies in range of
# [0, sampling_freq/2]

# first we need to know the sampling frequency of our signal. we have 1000 data points in
# 10s, thus
sampling_freq = 1000/10
# fft transform on original and noisy signals
sig_f = sfft.fft(sig)
sig_n_f = sfft.fft(sig_n)
# sig_f and sig_n_f have the same length as sig and sig_n. the [1:len(sig_f)//2] is the
# positive frequencies. as discussed above, the maximum frequency is sampling_freq/2, which
# is 50 Hz
# for plotting, we need to x axis, in this case frequencies in the fft transform
xf = np.linspace(0, sampling_freq/2, len(sig)//2)
# sig_f and sig_n_f are array of complex numbers: for each frequency component, fft gives us
# both the amplitude (or power) and phase of each frequency component
# for now, we just plot the power. you can get the power (or amplitude of the complex number)
# by taking np.abs on that number. for example, plot the result for sig:
plt.figure(3)
plt.plot(xf, np.abs(sig_f[0:len(sig)//2])/(len(sig)//2))
# x axis is frequency, and y axis is the amplitude (power) at each frequency
# you can generate sine waves with different amplitudes and try the fft again

# lets plot all signals as well as their fft results together:
plt.figure(4)
ax = plt.subplot(221)
ax.plot(t, sig)
ax.set_xlabel('time, s')
ax = plt.subplot(222)
ax.plot(xf, np.abs(sig_f[0:len(sig)//2]))
ax.set_xlabel('frequency, Hz')
ax = plt.subplot(223)
ax.plot(t, sig_n, 'r')
ax.set_xlabel('time, s')
ax = plt.subplot(224)
ax.plot(xf, np.abs(sig_n_f[0:len(sig)//2]), 'r')
ax.set_xlabel('frequency, Hz')

# TODO: practice fft transform. generate signals by summing of sines with different frequencies and amplitudes
# TODO: add random noise to the signal you generated. perform fft on these signals again
# TODO: plot the signals and fft result (called power spectrum)

"""
aliasing
"""
# demonstration of aliasing
# a 10 Hz sine wave, sampled at 40 Hz
t = np.linspace(0, 2, 2*40)
y = np.sin(2*np.pi*10*t)
# under sampling y, now using sampling frequency of 20, 10, 5 Hz
t_2 = t[::2]   # under sampling by half, now at 20 Hz
y_2 = y[::2]
t_4 = t[::4]
y_4 = y[::4]
t_8 = t[::8]
y_8 = y[::8]

# lets see what are the under sampled data look like
plt.figure(5)
ax = plt.subplot(411)
ax.plot(t, y)
ax.set_title('original data')
ax = plt.subplot(412)
ax.plot(t_2, y_2)
ax.set_title('under sample by factor of 2')
ax = plt.subplot(413)
ax.plot(t_4, y_4)
ax.set_title('under sample by factor of 4')
ax = plt.subplot(414)
ax.plot(t_8, y_8)
ax.set_xlabel('time, s')
ax.set_title('under sample by factor of 8')

"""
spectrum and spectrogram
"""
# you get spectrum directly from fft analysis; see above
# spectrogram is simply multiple spectrum as a function of time, calculated using a moving
# window, i.e. calculated on a portion of the signal at a time
# to calculate spectrogram, use scipy.signal.spectrogram
# let's calculate the power spectrum as well as spectrogram of sparrow's singing, in file
# 'chipingsparrow.wav'
# there seems no platform-independent solution for playing sound in python, unless you install
# some new external packages. for windows, winsound works (we are going to use it here). for
# Linux, you probably can check out ossaudiodev. for mac, you probably want to checkout pyaudio
# one walk around certainly is write any sound into e.g. .wav file and play it outside of python
# reading wave files seems also very complicated in Python. for now, lets use scipy.io.wavfile
import winsound
from scipy.io import wavfile  # reading and writing .wav file
# play the sound in the file
winsound.PlaySound('chipingsparrow.wav', winsound.SND_FILENAME)
# read the file
sparrow_sound = wavfile.read('chipingsparrow.wav')
# the sparrow_sound is a tuple with 2 elements, 1st is the sampling frequency, 2nd is the sound
# itself
sparrow_sound_sf = sparrow_sound[0]
sparrow_sound_data = sparrow_sound[1]
# get time vector
sparrow_sound_t = np.linspace(0, len(sparrow_sound_data)/sparrow_sound_sf, len(sparrow_sound_data))
# by convention, sound data is floating numbers in range of [0, 1]. our data here is int16. we will
# convert the data now
sparrow_sound_data = sparrow_sound_data.astype('float32')/(2**16/2)
# we can write the data into a new .wav file
wavfile.write('sparrow_sound.wav', sparrow_sound_sf, sparrow_sound_data)
winsound.PlaySound('sparrow_sound.wav', winsound.SND_FILENAME)

# now we can calculate the power spectrum of the sound with fft
# first get the frequency vector
sparrow_sound_xf = np.linspace(0, sparrow_sound_sf/2, len(sparrow_sound_data)//2)
# fft
sparrow_sound_fft = sfft.fft(sparrow_sound_data)
# the power at each frequency
sparrow_sound_fft_mag = np.abs(sparrow_sound_fft)
# remember that we only need the first half of the result
# plot the power spectrum
plt.figure(6)
ax = plt.subplot(211)
ax.plot(sparrow_sound_t, sparrow_sound_data)
ax.set_xlabel('time, s')
ax = plt.subplot(212)
# generally speaking, it make sense to use a log scale plot
ax.loglog(sparrow_sound_xf, sparrow_sound_fft_mag[:len(sparrow_sound_xf)])
ax.set_xlabel('frequency, Hz')

# for spectrogram, we can simple use the spectrogram function from scipy.signal
import scipy.signal as ssig
# check the documentation for more detail
# in short, two parameters are important: the window you are using as well as how
# much overlapping between adjacent windows. remember that spectrogram is essentially
# calculating spectrum on a small part of the signal, move to the adjacent part, and
# calculate spectrum again until the end of the signal
# for different types of windows, check scipy.signal.get_window
# lets try out a hanning window
wd_length = 1024   # window length of 1024 data point
wd = ssig.windows.hann(wd_length)
overlap = wd_length - 16  # each window is 16 data points apart
f, t, Sxx = ssig.spectrogram(sparrow_sound_data, sparrow_sound_sf,
                             window=wd, nperseg=wd_length, noverlap=overlap)
# the function returns a tuple with list of frequencies, time vector, and the spectrum
# for the list of frequencies at each time point
# plot the spectrogram
plt.figure(7)
plt.pcolormesh(t, f, Sxx)
plt.xlabel('time, s')
plt.ylabel('frequency, Hz')
plt.colorbar()
# TODO: varing the window type and size and/or overlapping length to see what happens

# lets look at some other signal
fs = 4096  # sampling frequency
t0 = np.linspace(0, 2, fs*2)  # time vector
y = ssig.chirp(t0, 100, 1, 200, 'quadratic')  # a signal with increasing frequencies; check the function for more detail
# let's save the signal as wav file and listen to it
wavfile.write('chrip_sig.wav', fs, y.astype('float32'))
winsound.PlaySound('chirp_sig.wav', winsound.SND_FILENAME)
# TODO: plot the signal waveform
# TODO: power spectrum of the signal
# now let's look at its spectrogram
wd_length = 128
overlap = wd_length - 1
wd = ssig.windows.hann(wd_length)
f, t, Sxx = ssig.spectrogram(y, fs,
                             window=wd, nperseg=wd_length, noverlap=overlap)
# plot
plt.figure(8)
plt.pcolormesh(t, f, Sxx)
plt.xlabel('time, s')
plt.ylabel('frequency, Hz')
plt.colorbar()


"""
filtering
"""
# scipy.signal contains a series of methods to generate different filters
# for now let's use the Butterworth filter, since its passband is flat
# to get a Butterworth filter, checkout scipy.signal.butter
# the function will return 2 coefficient by default - the Numerator b and denominator a, which
# you will need to pass into the filter function, in this case scipy.signal.filtfilt
# simply filter the signal will cause phase shift. to avoid this, you need to filter it twice,
# once forward and once backward. this is automatically handled by the filtfilt function

# low pass filter, note the cutoff is relative to Nyquist frequency, in our case 2048 Hz
b_low, a_low = ssig.butter(8, 0.1, btype='low')
# filter the chirp signal generated above
ylow = ssig.filtfilt(b_low, a_low, y)
# spectrogram of low-pass filtered signal
f, t, Sxx_low = ssig.spectrogram(ylow, fs,
                             window=wd, nperseg=wd_length, noverlap=overlap)
plt.figure(9)
ax = plt.subplot(311)
ax.plot(t0, y)
ax.set_title('original signal')
ax = plt.subplot(312)
ax.plot(t0, ylow)
ax.set_title('low pass filtered, cutoff = 204.8 Hz')
ax.set_xlabel('time, s')
ax = plt.subplot(313)
im = ax.pcolormesh(t, f, Sxx_low)
ax.set_xlabel('time, s')
ax.set_ylabel('frequency, Hz')
ax.figure.colorbar(im)

# TODO: try out high-pass filter and band pass filter on the chirp signal, y


