"""
Leaky-integrate-and-fire model and Hodgkin-Huxley model
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Neuron():

    def __init__(self, type, dt=0.1, noise_amp=0.):
        """
        create a single neuron
        :param type: str, type of the neuron
        :param dt: simulation time step, in millisecond
        :param noise_amp: noise amplitude in the neuron; noise is modeled with Gaussian distribution
        """
        # current simulation time
        self.t = 0
        # neuron type
        self.type = type
        self.dt = dt
        self.t = 0  # current time
        self.V = 0  # current membrane potential
        # parameters used to record
        self.V_all = None
        self.spikes = []
        self.I_all = None
        self.t_all = None
        self.current_step = 0
        # inputs and weights
        self.inputs = []
        self.weights = []
        self.delays = []
        # membrane potential noise parameter
        self.noise_amp = noise_amp

    def connect(self, inputs, weights=1, delays=0):
        """
        connect to the cell to different inputs
        :param inputs: list of object from constCurrent or synapticInput
        :param weights: list of same size as inputs; weight (strength) of each input
        :param delays: list of same size as inputs; synaptic delay of each input in millisecond
        :return: None
        """
        if isinstance(inputs, (constCurrent, synapticInput)):
            inputs = [inputs]
        self.inputs = inputs
        if not isinstance(weights, (list, tuple, np.ndarray)):  # in this case weights should be a number
            weights = weights * np.ones(len(inputs))
        elif len(weights) != len(inputs):
            raise ValueError('size of weights and inputs must match')
        if not isinstance(delays, (list, tuple, np.ndarray)):  # in this case delays should be a number
            delays = delays * np.ones(len(inputs))
        elif len(delays) != len(inputs):
            raise ValueError('size of weights and inputs must match')
        self.weights = weights
        self.delays = delays

    def update(self, I):
        """
        update the model for one time step
        """
        # record values
        self.V_all[self.current_step] = self.V
        self.I_all[self.current_step] = I

    def calculate_current(self, t):
        """
        calculate synaptic current or injected current
        :param t: current simulation time in millisecond
        :return: current the neuron receives, unit will be scaled by the time step used
        """
        activity = np.array([stim.output(t, self.dt) for stim in self.inputs])
        # add noise term
        if self.noise_amp:
            activity = activity + self.noise_amp * np.random.randn(*activity.shape)
        return np.sum(activity * self.weights)

    def initialize(self, duration):
        """
        initialize arrays to hold data
        """
        data_size = int(duration / self.dt)
        self.V_all = np.zeros(data_size)
        self.I_all = np.zeros(data_size)

    def run(self, duration=500):
        """
        run the neuron simulation with given duration and stimulus
        :param duration: duration of simulation in millisecond
        :return: None
        """
        # add synaptic delay to each input
        for idx, input in enumerate(self.inputs):
            if isinstance(input, synapticInput):
                input.spike_times += self.delays[idx]
            if isinstance(input, constCurrent):
                input.start += self.delays[idx]

        self.initialize(duration)

        self.t_all = np.linspace(0, duration - self.dt, int(duration/self.dt))
        for t in self.t_all:
            self.t = t
            I = self.calculate_current(t)
            self.update(I)
            self.current_step += 1

    def reset(self):
        """
        reset the state of the neuron to original, without change model parameters
        """
        self.V_all = None
        self.t = 0
        self.t_all = None
        self.I_all = None
        self.spikes = []
        self.current_step = 0


class LIFNeuron(Neuron):
    """
    a single leaky-integrate-and-fire neuron, equation:
        Cm*(dV/dt) = -gl*(V-El) + I
    membrane time constant is Cm/gl
    """

    def __init__(self, dt=0.1, Cm=100, El=-65, th=-50, gl=5, V_reset=-75):
        """
        :param dt: simulation time step, default is 0.1 ms
        :param Cm: membrane capacitance, default is 100 pF
        :param El: resting membrane potential, default is -65 mV
        :param th: spike threshold, default is -50 mV
        :param gl: membrane conductance, default is 5 nS
        :param V_reset: membrane potential after spike, default is -75 mV
        """
        Neuron.__init__(self, 'LIF', dt)
        self.El = El
        self.Cm = Cm
        self.th = th
        self.gl = gl
        self.V_reset = V_reset
        self.V = El  # set initial membrane potential to the resting membrane potential

    def update(self, I):
        # detect spike
        # note: at this point we are working on V from last update
        if self.V > self.th:
            # record the spike
            self.spikes.append(self.t - self.dt)
            # set peak of spike to 10 mV for visual display
            # reset membrane potential to reset membrane potential
            self.V_all[self.current_step - 1] = 10
            self.V = self.V_reset
        else:
            # update with 1st order Euler approximation
            self.V = self.V + self.dt*(I - (self.V - self.El)*self.gl)/self.Cm

        Neuron.update(self, I)


class HHNeuron(Neuron):
    """
    Hodgkin-Huxley model, equations:
        Cm*dV/dt = -gNa*m**3*h*(V-ENa) -gK*n**4*(V-EK) -gl*(V-El) + I
        dn/dt = alpha_n(V)*(1 - n) - beta_n(V)*n
        dh/dt = alpha_h(V)*(1 - h) - beta_h(V)*h
        dm/dt = alpha_m(V)(1 - m) - beta_m(V)*m
    where
        alpha_m(V ) =??? 0.32(V + 45)/(exp((V + 45)/4) ??? 1)
        beta_m(V ) = 0.28(V + 18)/(exp((V + 18)/5) ??? 1)
        alpha_h(V ) = 0.128exp(??? (V + 51 + a)/18)
        beta_h(V ) = 4/(1 + exp(??? (V + 28 + a)/5)
        alpha_h(V ) =??? 0.032(V + 43)/(exp(??? (V + 43/5) ??? 1)
        beta_h(V ) = 0.5exp(??? (V + 48)/40)
    in cortical neurons a non-inactivating K+ current (IM) is responsible for spike frequency adaptation:
        IM = gM*n*(V - EK)
        dn/dt = alpha_n(V)*(1 - n) - beta_n(V)*n
        alpha_n = 0.0001*(V + 30)/(1 - exp(-(V + 30)/9))
        beta_n = -0.0001*(V + 30)/(1 - exp(-(V + 30)/9))
    you can add the current yourself to see how this current affect the activity of the cell
    (see Destexhe and Par?? 1999), where gM = 0.5 mS/cm2, and gK is the same as above
    """

    def __init__(self, dt=0.01, Cm=1.0, gl=0.05, El=-70.0, gNa=60.0, ENa=50.0, gK=10.0, EK=-85.0):
        """
        :param dt: simulation time step, default is 0.01 ms. Note: larger time steps may cause numerical instability
        :param Cm: membrane capacitance, default is 1 microF/cm2
        :param gl: membrane leak conductance, default is 0.05 mS/cm2
        :param El: resting membrane potential, default is -70 mV
        :param gNa: maximum sodium conductance, 60 mS/cm2
        :param ENa: sodium equilibrium potential, 50 mV
        :param gK: maximum potassium conductance, 10 mS/cm2
        :param EK: potassium equilibrium potential, -90 mV
        """
        Neuron.__init__(self, 'HH', dt)
        self.Cm = Cm
        self.gl = gl
        self.El = El
        self.gNa = gNa
        self.ENa = ENa
        self.gK = gK
        self.EK = EK
        # initial condition
        self.V = self.El
        self.m = 0.05
        self.n = 0.5
        self.h = 0.2
        # debugging
        self.m_all = None
        self.n_all = None
        self.h_all = None

    def update(self, I):
        # first update m, n, h
        v_temp = self.V
        m0 = self.m
        n0 = self.n
        h0 = self.h
        self.m = m0 + self.dt * (self.alpha_m(v_temp) * (1 - m0) - self.beta_m(v_temp) * m0)
        self.h = h0 + self.dt * (self.alpha_h(v_temp) * (1 - h0) - self.beta_h(v_temp) * h0)
        self.n = n0 + self.dt * (self.alpha_n(v_temp) * (1 - n0) - self.beta_n(v_temp) * n0)
        # apply boundary condition
        self.m = max(0, min(1, self.m))
        self.h = max(0, min(1, self.h))
        self.n = max(0, min(1, self.n))
        # update V
        self.V = self.V + self.dt * (- self.gNa * m0 ** 3 * h0 * (v_temp - self.ENa)
                                     - self.gK * n0 ** 4 * (v_temp - self.EK)
                                     - self.gl * (v_temp - self.El) + I) / self.Cm

        Neuron.update(self, I)
        self.m_all[self.current_step] = self.m
        self.h_all[self.current_step] = self.h
        self.n_all[self.current_step] = self.n

    def initialize(self, duration):
        Neuron.initialize(self, duration)
        data_size = int(duration / self.dt)
        self.m_all = np.zeros(data_size)
        self.n_all = np.zeros(data_size)
        self.h_all = np.zeros(data_size)

    def reset(self):
        Neuron.reset(self)
        self.m_all = None
        self.h_all = None
        self.n_all = None
        self.V = self.El

    @staticmethod
    def alpha_m(V):
        return -0.32*(V + 45)/(np.exp(-(V + 45)/4) - 1)
        # return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))   # equation for squid axon

    @staticmethod
    def beta_m(Vm):
        return 0.28*(Vm + 18)/(np.exp(Vm + 18)/5 - 1)
        # return 4.0 * np.exp(-(Vm + 65.0) / 18.0)

    @staticmethod
    def alpha_h(Vm):
        return 0.128 * np.exp(-(Vm+51)/18)
        # return 0.07 * np.exp(-(Vm + 65.0) / 20.0)

    @staticmethod
    def beta_h(Vm):
        return 4/(1 + np.exp(-(Vm + 28)/5))
        # return 1.0 / (1.0 + np.exp(-(Vm + 35.0) / 10.0))

    @staticmethod
    def alpha_n(Vm):
        return -0.032*(Vm + 43)/(np.exp(-(Vm + 43)/5) - 1)
        # return 0.01 * (Vm + 55.0) / (1.0 - np.exp(-(Vm + 55.0) / 10.0))

    @staticmethod
    def beta_n(Vm):
        return 0.5*np.exp(-(Vm + 48)/40)
        # return 0.125 * np.exp(-(Vm + 65) / 80.0)

    @staticmethod
    def alpha_M(Vm):
        return 0.0001*(Vm+30)/(1 - np.exp(-(Vm+30)/9))

    @staticmethod
    def beta_M(Vm):
        return -0.0001*(Vm+30)/(1 - np.exp(-(Vm+30)/9))

    @staticmethod
    def dALLdt(X, t, self):
        """
        differential equation system to be integrated
        |  :param X: differentials to be integrated
        |  :param t: time vector to be integrated
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.calculate_current(t) -
                self.gNa * m ** 3 * h * (V - self.ENa) -
                self.gK * n ** 4 * (V - self.EK) -
                self.gl * (V - self.El)) / self.Cm
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt

    def oderun(self, duration=500):
        t = np.linspace(0, duration - self.dt, int(duration/self.dt))
        X = odeint(self.dALLdt, [self.El, 0.05, 0.5, 0.2], t, args=(self,))
        self.V_all = X[:, 0]
        self.m_all = X[:, 1]
        self.h_all = X[:, 2]
        self.n_all = X[:, 3]


class constCurrent():
    """
    generate constant current. for LIF neurons, use amplitude around 100. for HH neurons, use amplitude around 1
    """

    def __init__(self, start=100, duration=200, amplitude=200):
        """
        :param start: starting time of the current, millisecond
        :param duration: duration of the current, millisecond
        :param amplitude: amplitude of the current, unit will be scaled by the time step
        """
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.dt = None

    def output(self, t, dt=0.1):
        """"
        get the output of the source at time t
        :param t: float, time in millisecond
        :param dt: temporal resolution, default is 0.1 ms
        :return: current output; unit will be scaled by the time step
        """
        self.dt = dt
        if t >= self.start and t <= (self.start + self.duration):
            return self.amplitude
        else:
            return 0


class synapticInput():
    """synaptic activities"""

    def __init__(self, spike_times=(100.0, ), tau=3.0):
        """
        :param spike_times: tuple, list or array, time of presynaptic activities in millisecond
        :param tau: decay time constant of the synapse, default is 3.0 ms
        """
        # only get the unique spike times
        self.spike_times = np.unique(np.array(spike_times))
        # sort the spike times
        self.spike_times.sort()
        self.next_spike = 0
        self.dt = None
        self.val = 0  # current value of output
        self.tau = tau
        self.t = 0   # current time

    def output(self, t, dt=0.1):
        """
        get the synaptic activity at time t
        :param t: float, time in millisecond
        :param dt: temporal resolution, default is 0.1 ms
        :return: synaptic activity, 1 for active and 0 for no activity
        """
        self.dt = dt
        if np.any(np.logical_and(self.spike_times >= t-self.dt/2, self.spike_times < t+self.dt/2)):
            self.val += 1
        self.val = self.val - self.dt * self.val / self.tau
        self.t = t
        return self.val

    def poisson_spikes(self, start_time=100, duration=300, firing_rate=100, dt=0.1):
        """
        generate Poisson spike train within the interval [start_time, start_time + duration]
        :param start_time: float, millisecond
        :param duration: float, millisecond
        :param firing_rate: float, spikes/second
        :param dt: temporal resolution, default is 0.1 ms
        :return:
        """
        self.dt = dt
        # generate spikes based on time step
        n_bins = int(duration/self.dt)
        p = np.random.rand(n_bins)
        self.spike_times = start_time + np.where(p < self.dt*firing_rate * .001)[0] * self.dt


# TODO:
# 1. familiar yourself with working with objects
# 2. choose either the LIF neuron or HH neuron, find the threshold constant current (the minimum constant current that
        # make the neuron spiking)
# 3. do the same, this time use the synapticInput
# 4. let's say, we want the precision of the stimulation threshold to be 1 digit after the integer, i.e. should be in
#    form of 19.5 or 4.1 and so on. find the thresholds in this precision.
# 5. do the neuron parameters affect the value of the stimulation threshold? which neuron parameters affect the
#    stimulation threshold? how do these neuron parameters affect the stimulation threshold?
# 6. now lets add some noise in the system. in each neuron model, there is a 'noise_amp' parameter. now change this
#    parameter to non-zero (try to use a value that causes observable fluctuations, but not spontaneous spikes. A value
#    in range of [1, 100] seems working well for the LIF model). repeat 2-5. how should you measure the threshold
#    current in this case? and how adding system noise affect the threshold current value?
# 7. coincidence detector simulation:
    # a. connect the neuron to 2 synapses. with the threshold value you found in question 3, set the strength of each
    #    synapse to 75% of the threshold. run the simulation with each synapse having only 1 spike. now, modified the
    #    time difference between the two spikes from different synapses. what is the maximum time difference such that
    #    the simulated cell can still be driven to active? let's call this time difference the coincidence window
    # b. now modify the strength of the synapses. how the coincidence window is affected by the synaptic strength?
    # c. now modify the neuron parameters. which parameters you think will affect the coincidence window? how they
    #    affect the coincidence window? should you take it into account, if those parameters you changed also affect the
    #    stimulation threshold?
    # write a script to run the simulations, and make figures to report the result. you can decide yourself if you want
    # to include noise or not.
