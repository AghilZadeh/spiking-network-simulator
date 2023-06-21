####################################### spiking neural netwrok ##########################################

############################################ Populations ################################################


import numpy as np

class FSI:
    NN       =     np.array([100])               # neuron numbers
    Tag      =     {"FSI":0}                      # number labels starting from 0
    Colors   =     np.array([[0, 0, 1]])          # colors for plots
    size = len(NN)

    # connectivity parameters of populations (from column to rows)
    CP       =     np.array([[0.58]])             # connectivity probabilities
    CW       =     np.array([[-60.]])              # connectivity weigths (mv.ms/spike), negative is inhibitory
    mod      =     1                              # connectivity strength
    CW       =     CW*mod    
    
    
    # single neuron parameters of populations
    Tau      =     np.array([10.])                 # decay time constants (ms)
    
    
    # synaptic parameters
    Tausyn   =     10.                             # synaptic time constants for all synapses (ms)
    
    # synaptic failures
    consider_failure = False                      # turns failure calculations on or off (True or False)
    FailRate =     np.array([[0.5]])              # failure rates of spike transfer of synapses (between 0 to 1)
    failure_update_time = 50                      # time interval to update stochastic failure matrix for spikes (ms)

    
    # LIF parameters
    Vthr     =     np.array([-50.])            # threshold voltage (mv)
    Vreset   =     np.array([-80.])            # reset voltage (mv)
    Vrest    =     np.array([-70.])            # rest voltage (mv) and initial conditions
    Vspike   =     np.array([40.])             # spike voltage (mv)
    
    # external input parameters
    MeanIx   =     np.array([50.])             # mean value of external inputs (mv)
    SigmaIx  =     np.array([0.])              # standard deviation [inhomogeneity] of external inputs for differnt neurons (mv)
    
    # noise on neurons parameters
    SigmaNoise =   np.array([0.])              # standard deviation of Gaussian noise [in space and time] input on neurons (mv)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # to be calculated using mean dynamics of population
    MeanSpikeRate  =   np.array([0.])                   # mean spike count for each population for the whole time series
    SpikeRate  =   []                                   # spike rate time series in dt bins of populations
    CV         =   []                                   # CV(std/mean) using mean spike count for each population
    AutoCorr   =   []                                   # auto correlation of mean spike count for each population
    ISI        =   []                                   # interspike intervals distribution for each population
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



                # Populations properties (Adapted from Gittis et al. 2010 & Taverna et al. 2008) #
    
class Striatum:
    NN       =     np.array([490, 490, 20])             # neuron numbers
    Tag      =     {"dSPN":0, "iSPN":1, "FSI":2}        # number labels starting from 0
    Colors   =     np.array([[0, 0, 1],
                             [0, 0.5, 0],
                             [1, 0, 0]])                # colors for plots
    size = len(NN)

    # connectivity parameters of populations (from column to rows)
    CP       =     np.array([[0.26, 0.27, 0.53], 
                             [0.06, 0.36, 0.36], 
                             [0, 0, 0.58]])             # connectivity probabilities
    CW       =     np.array([[-0.4, -1.3, -5.], 
                             [-1.1, -1.1, -5.], 
                             [0, 0, -0.6]])             # connectivity weigths (mv.ms/spike), negative is inhibitory
    mod      =     10                                   # connectivity strength
    CW       =     CW*mod    
    
    
    # single neuron parameters of populations
    Tau      =     np.array([10., 10., 10.])            # decay time constants (ms)
    
    
    # synaptic parameters
    Tausyn   =     10.                                  # synaptic time constants for all synapses (ms)
    
    # synaptic failures
    consider_failure = False                            # turns failure calculations on or off (True or False)
    FailRate =     np.array([[0.7, 0.5, 0],
                             [0.7, 0.6, 0],
                             [0, 0, 0]])                # failure rates of spike transfer of synapses (between 0 to 1)
    failure_update_time = 50                            # time interval to update stochastic failure matrix for spikes (ms)

    
    # LIF parameters
    Vthr     =     np.array([-50., -50., -50.])            # threshold voltage (mv)
    Vreset   =     np.array([-80., -80., -80.])            # reset voltage (mv)
    Vrest    =     np.array([-70., -70., -70.])            # rest voltage (mv) and initial conditions
    Vspike   =     np.array([40., 40., 40.])               # spike voltage (mv)
    
    # external input parameters
    MeanIx   =     np.array([30., 26., 22.])
    #MeanIx   =     np.array([45., 35., 23.])               # mean value of external inputs (mv)
    SigmaIx  =     np.array([0., 0., 0.])                  # standard deviation [inhomogeneity] of external inputs for neurons (mv)
    
    # noise on neurons parameters
    SigmaNoise =   np.array([2., 2., 2.])                  # standard deviation of Gaussian noise [in space and time] input on neurons (mv)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # to be calculated using mean dynamics of population
    MeanSpikeRate  =   np.array([0, 0, 0])              # mean spike count for each population for the whole time series
    SpikeRate  =   []                                   # spike rate time series in dt bins of populations
    CV         =   []                                   # CV(std/mean) using mean spike count for each population
    AutoCorr   =   []                                   # auto correlation of mean spike count for each population
    ISI        =   []                                   # interspike intervals distribution for each population
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

                # Populations properties (Adapted from Gittis et al. 2010 & Taverna et al. 2008) #
    
class Striatum2:
    NN       =     np.array([490, 490, 20])             # neuron numbers
    Tag      =     {"dSPN":0, "iSPN":1, "FSI":2}        # number labels starting from 0
    Colors   =     np.array([[0, 0, 1],
                             [0, 0.5, 0],
                             [1, 0, 0]])                # colors for plots
    size = len(NN)

    # connectivity parameters of populations (from column to rows)
    CP       =     np.array([[0.26, 0.27, 0.53], 
                             [0.06, 0.36, 0.36], 
                             [0, 0, 0.58]])             # connectivity probabilities
    CW       =     np.array([[-0.4, -1.3, -5.], 
                             [-1.1, -1.1, -5.], 
                             [0, 0, -0.6]])             # connectivity weigths (mv.ms/spike), negative is inhibitory
    mod      =     100                                 # connectivity strength
    CW       =     CW*mod    
    
    
    # single neuron parameters of populations
    Tau      =     np.array([10., 10., 10.])            # decay time constants (ms)
    
    
    # synaptic parameters
    Tausyn   =     10.                                  # synaptic time constants for all synapses (ms)
    
    # synaptic failures
    consider_failure = False                            # turns failure calculations on or off (True or False)
    FailRate =     np.array([[0.7, 0.5, 0],
                             [0.7, 0.6, 0],
                             [0, 0, 0]])                # failure rates of spike transfer of synapses (between 0 to 1)
    failure_update_time = 50                            # time interval to update stochastic failure matrix for spikes (ms)

    
    # LIF parameters
    Vthr     =     np.array([-50., -50., -50.])            # threshold voltage (mv)
    Vreset   =     np.array([-80., -80., -80.])            # reset voltage (mv)
    Vrest    =     np.array([-70., -70., -70.])            # rest voltage (mv) and initial conditions
    Vspike   =     np.array([40., 40., 40.])               # spike voltage (mv)
    
    # external input parameters
    MeanIx   =     np.array([72., 60., 33.])               # mean value of external inputs (mv)
    SigmaIx  =     np.array([0., 0., 0.])                  # standard deviation [inhomogeneity] of external inputs for neurons (mv)
    
    # noise on neurons parameters
    SigmaNoise =   np.array([2., 2., 2.])                  # standard deviation of Gaussian noise [in space and time] input on neurons (mv)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # to be calculated using mean dynamics of population
    MeanSpikeRate  =   np.array([0, 0, 0])              # mean spike count for each population for the whole time series
    SpikeRate  =   []                                   # spike rate time series in dt bins of populations
    CV         =   []                                   # CV(std/mean) using mean spike count for each population
    AutoCorr   =   []                                   # auto correlation of mean spike count for each population
    ISI        =   []                                   # interspike intervals distribution for each population
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class OnePopulation:
    NN       =     np.array([1000])               # neuron numbers
    Tag      =     {"pop":0}                      # number labels starting from 0
    Colors   =     np.array([[0, 0, 1]])          # colors for plots
    size = len(NN)

    # connectivity parameters of populations (from column to rows)
    CP       =     np.array([[1]])             # connectivity probabilities
    CW       =     np.array([[-1.]])              # connectivity weigths (mv.ms/spike), negative is inhibitory
    mod      =     1                              # connectivity strength
    CW       =     CW*mod    
    
    
    # single neuron parameters of populations
    Tau      =     np.array([10.])                 # decay time constants (ms)
    
    
    # synaptic parameters
    Tausyn   =     10.                             # synaptic time constants for all synapses (ms)
    
    # synaptic failures
    consider_failure = False                      # turns failure calculations on or off (True or False)
    FailRate =     np.array([[0.5]])              # failure rates of spike transfer of synapses (between 0 to 1)
    failure_update_time = 50                      # time interval to update stochastic failure matrix for spikes (ms)

    
    # LIF parameters
    Vthr     =     np.array([-50.])            # threshold voltage (mv)
    Vreset   =     np.array([-80.])            # reset voltage (mv)
    Vrest    =     np.array([-70.])            # rest voltage (mv) and initial conditions
    Vspike   =     np.array([40.])             # spike voltage (mv)
    
    # external input parameters
    MeanIx   =     np.array([200.])             # mean value of external inputs (mv)
    SigmaIx  =     np.array([0.])              # standard deviation [inhomogeneity] of external inputs for differnt neurons (mv)
    
    # noise on neurons parameters
    SigmaNoise =   np.array([0.])              # standard deviation of Gaussian noise [in space and time] input on neurons (mv)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # to be calculated using mean dynamics of population
    MeanSpikeRate  =   np.array([0.])                   # mean spike count for each population for the whole time series
    SpikeRate  =   []                                   # spike rate time series in dt bins of populations
    CV         =   []                                   # CV(std/mean) using mean spike count for each population
    AutoCorr   =   []                                   # auto correlation of mean spike count for each population
    ISI        =   []                                   # interspike intervals distribution for each population
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


