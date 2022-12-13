# spiking neural network simulator
This repository aims at introducing a spiking neural network simulator along with providing instructions on how to implement such networks.  With this simulator, one can build leaky integrate and fire (LIF) neural networks with realistic synaptic transmissions and desired neural structures and inputs. 

The simulator basically runs a giant dynamical system, with the dynamical variables representing membrane potentials of neurons. 
Generally, one can use any sort of connectivity matrix in this simulator. Additionally, there are built-in functions that can construct multi-population networks by creating the appropriate connectivity matrices with given statistics. Moreover, any time-dependent input can be defined to feed into the network. Built-in functions are designed to construct random Poission inputs with given statistics similar to those of cortex.

The simulator outputs the selected neurons membrane potentials and spike times. Using these outputs, rastorplots and other visuals can be plotted using incorporated functions. Several statistical measures can also be calculated by built-in functions.

This repository is under construction and may change!
