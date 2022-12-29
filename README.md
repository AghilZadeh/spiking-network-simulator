# spiking neural network simulator

## Introduction

This repository aims at introducing a spiking neural network simulator along with providing instructions on how to implement such networks.  With this simulator, one can build leaky integrate and fire (LIF) neural networks with realistic synaptic transmissions and desired neural structures and inputs. 

The simulator basically runs a giant dynamical system, with the dynamical variables representing membrane potentials of neurons. 
Generally, one can use any sort of connectivity matrix in this simulator. Additionally, there are built-in functions that can construct multi-population networks by creating the appropriate connectivity matrices with given statistics. Moreover, any time-dependent input can be defined to feed into the network. Built-in functions are designed to construct random Poission inputs with given statistics similar to those of cortex.

The simulator outputs the selected neurons membrane potentials and spike times. Using these outputs, rastorplots and other visuals can be plotted using incorporated functions. Several statistical measures can also be calculated by built-in functions.

## Libraries

This simulator consists of three libraries as follows:

1. neurosimulator: This library is the main module for dynamics of a neural network. It consists of codes for initializing a network, running the dynamics on that network, visualizing the output and performing statistical analysis on the dynamics.
2. populations: This library consists of predefined populations to be used for neurosimulator. There exist single population and multi-population networks of striatum with inferred connectivity profiles from experimental literature.
3. plasticity: Here online plasticity rules can be defined to be added to the network dynamics. A predefined anti-hebbian rule is incorporated. With an online plasticity rule, connectivity weights of the network change depending on neural network activity such as pre-post neural firing rates.

## Sample code

a netwrok can be defined and run using few lines of code as simple as below:
'''python
import neurosimulator as ns
from populations import Striatum
ns.initNetwork(Striatum)
ns.runplot(1000, ns.monitorind([10,10,10]));
'''
The above code imports **neurosimulator** and from **populations** the striatum predefined population, consisting of three different cell types (FSI, dSPN, iSPN). Then it initializes the network in the simulator using **initNetwork()**. Using **runplot()**, the simulator runs the network for 1000 ms, chooses 10 randomly selected neurons from each three populations by **monitorind([10,10,10])** and plots the rasterplot of spike activity. 

