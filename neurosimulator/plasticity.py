####################################### spiking neural netwrok ##########################################


#######################################  PLASTICITY RULES   #########################################
import numpy as np
import neurosimulator.spiking as ns
import itertools


class Off:
    def __init__(self):
        self.istrue = False   # turns plasticity on or off with True or False

        
class AntiHebbian:
    
    def __init__(self):
        Neuron = ns.Neuron
        dt = ns.dt
        initW = Neuron.initWmat
        
        self.istrue = True    # turns plasticity on or off with True or False

        # kernel function properties
        self.Wdict       =  {-1:initW*2, 0:initW*0 ,1:initW*(1/2)}                   # convergent values for each sign in the kernel 
        self.alpha       =  0.1                                                      # convergence pace (exponential)         
        self.kwindow     =  20                                                       # half window of kernel (ms)
        
        self.monitorsize =  round(self.kwindow/dt) + 1 
        self.Monitor     =  np.zeros((self.monitorsize, Neuron.Tag.size))
        self.Kerneltime  =  np.linspace(-self.kwindow, +self.kwindow, 2*self.monitorsize-1)
        self.Kernelarray =  self.Kernel(self.Kerneltime)
        self.absKernel   =  np.abs(self.Kernelarray)
        self.TargetW     =  np.array([self.Wdict[s] for s in np.sign(self.Kernelarray)])

    def Kernel(self, t):            # STDP kernel with time as post-pre synaptic difference                 
        return np.sign(t)

    def updatedW(self):
        Neuron = ns.Neuron
        dt = ns.dt

        W  = Neuron.Wmat
        C  = Neuron.Cmat
        dW = np.zeros(Neuron.Wmat.shape)
        self.Monitor = np.append(self.Monitor[1:,:], [Neuron.Spike], axis=0)
        
        for i, j in Neuron.Cpairs:    # i is postsynaptic and j is presynaptic
            if (self.Monitor[-1,i] or self.Monitor[-1,j]):
                preSpike     = self.Monitor[:,j]
                postSpike    = self.Monitor[:,i]
                prepostarray = np.append(postSpike[:]*preSpike[-1], np.flip(preSpike[:-1]*postSpike[-1]))
                targetW      = self.TargetW[:,i,j]
                dW[i,j]      = -self.alpha*(W[i,j]-np.sum(targetW*self.absKernel*prepostarray))*dt
        return(W+dW)        
                
            

