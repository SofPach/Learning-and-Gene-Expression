#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
import pickle
import scipy.stats.qmc as qmc
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import timeit
import concurrent.futures
rdm_seed = 123
# In[ ]:


def external_input_to_GRN(V, coupling, dependence):
    ## Spike rate array computation
    V_spikes = np.zeros(int(V.shape[0]/10/10))
    for i in range(int(V.shape[0]/10/10)):
        V_spikes[i] = np.sum(V[i*100:(i+1)*100]==40)*10

    if coupling=='no':
        V_spikes_rescaled = V_spikes-3.5
    else:
        ## Rescaling V_spikes
        maxx = 100
        x = V_spikes-maxx/2
        #y = 3.5*x/np.sqrt(1+x**2)
        y = 3.5*x/(1+np.abs(x))
        V_spikes_rescaled = y

    ## WHAT DEPENDENCE DO WE WANT OUR GENES TO HAVE ON SPIKES/MEMBRANE POTENTIAL? ##
    if dependence == '+r': # direct dependence
        g_ext = V_spikes_rescaled
    elif dependence == '-r': # inverse dependence
        g_ext = -V_spikes_rescaled
    else: # no dependence
        g_ext = np.zeros(len(V_spikes_rescaled))

    return g_ext


# In[ ]:

def g(u):
    '''
    Sigmoid function 
    '''
    gu = 1/2*(u/np.sqrt(u**2+1)+1)
    return(gu)


############################################## PARAMETERS AND VECTORS FOR FOLLOWING FUNCTION #################################################
N = 2
## TIME ##
tf = 1000
dt_neur = 0.001
t_neur = np.arange(0, tf, dt_neur)
dt_gene = 0.1
t_gene = np.arange(0, tf, dt_gene)

# initialization of coupling arrays
V_th = np.zeros((len(t_neur)))
h = np.zeros((N, len(t_gene)))

## PARAMETERS neuron##
tau = 0.01 # s (seconds)
V_rest = -65 # mV
R_m = 10 # MOhm
t_refract = 0.008 # s (seconds)
E_L = -65
V_th_theoretical = -50 # mV
V_th = np.ones((len(t_neur)))*V_th_theoretical

## VECTORS neuron##
ii = [1]*120000 + [4]*20000 + [1]*60000 
I_e = np.array(ii*5) # input current


V:np.ndarray = np.zeros(len(t_neur)) # we create a vector of zeros for the storage of membrane potentials along time
V[0] = V_rest # seting the initial value for membrane potential
V_non_coup = np.zeros(V.shape[0])

## PARAMETERS genes ##
R = 1
d = 0.05

## VECTORS genes ##
gene_conc_base = np.zeros((N, len(t_gene)))
gene_conc_spike = np.zeros((N, len(t_gene)))
gene_conc_diff = np.zeros((N, len(t_gene)))
gene_conc_dev = np.zeros((N, len(t_gene)-1))
##################################################################################################################################################33

#def set_item(arr, index, value):
#    arr[index] = value

def while_in_coupled(V, j, k, spike):
    i = j*100 + k
    V[i] = 1/tau*(E_L - V[i-1] + R_m*I_e[i])*dt_neur + V[i-1] # fill V[i] with the value for membrane potential yield by the LIF model equation
    
    if V[i]>V_th[i] and i<(len(t_neur)-3): # if such a value (V[i]) beats the threshold value V_th and we have enough space for a spike in our vector
                                   # we consider that the neuron spikes and, therefore, we glue an action potential
        spike = spike + 1 # add 1 to the spike count
        
        # DEPOLARIZATION
        V[i] = 40 # change the V[i] value to the depolarizaed value that characterizes the spike
        i = i+1 # increment our time index
        
        # HYPERPOLARIZATION
        V[i] = V_rest - 10 # set the next V[i] to the hyperpolarized value that characterizes spikes after depolarization
        i = i+1 # increment our time index
        
        # REFRACTORINESS
        new_i = i + int(t_refract/dt_neur) # select our final index for refractoriness (the neuron will be in its refractory period from i to new_i)
        V[i:new_i] = V_rest # set V_rest as the value of the neuron's membrane potential for the whole refractory period
        i = new_i - 1 # update our time index to the time in which V was last filled
    return (i, spike)

def coupled_system(N, param_set, prop_list, ic):
    T = param_set.reshape((N, N))

    gene_conc_base[:, 0] = ic
    gene_conc_spike[:, 0] = ic
    global V
    V = np.zeros(len(t_neur))
    spike = 0
    
    ## ALGORITHM ##
    for j in range(len(t_gene)):
        if j==0:
            k = 1
        else:
            k = 0
        while k<100:
            i, spike = while_in_coupled(V, j, k, spike)
            i = i+1 # increase the time index by one and turn back to check while condition
            k = -j*100 + i
        
        h_base = np.zeros(N)
        h_spike = np.zeros(N)
        for gene in range(N):
            g_ext_base = external_input_to_GRN(V_non_coup[j*100:(j+1)*100], 'no', prop_list[gene]) # V of the last 100 neuronal timesteps
            h_base[gene] = g_ext_base
        
            g_ext_spike = external_input_to_GRN(V[j*100:(j+1)*100], 'yes', prop_list[gene]) # V of the last 100 neuronal timesteps
            h_spike[gene] = g_ext_spike
            
        if j<len(t_gene)-1:
            u = np.sum(T*gene_conc_base[:,j], axis=1) + h_base*gene_conc_base[:,j]
            gene_conc_base[:, j+1] = dt_gene*(R*g(u) - d*gene_conc_base[:, j]) + gene_conc_base[:, j]
        
            u = np.sum(T*gene_conc_spike[:,j], axis=1) + h_spike*gene_conc_spike[:,j]
            gene_conc_spike[:, j+1] = dt_gene*(R*g(u) - d*gene_conc_spike[:, j]) + gene_conc_spike[:, j]
        
            gene_conc_diff[:, j+1] = np.abs(gene_conc_spike[:, j+1] - gene_conc_base[:, j+1])
        
            gene_conc_dev[:, j] = (gene_conc_diff[:, j+1]-gene_conc_diff[:, j])/dt_gene
        
            if (10000 + 100*(j+1)) < len(t_neur):
                V_th[10000 + 100*j: 10000 + 100*(j+1)] = V_th[10000 + 100*j - 2] + 0.1*np.sum(gene_conc_dev[:, j])

    return(V, V_th, gene_conc_spike, gene_conc_base, gene_conc_dev, spike)


# In[ ]:

def parameter_search(topology, n_genes, n_samples, rdm_seed):
    ''' 
    Given a file with all possible network topologies for a certain number of genes, this function samples for each of them a 
    certain number of combinations of parameters based on the discrepancy of the method. The method for the sampling is the
    Quasi Monte Carlo Latin Hypercube from scipy.stats.

    INPUTS: 
    - network_space_file_name: name of the txt file containing row vectors corresponding to the interaction matrices 
                               of all possible n_gene network topologies where
                               - 0 means no action 
                               - 1 means activation
                               - 2 means inhibition
    - n_genes: number of genes we are considering for our networks
    - n_samples: number of samples we want to draw with our sampling method
    - rdm_seed (for reproductibility of results)
    
    OUTPUTS:
    - top_new: 2d array where each row corresponds to a network topology with a certain set of parameter values for the 
           1's (activations) and 2's (inhibitions) in between their corresponding bounds (we set the activation bounds
           to [0.1, 3.5] and the inhibition bounds to [-3.5, -0.1]
    '''
    # Identify the number of parameters we will want to sample, those in the topology that are different from 0
    non_0 = topology[topology!=0] 
    n_param = non_0.shape[0] 

    # Establish lower and upper bounds for the parameters we want to sample
    l_bound = np.zeros(n_param)
    u_bound = np.zeros(n_param)
    
    l_bound [non_0==1] = 0.1 # lower bound for activation parameters
    u_bound [non_0==1] = 3.5 # upper bound for activation parameters
    
    l_bound [non_0==2] = -3.5 # lower bound for inhibition parameters
    u_bound [non_0==2] = -0.1 # upper bound for inhibition parameters

    # Latin Hypercube sampling
    lhc_sampler = qmc.LatinHypercube(d=n_param, seed=rdm_seed) # it samples n_param parameters
    sample = lhc_sampler.random(n = n_samples) # n_sample times each n_param parameters
    # discrepancy = qmc.discrepancy(sample) # use if we want to know the discrepancy (the lower the better)
    sample = qmc.scale(sample, l_bound, u_bound) # scale the sampling based on the lower and upper bounds
    
    # Put back sampled parameters into their topology
    sample_row = sample.reshape(n_param*n_samples)
    top_new = np.tile(topology, (n_samples,1))
    top_new[top_new!=0] = sample_row #return an array with n_samples rows 

    return (top_new)


def feature_extraction(spike, gene_conc_spike, gene_conc_base):
    ## Extraction of features ##
    features = np.zeros(4)
    n_point = 4000
    #1 number of spikes
    features[0] = spike
    #2 fluctuation of each gene in last n points
    fluctuation = np.sum(np.abs(gene_conc_spike[:,n_point+1:]-gene_conc_spike[:,n_point:-1]), axis = 1)
    features[1:3] =  fluctuation
    #3 sum of the difference of stable state in gene_spike_conc and gene_base_conc
    stable_spike_search = np.round(gene_conc_spike, 3)
    stable_spike_0 = max(set(stable_spike_search[0]), key = list(stable_spike_search[0]).count)
    stable_spike_1 = max(set(stable_spike_search[1]), key = list(stable_spike_search[1]).count)
    stable_base_search = np.round(gene_conc_base, 3)
    stable_base_0 = max(set(stable_base_search[0]), key = list(stable_base_search[0]).count)
    stable_base_1 = max(set(stable_base_search[1]), key = list(stable_base_search[1]).count)
    features[3] = np.abs(stable_spike_0-stable_base_0) + np.abs(stable_spike_1-stable_base_1)
    return(features)


########################### This is a vectorized equivalent of the function we had before  (easier to parallelize in the future(?))
    
def create_arrays(N, n_samples, n_ic, network_space_file_name, num_features):
    # Things I need for the next function
    data = np.loadtxt(network_space_file_name, delimiter=' ')
    ic_combinations = list(itertools.product(np.linspace(0,20,n_ic), np.linspace(0,20,n_ic)))
    list_dep = np.array(['+r', '-r', 'nd'])
    list_dep_combinations = np.array(list(itertools.product(np.arange(0,3,1), np.arange(0,3,1))))[:-1]
    
    tops_with_params = np.zeros((len(data)*n_samples, 4))
    for index_top, top in enumerate(data): #39, repeat 39,50*8*25
        tops_with_params[index_top*n_samples:index_top*n_samples + n_samples,:] = parameter_search(top, N, n_samples, 123)
    
    data_all = np.repeat(data, n_samples*len(list_dep_combinations)*n_ic**2, axis = 0)
    top_with_params_all = np.repeat(tops_with_params, len(list_dep_combinations)*n_ic**2, axis = 0)
    list_dep_combinations_all = np.tile(np.repeat(list_dep_combinations, n_ic**2, axis = 0), (len(data)*n_samples, 1))
    ic_all = np.tile(ic_combinations, (len(data)*n_samples*len(list_dep_combinations),1))
    
    # Define the size of the array
    num_rows = n_samples*n_ic**2*len(list_dep_combinations)*len(data)
    chunk_size = n_samples*n_ic*8  # Define the size of each chunk
    
    # Initialize the array
    array_results = np.zeros((num_rows, num_features))
    return (data_all, top_with_params_all, list_dep_combinations_all, ic_all, list_dep, num_rows, chunk_size, array_results)

# Define the function that processes and fills a chunk
def process_chunk(start_index, end_index, data_all, top_with_params_all, list_dep_combinations_all, ic_all, list_dep, num_features):
    # Example: Fill the chunk with some computation
    # Replace this with your actual computation logic
    chunk = np.zeros((end_index - start_index, num_features))
    for i in range(start_index, end_index):
        V, V_th, gene_conc_spike, gene_conc_base, gene_conc_dev, spike = coupled_system(N, top_with_params_all[i], list_dep[list_dep_combinations_all[i]], ic_all[i])
        chunk[i - start_index, :] = feature_extraction(spike, gene_conc_spike, gene_conc_base)
        
    return start_index, end_index, chunk

def main(N, n_samples, n_ic, network_space_file_name, result_file_name, max_worker_number):
    num_features = 4
    data_all, top_with_params_all, list_dep_combinations_all, ic_all, list_dep, num_rows, chunk_size, array_results = create_arrays(N, n_samples, n_ic, network_space_file_name, num_features)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_worker_number) as executor:
        # Prepare chunk indices
        futures = []
        for start_index in range(0, num_rows, chunk_size):
            end_index = min(start_index + chunk_size, num_rows)
            futures.append(executor.submit(process_chunk, start_index, end_index, data_all, top_with_params_all, list_dep_combinations_all, ic_all, list_dep, num_features))
        
        # As each future completes, fill the respective chunk in the array
        for future in concurrent.futures.as_completed(futures):
            start_index, end_index, chunk = future.result()
            array_results[start_index:end_index, :] = chunk

    array_features_all = np.zeros((num_rows, 12+num_features))
    array_features_all[:,0:4] = data_all
    array_features_all[:,4:8] = top_with_params_all
    array_features_all[:,8:10] = list_dep_combinations_all
    array_features_all[:,10:12] = ic_all
    array_features_all[:,12:] = array_results
    np.save(result_file_name, array_features_all)
    return array_features_all

# In[ ]:

def save_df(file_name, df):
    '''
    Function for saving a dataframe
    '''
    # open a file, where you ant to store the data
    file = open(file_name, 'wb')
    # dump information to that file
    pickle.dump(df, file)
    # close the file
    file.close()

def load_df(file_name):
    '''
    Function for loading a dataframe
    '''
    # open a file, where you stored the pickled data
    file = open(file_name, 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    return data


# In[ ]:

