
import os
from itertools import product

abspath = os.path.abspath('run_model.py')
dname = os.path.dirname(abspath)
os.chdir(dname)

from decifit.decifit import *

############
### DATA ###
############

#GENERAL SETTINGS
inst = 0  #instance
instance = decifit(inst)

instance.two_factor = False # Are the price factors explicitly included as state variables

#MC
instance.markov = True
n_markov_states = 10     #  15 or None or so
markov_method = 'SAA'  # Pick from:
# ‘SAA’: use k-means to train Markov chain.
# ‘SA’: use stochastic approximation to train Markov chain.
# ‘RSA’: use robust stochastic approximation to train Markov chain. 

#TS
num_samples = 5 # of the continuous uncertainties TS Approach
precision = 3  # of the binarization

if not instance.two_factor and not instance.markov: # solving deterministic model
    instance.percentile = True
    instance.extensive = True  
    instance.percentile_level = 50  # between 0 and 100  

max_iter_ip = 200
max_time_ip = 1*60*60    #in seconds
instance.cuts = ['B','LG','SB']  
instance.percentile_level = 50 # if solving deterministic model, between 0 and 100  

extract_general_results = True
extract_average_shutdown_results = True
instance.print_decision_matrix = True

n_simulations = 5000   # TO EVALUATE THE POLICIES


############
### RUN  ###
############

instance.data.generate_derived_data()
instance.construct_model(precision, num_samples, n_markov_states, markov_method, relaxation=False)
instance.solve_model(max_iter_ip, max_time_ip, relaxation=False)
instance.evaluate_policy(n_simulations) #evaluates both true and normal

if extract_general_results:
    instance.print_results()
if extract_average_shutdown_results:
    instance.get_average_shutdown_stage()
       



