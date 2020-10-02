# Libraries
import sys
import os.path
import numpy as np
from msppy.msp import MSIP,MSLP
from msppy.solver import Extensive, SDDiP,SDDP
from msppy.evaluation import Evaluation, EvaluationTrue
import gurobipy
import pandas as pd
import numpy
import csv
from scipy.stats import norm

from decifit.utils.ReadData import Data


class decifit:

    def __init__(self, instance):

        self.instance = instance
        self.data = Data(instance)

        self.Evaluation = None

        self.precision = 2
        self.max_iter = 10
        self.num_samples = 6        
        self.n_markov_states = 10
        self.sample_seed = 5

        self.markov = True
        self.two_factor = False

        self.percentile = False
        self.percentile_level = 50
        self.extensive = False

        self.freq_comp = 3

        self.cuts = ['LG', 'B', 'SB']
        self.print_decision_matrix = False


    def sample_path_generator(self,random_state, size):

        output = np.empty([size,self.data.dat['num_stages'],4])  #T:time horizon
        output[:,0,0] = self.data.dat['xi_0']   #equilibrium factor
        output[:,0,1] = self.data.dat['chi_0']   #deviation factor
        output[:,0,2] = np.exp(output[:,0,0]+ output[:,0,1]) #prices
        output[:,0,3] = output[:,0,2]  # discounted prices
        for t in range(1,self.data.dat['num_stages']):
            standard_normal_values = random_state.multivariate_normal(
                mean=[0, 0], cov=[[self.data.dat['var_std_normal'], self.data.dat['covariance']],
                                  [self.data.dat['covariance'], self.data.dat['var_std_normal']]], size=size)
            output[:,t,0] = self.data.dat['alpha_xi'][self.data.dat['num_years_per_stage'][t]-1] + \
                             output[:,t-1,0] + \
                             standard_normal_values[:,0]*self.data.dat['eps_xi_par'][self.data.dat['num_years_per_stage'][t]-1]
            output[:,t,1] = self.data.dat['alpha_chi'][self.data.dat['num_years_per_stage'][t]-1] + \
                             output[:,t-1,1]*self.data.dat['beta_chi'][self.data.dat['num_years_per_stage'][t]-1] + \
                             standard_normal_values[:,1]*self.data.dat['eps_chi_par'][self.data.dat['num_years_per_stage'][t]-1]
            output[:,t,2] = np.minimum(np.exp(output[:,t,0]+ output[:,t,1]),
                                       np.repeat(self.data.dat['P_UPPER_BOUND'],size))   #PRICE
            output[:, t, 3] = output[:,t,2]*self.data.dat['discount_factor'][t]    #DISCOUNTED PRICE
        return output

    
    def construct_model(self, precision, num_samples, n_markov_states, markov_method, relaxation):

        self.n_markov_states = n_markov_states
        self.num_samples = num_samples

        if self.two_factor:   # for the time series approach
            self.data.dat['query'].append('dev_fact')
            self.data.dat['query'].append('eq_fact')
            self.read_plf(os.path.join(os.getcwd(),r'decifit/utils/plf_data.csv'))
        else: 
            self.data.dat['prices'] = self.sample_path_generator(random_state=np.random.RandomState(6), size=1000)

        if relaxation:   #this constructs the lp relaxation of the problem
            self.model_lp = MSLP(T=self.data.dat['num_stages'], sense=-1)
            model = self.model_lp
        elif not relaxation: #constructs the integer version
            self.model_ip = MSIP(T=self.data.dat['num_stages'], sense=-1) #, bound=self.data.dat['OBJ_BOUND'], should be a uniform bound on the optimum in each stage optimization problem
            model = self.model_ip
        # sense=-1: maximization problem, sense = 1: minimization, discount = 0.995

        if self.markov:
            model.add_Markovian_uncertainty(self.sample_path_generator)

        for t in range(self.data.dat['num_stages']):   

            m = model.models[t]

            # ----------------------------------- #
            # VARIABLES
            # ----------------------------------- #

            # STATE VARIABLES:
            if not relaxation:
               variable_type = 'B'
            elif relaxation:
                variable_type = gurobipy.GRB.CONTINUOUS

            x_activities_now, x_activities_past = m.addStateVars(
                [(tt,a) for a in self.data.dat['A_ACTIVITIES'] for tt in range(0,self.data.dat['last_stage_to_start_activity'][a])],
                name='x_act', lb=0, ub=1, vtype=variable_type)
            y_shutdown_now, y_shutdown_past = m.addStateVar(name='y_sd', lb=0, ub=1, vtype=variable_type)

            #For the Time Series approach, the factors also become state variables
            if not self.markov and not self.percentile:  # TS APPROACH
                equilibrium_factor_now, equilibrium_factor_past = m.addStateVar(
                    lb=self.data.dat['lower_bound_eq_factor'], ub=self.data.dat['upper_bound_eq_factor'],
                    name='eq_fact', vtype=gurobipy.GRB.CONTINUOUS) #xi
                deviation_factor_now, deviation_factor_past = m.addStateVar(
                    lb=self.data.dat['lower_bound_dev_factor'], ub=self.data.dat['upper_bound_dev_factor'],
                    name='dev_fact', vtype=gurobipy.GRB.CONTINUOUS) #chi

            # CONTROL VARIABLES:
            if not self.two_factor:
                cost = m.addVar(
                    vtype=gurobipy.GRB.CONTINUOUS, name='cost', lb=0, ub=self.data.dat['MAX_COST'],
                    obj=-self.data.dat['discount_factor'][t])
                if self.markov:
                    if self.n_markov_states == 1:
                        q_total_production = m.addVar(  
                        lb=self.data.dat['Q_MIN'], ub=self.data.dat['Q_MAX'], vtype=gurobipy.GRB.CONTINUOUS,
                        name='q_total_production',
                        obj=numpy.percentile(self.data.dat['prices'][:,t,2], self.percentile_level) *
                        self.data.dat['discount_factor'][t]) 
                    else:
                        q_total_production = m.addVar(
                            lb=self.data.dat['Q_MIN'], ub=self.data.dat['Q_MAX'], vtype=gurobipy.GRB.CONTINUOUS,
                            name='q_total_production', uncertainty_dependent=3)  #put the (price)uncertainty in the objective coefficient
                else:
                    q_total_production = m.addVar(  
                        lb=self.data.dat['Q_MIN'], ub=self.data.dat['Q_MAX'], vtype=gurobipy.GRB.CONTINUOUS,
                        name='q_total_production',
                        obj=numpy.percentile(self.data.dat['prices'][:,t,2], self.percentile_level) *
                        self.data.dat['discount_factor'][t]) 

            else:
                q_total_production = m.addVar(
                    lb=self.data.dat['Q_MIN'], ub=self.data.dat['Q_MAX'], vtype=gurobipy.GRB.CONTINUOUS, name='q_total_production')
                profit = m.addVar(
                    obj=self.data.dat['discount_factor'][t], vtype=gurobipy.GRB.CONTINUOUS,
                    name='profit', lb=-self.data.dat['MAX_COST'], ub=self.data.dat['MAX_REV'])
                revenue = m.addVar(
                    vtype=gurobipy.GRB.CONTINUOUS, name='revenue', lb=0, ub=self.data.dat['MAX_REV']) #CAN BE NEG
                cost = m.addVar(
                    vtype=gurobipy.GRB.CONTINUOUS, name='cost', lb=0, ub=self.data.dat['MAX_COST'])


            price = m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='price', lb=-20, ub=self.data.dat['P_UPPER_BOUND'])
            if self.two_factor:
                lambda_sos = m.addVars(
                    self.data.dat['num_breakpoints'], lb=0, ub=1, vtype=gurobipy.GRB.CONTINUOUS, name='lambda_sos')
                # McCormick variables, bounds are defined here. Could be based on time t.
                z_mcc = m.addVars([(tt,a) for tt in range(t+1) for a in self.data.dat['A_ACTIVITIES'] if tt <=
                                   self.data.dat['last_stage_to_start_activity'][a]],
                                  lb=0, ub=self.data.dat['P_UPPER_BOUND'], vtype=gurobipy.GRB.CONTINUOUS, name='z_mcc')



            # ----------------------------------- #
            # CONSTRAINTS
            # ----------------------------------- #

            if self.two_factor:
                # OBJECTIVE FUNCTION Contribution
                m.addConstr(profit == revenue - cost)

            # PRODUCTION PROFILES
            m.addConstr(q_total_production <= self.data.dat['Q_BASE'][t] + gurobipy.quicksum(
                self.data.dat['Q_ADD'][(a,tau)][(t - tau)] * x_activities_now[tau, a] for a in self.data.dat['A_ACTIVITIES']
                for tau in range(t+1-self.data.dat['lagged_investment']) if tau <= self.data.dat['last_stage_to_start_activity'][a]-1))
            
            m.addConstr(q_total_production <= self.data.dat['Q_MAX'] *
                        (1 - y_shutdown_now*(1-self.data.dat['lagged_investment'])
                         - y_shutdown_past*self.data.dat['lagged_investment']))
            # Capacities: see variable definition

            #ACTIVITIES
            # Start an activity at most once per time period
            m.addConstrs(gurobipy.quicksum(x_activities_now[tt, a]
                                           for tt in range(self.data.dat['last_stage_to_start_activity'][a])) <= 1
                         for a in self.data.dat['A_ACTIVITIES'])
            # Maximum amount of activities that can be started each year:
            m.addConstr(gurobipy.quicksum(x_activities_now[t, a]
                                          for a in self.data.dat['A_ACTIVITIES']
                                          if t <= self.data.dat['last_stage_to_start_activity'][a]-1) <= 1)

            # FIXING STATE VARIABLES FOR ALL PERIODS EXCEPT NOW
            m.addConstrs(x_activities_now[tau, a] == x_activities_past[tau, a]
                         for tau in self.data.dat['T_STAGES'] if tau != t
                         for a in self.data.dat['A_ACTIVITIES'] if tau<=self.data.dat['last_stage_to_start_activity'][a]-1)
            m.addConstr(y_shutdown_now >= y_shutdown_past)
            if t==0:
                m.addConstr(y_shutdown_past == 0)
                m.addConstrs(x_activities_now[tau, a] == 0 for a in self.data.dat['A_ACTIVITIES']
                             for tau in range(self.data.dat['last_stage_to_start_activity'][a])
                             if tau > 0)
                m.addConstrs(x_activities_past[tau, a] == 0 for a in self.data.dat['A_ACTIVITIES']
                             for tau in range(self.data.dat['last_stage_to_start_activity'][a]))
            # COSTS
            m.addConstr(
                cost >= self.data.dat['C_OPEX'][t] * (1 - y_shutdown_now*(1-self.data.dat['lagged_investment'])
                                                      - y_shutdown_past*self.data.dat['lagged_investment']) +
                gurobipy.quicksum(self.data.dat['C_ACT'][a] * x_activities_now[t, a] for a in self.data.dat['A_ACTIVITIES']
                                  if t <= self.data.dat['last_stage_to_start_activity'][a]-1) +
                self.data.dat['C_DECOM'][t] * (y_shutdown_now-y_shutdown_past))  #only one if we have the change

            # DECOMMISSIONING
            if t == (self.data.dat['num_stages'] - 1):  # that is the last time period
                m.addConstr(y_shutdown_now == 1)


            # REVENUE & PRICES

            if not self.two_factor:
                if t == 0:
                    m.addConstr(price == np.exp(self.data.dat['xi_0']+self.data.dat['chi_0']))
                else:
                    if self.markov:
                        m.addConstr(price == 0, uncertainty_dependent={'rhs':2}) #just to keep track of it. might remove
                    else:
                        m.addConstr(price == numpy.percentile(self.data.dat['prices'][:,t,2], self.percentile_level)) #deterministic

            else: #i.e. two_factor

                if not self.markov: #TIME SERIES APPROACH

                    if t == 0:
                        # deterministic contraints
                        # define equilibrium and deviation factors previous period! (INITIALIZATION)
                        m.addConstr(equilibrium_factor_now <= self.data.dat['xi_0'])
                        m.addConstr(deviation_factor_now <= self.data.dat['chi_0'])
                    else:
                        error_term_eq = m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='error_term_eq', lb=-4,ub=4)
                        error_term_dev = m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name='error_term_dev', lb=-4,ub=4)
                        uncertainty_realization_eq = m.addConstr(error_term_eq == 0)
                        uncertainty_realization_dev = m.addConstr(error_term_dev == 0)
                        def f(random_state):
                            standard_normal_values = random_state.multivariate_normal(
                                mean=[0, 0], cov=[[self.data.dat['var_std_normal'], self.data.dat['covariance']],
                                                  [self.data.dat['covariance'], self.data.dat['var_std_normal']]])
                            scaled_values = [standard_normal_values[0]*self.data.dat['eps_xi_par'][self.data.dat['num_years_per_stage'][t]-1],
                                             standard_normal_values[1]*self.data.dat['eps_chi_par'][self.data.dat['num_years_per_stage'][t]-1]]
                            return_values = [0,0]
                            for i in range(2): #preventing issues with bounds / extreme outcomes
                                if scaled_values[i] < self.data.dat['error_low'][i]:
                                    return_values[i] = self.data.dat['error_low'][i]
                                elif scaled_values[i] > self.data.dat['error_low'][i]:
                                    return_values[i] = -self.data.dat['error_low'][i]
                                else:
                                    return_values[i] = scaled_values[i]
                            return [round(i,precision) for i in scaled_values]

                        m.add_continuous_uncertainty(uncertainty=f, locations=[uncertainty_realization_eq,uncertainty_realization_dev])

                        epsilon = 1/numpy.power(10,precision+1) #for the rounding!
                        m.addConstr(equilibrium_factor_now - equilibrium_factor_past -
                                    self.data.dat['alpha_xi'][self.data.dat['num_years_per_stage'][t]-1] <= error_term_eq+50*epsilon)
                        m.addConstr(deviation_factor_now - deviation_factor_past *
                                    self.data.dat['beta_chi'][self.data.dat['num_years_per_stage'][t]-1] -
                                    self.data.dat['alpha_chi'][self.data.dat['num_years_per_stage'][t]-1] <= error_term_dev+50*epsilon)

                        m.addConstr(equilibrium_factor_now - equilibrium_factor_past -
                                    self.data.dat['alpha_xi'][self.data.dat['num_years_per_stage'][t]-1] >= error_term_eq-49*epsilon)
                        m.addConstr(deviation_factor_now - deviation_factor_past *
                                    self.data.dat['beta_chi'][self.data.dat['num_years_per_stage'][t]-1] -
                                    self.data.dat['alpha_chi'][self.data.dat['num_years_per_stage'][t]-1] >= error_term_dev-49*epsilon)

                else: #two-factor markov chain approach, not necessary...

                    equilibrium_factor_now = m.addVar(vtype=gurobipy.GRB.CONTINUOUS,name='eq_fact',
                                                      lb=self.data.dat['lower_bound_eq_factor'],
                                                      ub=self.data.dat['upper_bound_eq_factor'])
                    deviation_factor_now = m.addVar(vtype=gurobipy.GRB.CONTINUOUS,name='dev_fact',
                                                    lb=self.data.dat['lower_bound_dev_factor'],
                                                    ub=-self.data.dat['lower_bound_dev_factor'])

                    m.addConstr(equilibrium_factor_now == 0, uncertainty_dependent={'rhs':0})
                    m.addConstr(deviation_factor_now == 0, uncertainty_dependent={'rhs':1})

                #REVENUE & PRICES & LINEARIZATION 
                
                # McCormick
                m.addConstrs(price - self.data.dat['P_UPPER_BOUND'] * (1 - x_activities_now[tau, a]) <= z_mcc[(tau, a)]
                             for a in self.data.dat['A_ACTIVITIES']
                             for tau in range(t+1-self.data.dat['lagged_investment'])
                             if tau <= self.data.dat['last_stage_to_start_activity'][a]-1)
                m.addConstrs(z_mcc[(tau, a)] <= self.data.dat['P_UPPER_BOUND'] * x_activities_now[(tau, a)]
                             for a in self.data.dat['A_ACTIVITIES']
                             for tau in range(t+1-self.data.dat['lagged_investment'])
                             if tau <= self.data.dat['last_stage_to_start_activity'][a]-1)
                m.addConstrs(z_mcc[(tau, a)] <= price for a in self.data.dat['A_ACTIVITIES']
                             for tau in range(t+1-self.data.dat['lagged_investment'])
                             if tau <= self.data.dat['last_stage_to_start_activity'][a]-1)

                m.addConstr(revenue <= self.data.dat['Q_BASE'][t] * price + gurobipy.quicksum(
                self.data.dat['Q_ADD'][(a,tau)][(t - tau)] * z_mcc[(tau,a)]    #z_mcc is x^act multiplied by price
                for a in self.data.dat['A_ACTIVITIES'] for tau in range(t+1-self.data.dat['lagged_investment'])
                if tau <= self.data.dat['last_stage_to_start_activity'][a]-1))
                m.addConstr(revenue <= self.data.dat['MAX_REV'] *
                            (1 - y_shutdown_now*(1-self.data.dat['lagged_investment'])
                             - y_shutdown_past*self.data.dat['lagged_investment']))

                # SOS2 linearization https://www.gurobi.com/documentation/8.1/refman/constraints.html#subsubsection:SOSConstraints
                if not relaxation:
                    m.addSOS(gurobipy.GRB.SOS_TYPE2, [lambda_sos[i] for i in range(self.data.dat['num_breakpoints'])])

                m.addConstr(equilibrium_factor_now + deviation_factor_now >=
                            gurobipy.quicksum(self.data.dat['breakpoints_factor'][i] * lambda_sos[i]
                                              for i in range(self.data.dat['num_breakpoints'])))
                m.addConstr(price <= gurobipy.quicksum(self.data.dat['breakpoints_price'][i] * lambda_sos[i]
                                                       for i in range(self.data.dat['num_breakpoints'])))
                m.addConstr(gurobipy.quicksum(lambda_sos[i] for i in range(self.data.dat['num_breakpoints'])) == 1)

        #####################################
        ## discretization or binarization ##
        #####################################
        if self.markov:
            model.sample_seed = self.sample_seed # seed for the sample to train Markov Chain
            model.discretize(
                n_Markov_states=n_markov_states,
                n_sample_paths=5000, #50000
                random_state=numpy.random.RandomState([3,3]),  #this  just affects the centroid starting point for SAA
                method=markov_method
            )
        else: #TIME SERIES APPROACH
            if self.two_factor:
                model.discretize(n_samples=num_samples, random_state=888)  #this is the seed and number of samples
                if not relaxation:
                    # Binarize everything before bin_stage, standard is zero
                    self.model_ip.binarize(bin_stage=self.data.dat['num_stages'], precision=precision)

    def solve_extensive_model(self,max_time):

        self.model_extensive_solved = Extensive(self.model_ip)
        self.model_extensive_solved.solve(outputFlag=0)

    def solve_model(self, max_iter, max_time, relaxation):

        if relaxation:
            model = self.model_lp
        else:
            model = self.model_ip

        # solving the model
        if relaxation:
            self.model_lp_solved = SDDP(self.model_lp)
            self.model_lp_solved.solve(
                n_processes = 1,
                max_iterations=max_iter,
                max_stable_iterations = 20,  # The maximum number of iterations to have the same deterministic bound
                max_time=max_time, # hours*minutes*seconds (in seconds)
            )
            self.data.dat['OBJ_BOUND'] = self.model_lp_solved.db[-1]
        elif not relaxation:            
            self.model_ip_solved = SDDiP(self.model_ip)
            self.model_ip_solved.solve(
                n_processes = 1, n_steps = 1, max_iterations=max_iter,
                max_stable_iterations = 11,  # The maximum number of iterations to have the same deterministic bound
                max_time=max_time, # hours*minutes*seconds (in seconds)
                n_simulations=5000, #500,    -1 calculates the epv
                #freq_evaluations=5, tol = 1e-3, #optimality gap
                tol_diff=1e-4,
                freq_comparisons=self.freq_comp, #Turn to 1 the obtain confidence interval on policy value
                query_policy_value = False, # turn this on to get ALL the output for the graph (CI)
                cuts=self.cuts, #['LG','B','SB']
            )
            self.data.evaluation['first_stage_solution'] = self.model_ip_solved.first_stage_solution
            self.data.evaluation['bounds'] = self.model_ip_solved.bounds
            self.data.evaluation['pv'] = self.model_ip_solved.pv
            self.data.evaluation['db'] = self.model_ip_solved.db
            self.data.evaluation['total_time'] = self.model_ip_solved.total_time
            self.data.evaluation['iteration'] = self.model_ip_solved.iteration
            self.get_first_stage_solution()

            for key, value in self.model_ip_solved.first_stage_solution.items():
                if value > 0.99 and value < 1.01:
                    print(key)


    # We have to set bounds for the TS approach. If done wrong, this leads to problems with complete recourse.
    def set_bounds(self,num_samples):

        T = self.data.dat['num_stages']
        sample_paths = self.model_lp._enumerate_sample_paths(T-1)[1] #returns (n_sample_paths,sample paths)
        num_scenarios = len(sample_paths) #T = 6, 6 discretizations -> 7776 scenarios..

        xi_error_realization = {(t,i):0 for t in range(1,T) for i in range(num_samples)}
        chi_error_realization = {(t,i):0 for t in range(1,T) for i in range(num_samples)}
        for t in range(1,T):
            key_xi = list(self.model_lp.models[t].uncertainty_rhs.keys())[0]  # equilibrium
            key_chi = list(self.model_lp.models[t].uncertainty_rhs.keys())[1]  # dev
            for i in range(num_samples):
                xi_error_realization[(t,i)] = self.model_lp.models[t].uncertainty_rhs[key_xi][i]
                chi_error_realization[(t,i)] = self.model_lp.models[t].uncertainty_rhs[key_chi][i]
        scenarios_prices = {(t,i):0 for t in range(T) for i in range(num_scenarios)}
        scenarios_xi = {(t,i):0 for t in range(T) for i in range(num_scenarios)}
        scenarios_chi = {(t,i):0 for t in range(T) for i in range(num_scenarios)}
        for i in range(num_scenarios):
            for t in range(T):
                if t == 0:
                    scenarios_xi[(t,i)] = round(self.data.dat['xi_0'],2)
                    scenarios_chi[(t,i)] = round(self.data.dat['chi_0'],2)
                else:
                    sample = sample_paths[i][t]
                    scenarios_xi[(t,i)] = round(scenarios_xi[(t-1,i)] +
                                                self.data.dat['alpha_xi'][self.data.dat['num_years_per_stage'][t]-1] +
                                                xi_error_realization[(t,sample)],2)
                    scenarios_chi[(t,i)] = round(scenarios_chi[(t-1,i)]*(
                        self.data.dat['beta_chi'][self.data.dat['num_years_per_stage'][t]-1]) +
                                                 self.data.dat['alpha_chi'][self.data.dat['num_years_per_stage'][t]-1]
                                                 + chi_error_realization[(t,sample)],2)
                scenarios_prices[(t,i)] = numpy.exp(scenarios_xi[(t,i)]+scenarios_chi[(t,i)])

        self.data.dat['lower_bound_eq_factor'] = round(min(scenarios_xi.values())-1,2)
        self.data.dat['upper_bound_eq_factor'] = round(max(scenarios_xi.values())+1,2)
        self.data.dat['lower_bound_dev_factor'] = round(min(scenarios_chi.values())-1,2)
        self.data.dat['upper_bound_dev_factor'] = round(max(scenarios_chi.values())+1,2)


    def get_first_stage_solution(self):

        self.data.evaluation['first_stage_decision'] = []
        if self.markov:
            m=self.model_ip.models[0][0]         # first stage problem.
        else:
            m=self.model_ip.models[0]
        states_ = m.states   #all variables    controls/states
        for i in range(len(states_)):
            if states_[i].x ==1:
                if 'x_act' in states_[i].varName or 'y_sd' in states_[i].varName:
                    activity = states_[i].varName
                    self.data.evaluation['first_stage_decision'].append(activity)


    def print_results(self):

        if os.path.exists('results.csv'):
            fff = open('results.csv','a') # append if already exists
        else:
            fff = open('results.csv','w') # make a new file if not
            print('Instance;    Two_Factor;   Markov; Num_Markov_States;sample_seed ;Num_Samples; Percentile_Level; '
                  ' Cuts; First_Stage_Dec;    Num_Its;    Time;   '
                  ' UB_eval; CI_eval_L;CI_eval_U   ;Gap_eval;'
                  'UB_eval_true;    CI_eval_true_L; CI_eval_true_U;  Gap_eval_true; ',
                  file = fff)
        if len(self.data.evaluation['first_stage_decision'])>0:
            fsd = self.data.evaluation['first_stage_decision'][0]
        else:
            fsd = 'do_nothing'
        print(self.instance,self.two_factor,self.markov,self.n_markov_states,self.sample_seed,self.num_samples,self.percentile_level,
              self.cuts,fsd,
              self.model_ip_solved.iteration, self.model_ip_solved.total_time,
              self.data.evaluation['db_eval'], self.data.evaluation['CI_eval'][0],self.data.evaluation['CI_eval'][1],
              self.data.evaluation['gap_eval'], sep=';', file=fff, end = '')
        if self.markov:
            print(end=';', file=fff)
            print( self.data.evaluation['db_eval_true'], self.data.evaluation['CI_eval_true'][0],
                   self.data.evaluation['CI_eval_true'][1],self.data.evaluation['gap_eval_true'], sep=';',file=fff, end = '')
        print('',file=fff,)
        fff.close()

    def get_average_shutdown_stage(self):

        filename = 'average_shutdown_stage.csv'
        if os.path.exists(filename):
            ff = open(filename,'a') # append if already exists
        else:
            ff = open(filename,'w') # make a new file if not
            print('Instance; Discount_Rate;  Plugging_Cost; Increase_Plugging_cost; OPEX ;'
                  'Expected_ShutDown_Stage; Expected_ShutDown_Year; Expected_ShutDown_Year_Effective; gap; bound; '
                  'sigma_chi; sigma_xi', file=ff)

        count = 0
        count2 = 0
        count3 = 0
        for i in range(len(self.data.evaluation['decision_matrix_true'])):
            for j in range(6):
                if self.data.evaluation['decision_matrix_true'].iloc[i,j]=='y_sd':
                    count += j
                    count2 += sum(self.data.dat['num_years_per_stage'][0:j])
                    count3 += sum(self.data.dat['num_years_per_stage'][0:j+1])
        expected_time = count/len(self.data.evaluation['decision_matrix_true'])
        expected_year = count2/len(self.data.evaluation['decision_matrix_true'])
        expected_year_eff = count3/len(self.data.evaluation['decision_matrix_true'])
        print(self.instance, self.data.dat['r_riskfree'], self.data.dat['DECOM_COST_FIXED'],
              self.data.dat['DECOM_COST_INCREASE'], self.data.dat['OPEX'], expected_time, expected_year, expected_year_eff,
              self.data.evaluation['gap_eval'],self.data.evaluation['db_eval'],
              self.data.dat['sigma_chi'],self.data.dat['sigma_xi'],
              sep=';', file=ff)

        ff.close()


    def evaluate_policy(self,n_simulations):

        evaluate = [False]
        # When evaluating the obtained policy from TS approach we sometimes get computational issues
        if self.markov:             
            evaluate.append(True) 
        
        for true_distribution in evaluate:   #[True,False]
            if true_distribution:
                result = EvaluationTrue(self.model_ip)
                string = '_true'
            else:
                result = Evaluation(self.model_ip)
                string = ''
            result.run(n_simulations=n_simulations, query_stage_cost=True, query=self.data.dat['query'])

            #I could add this to the dataframe
            self.data.evaluation['db_eval'+string] = result.db
            self.data.evaluation['CI_eval'+string] = result.CI
            self.data.evaluation['gap_eval'+string] = result.gap
            self.data.evaluation['pv2_eval'+string] = result.pv
            self.data.evaluation['epv_eval'+string] = result.epv

            decision_matrix = pd.DataFrame(index=list(range(n_simulations)), columns=self.data.dat['T_STAGES'])
            decision_matrix = decision_matrix.fillna('-')
            for i in range(n_simulations):
                for (t,a) in self.data.dat['x_act_combinations']:
                    if result.solution["x_act[{},{}]".format(t, a)].at[i,t]==1:
                        decision_matrix.at[i,t] = a
                operating = True
                for t in self.data.dat['T_STAGES']:
                    if result.solution['y_sd'].at[i,t]==1 and operating:
                        decision_matrix.at[i,t] = 'y_sd'
                        operating = False

            self.data.evaluation['decision_matrix'+string] = decision_matrix
            self.data.evaluation['prices'+string] = result.solution['price']
            if self.two_factor:
                self.data.evaluation['eq_fact'+string] = result.solution['eq_fact']
                self.data.evaluation['dev_fact'+string] = result.solution['dev_fact']


            #######################
            ### DECISION MATRIX ###
            #######################
        if self.print_decision_matrix:

            if self.markov: 
                decision_matrix = self.data.evaluation['decision_matrix_true']
                prices = self.data.evaluation['prices_true']
            else:
                decision_matrix = self.data.evaluation['decision_matrix']
                prices = self.data.evaluation['prices']
            

            summary_decision_matrix = decision_matrix.groupby(decision_matrix.columns.tolist(),as_index=False).size()

            unique_solutions = decision_matrix.drop_duplicates()
            unique_solutions = unique_solutions.reset_index(drop=True)
            rows_to_instances = {i:[] for i in range(len(unique_solutions))}  #rows are the unique solutions
            for index, row in decision_matrix.iterrows():
                for index2,row2 in unique_solutions.iterrows():
                    if list(row) == list(row2):
                        rows_to_instances[index2].append(index)
            unique_solutions['count'] = 0

            rows_dec_mat = list(summary_decision_matrix.index)
            for i in range(len(rows_dec_mat)):
                for j in range(len(unique_solutions)):
                    if list(summary_decision_matrix.iloc[i,0:6]) == list(unique_solutions.iloc[j,0:6]):
                        unique_solutions['count'][j] = summary_decision_matrix['size'][i]

            ordering = {'new_wells8':0,'new_wells4':1,'sidetrack2':2,'sidetrack1':3,'-':4,'y_sd':5}
            unique_solutions['order'] = 0
            for i in [5,4,3,2,1,0]:
                unique_solutions['order'] = [ordering[j] for j in unique_solutions[i]]
                unique_solutions = unique_solutions.sort_values(by='order',ascending=True)

            order = list(unique_solutions.index)
            def manual_sorting(col):
                output = []
                for i in col:
                    for j in range(len(order)):
                        if i==order[j]:
                            output.append(j)
                return output

            vars = ['prices']
            statistics = [i+'_'+j for i in vars  for j in ['mean','min','max']] 
            stat_df = {s: pd.DataFrame(index=range(len(unique_solutions))) for s in statistics}

            num_stages = len(decision_matrix.columns)
            for t in range(num_stages): # initialize
                for stat in statistics:
                    stat_df[stat][str(t)] = 0.0
            
            for i in range(len(unique_solutions)):
                subset = prices.iloc[rows_to_instances[i],:]
                for t in range(num_stages):
                    var = 'prices'
                    stat_df[var+'_min'][str(t)][i] = subset.iloc[:,t].min()
                    stat_df[var+'_max'][str(t)][i] = subset.iloc[:,t].max()
                    stat_df[var + '_mean'][str(t)][i] = subset.iloc[:, t].mean()

            for stat in statistics:
                print(stat)
                stat_df[stat]['row_number'] = list(stat_df[stat].index)
                print((stat_df[stat].sort_values(by='row_number', key=manual_sorting)))

    # This takes some time. 
    def generate_plf(self, num_segments, n_samples, mip_gap, time_limit):  # generate piecewise linear function

        self.two_factor_copy = self.two_factor
        self.two_factor = False

        df = self.sample_path_generator(numpy.random, n_samples)
        # y = [item for sublist in df.values.tolist() for item in sublist]
        y = [df[i][t][0] for i in range(len(df)) for t in range(len(df[0]))]
        y.sort()
        x = numpy.log(y)
        plf = PLFG(x, y, num_segments)
        plf.pflg_model(mip_gap, time_limit)
        plf.plot()

        self.data.dat['MAE'] = plf.MAE
        self.data.dat['num_segments'] = num_segments
        self.data.dat['num_breakpoints'] = num_segments + 1
        self.data.dat['breakpoints_price'] = [round(price, 2) for price in plf.f_b]
        self.data.dat['breakpoints_factor'] = [round(b, 2) for b in plf.b]
        self.data.dat['slopes'] = plf.c
        self.data.dat['intercepts'] = plf.d

        self.two_factor = self.two_factor_copy

        return y

    def write_plf(self, filename):

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([self.data.dat['num_segments']])
            writer.writerow(self.data.dat['breakpoints_price'])
            writer.writerow(self.data.dat['breakpoints_factor'])

    def read_plf(self, filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
            for i, line in enumerate(reader):
                if i == 0:
                    self.data.dat['num_segments'] = int(line[0])
                    self.data.dat['num_breakpoints'] = int(line[0] + 1)
                elif i == 1:
                    self.data.dat['breakpoints_price'] = line
                elif i == 2:
                    self.data.dat['breakpoints_factor'] = line

        
