# libraries
import numpy
from scipy.stats import norm
import pandas

import os


class Data:

    def __init__(self, instance):

        self.instance = instance
        self.dat = {}
        self.dat.update(updated_variables=[])
        self.evaluation = {}

        ###################
        #### BASE DATA ####
        ###################

        self.dat.update(lagged_investment=1)  # number of stages lag

        self.dat.update(num_stages=6)  # T = num_stages - 1
        self.dat.update(
            num_years_per_stage=[1, 1, 1, 2, 2, 3])  # total of ten years, 
        self.dat.update(
            r_riskfree=0.02)  

        self.dat['A_ACTIVITIES'] = []
        self.dat['Q_ADD_YEARLY'] = {}
        self.dat['last_stage_to_start_activity'] = {}  
        self.dat['C_ACT'] = {}  # Million dollars

        excel_data_df = pandas.read_excel(os.path.join(os.getcwd(),r'decifit/utils/data.xlsx'), sheet_name='Main')
        for index, row in excel_data_df.iterrows():
            if row['activity']=='q_base_yearly':
                self.dat['Q_BASE_YEARLY'] = list(excel_data_df.iloc[0,1:11])
            else:
                self.dat['A_ACTIVITIES'].append(row['activity'])
                self.dat['Q_ADD_YEARLY'].update({row['activity']:list(row[1:11])})
                self.dat['last_stage_to_start_activity'].update({row['activity']:int(row['last_stage'])})
                self.dat['C_ACT'].update({row['activity']:row['cost']})
        
        self.dat.update(Q_BASE=[0 for i in range(self.dat['num_stages'])])
        self.dat.update(Q_MIN=0)

        self.dat['OPEX'] = 50
        self.dat['DECOM_COST_FIXED'] = 350
        self.dat['DECOM_COST_INCREASE'] = 0

        ######################################################################################
        ##### Estimated price data #####
        ######################################################################################

        self.dat.update(P_UPPER_BOUND=200)  # dollar/barrel, as tight as possible


        self.dat.update(chi_0=0.534)  # short-term increment of the log of the spot price (Deviation)
        self.dat.update(xi_0=3.633)  # long-term increment of the log of the spot price (Equilibrium)
        # numpy.exp(4.1)= 60
        self.dat.update(sigma_xi=0.149)  # (10%) The volatility of the long-term factor  (equilibrium)
        self.dat.update(sigma_chi=0.273)  # (29%) the volatility of the short term factor (deviation)
        #self.dat.update(mu_xi=0.02)
        self.dat.update(mu_xi=-0.007)  # (-0.5%)The risk neutral drift rate for the long-term factor
        self.dat.update(lambda_chi=-0.147)  # risk premium for  the short-term factor
        self.dat.update(kappa=0.407)  # mean-reversion coefficient, HALF_LIFE = ln(2)/kappa
        # http://marcoagd.usuarios.rdc.puc-rio.br/half-life.html
        self.dat.update(rho=0.306)  # The correlation coefficient between the random increments


        #################################
        ### Piecewise Linear Function ###
        #################################

        #this is updated later on
        self.dat.update(num_breakpoints=6)

        self.dat.update(breakpoints_price= [0.01]+
                                           list(numpy.round(numpy.linspace(start=1, stop=self.dat['P_UPPER_BOUND'],
                                                                           num=self.dat['num_breakpoints']-1), 3)))
        self.dat.update(breakpoints_factor=[numpy.round(numpy.log(x), 3) for x in self.dat['breakpoints_price']])
        self.dat.update(MAE=100)

        #################
        ### INSTANCES ###
        #################

        base_changes = True
        if base_changes:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1', 'sidetrack2','new_wells4']) #
        if self.instance == 0:
            print('use the three activities')  
        elif self.instance == 1:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1', 'sidetrack2','sidetrack3', 'new_wells4','new_wells6',
                'new_wells8','sidetrack1_alt', 'sidetrack2_alt','sidetrack3_alt', 'new_wells4_alt','new_wells6_alt',
                'new_wells8_alt'])
        elif self.instance == 2:
            self.update_parameter('A_ACTIVITIES', [])
        elif self.instance == 3:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1'])
        elif self.instance == 4:
            self.update_parameter('A_ACTIVITIES', ['sidetrack2'])
        elif self.instance == 5:
            self.update_parameter('A_ACTIVITIES', ['new_wells4'])
        elif self.instance == 6:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1','sidetrack2'])
        elif self.instance == 7:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1','new_wells4'])
        elif self.instance == 8:
            self.update_parameter('A_ACTIVITIES', ['sidetrack2','new_wells4'])  
        elif self.instance == 9:
            self.update_parameter('A_ACTIVITIES', ['sidetrack1','sidetrack2','new_wells4','new_wells8'])  

        ########################
        ##### DERIVED DATA #####
        ########################

    def generate_derived_data(self):

        self.dat.update(num_activities=len(self.dat['A_ACTIVITIES']))

        self.dat['C_OPEX'] = list(numpy.repeat(self.dat['OPEX'], self.dat['num_stages']))  # Million dollars
        for i in range(self.dat['num_stages']):
            opex_count = 0
            for j in range(self.dat['num_years_per_stage'][i]):
                opex_count += self.dat['OPEX']*numpy.power(1/(1+self.dat['r_riskfree']),j)
            self.dat['C_OPEX'][i] = round(opex_count,2)

        self.dat['C_DECOM'] = list(numpy.repeat(self.dat['DECOM_COST_FIXED'], self.dat['num_stages']))  # Million dollars
        decom_costs = [round(self.dat['DECOM_COST_FIXED']*numpy.power(1+self.dat['DECOM_COST_INCREASE'],t),2) for t in range(self.dat['num_stages'])]
        self.update_parameter('C_DECOM', decom_costs)


        self.dat.update(T_STAGES = list(range(self.dat['num_stages'])))  # ={0,...,T-1}
        self.dat.update(year_sequence = [0] + list(numpy.cumsum(self.dat['num_years_per_stage'][0:self.dat['num_stages'] - 1])))

        self.dat.update(discount_factor = [round(1 / numpy.power(1 + self.dat['r_riskfree'], t), 2) for t in self.dat['year_sequence']])

        year = 0
        for i in range(self.dat['num_stages']):
            self.dat['Q_BASE'][i] = sum(self.dat['Q_BASE_YEARLY'][year:year + self.dat['num_years_per_stage'][i]])
            year = year + self.dat['num_years_per_stage'][i]

        self.dat.update(Q_ADD={(a, t): [] for a in self.dat['A_ACTIVITIES'] for t in
                                self.dat['T_STAGES']})  # depends on activity and stage in which it is started
        for a in self.dat['A_ACTIVITIES']:
            for tau in range(self.dat['last_stage_to_start_activity'][a]):
                year = 0
                for i in range(self.dat['num_stages'] - tau):
                    self.dat['Q_ADD'][(a, tau)].append(
                        round(sum(self.dat['Q_ADD_YEARLY'][a][year:year + self.dat['num_years_per_stage'][tau + i]]),
                              2))  # t = tau + i
                    year = year + self.dat['num_years_per_stage'][tau + i]

        self.dat.update(Q_MAX = self.dat['Q_BASE'][0] + max([self.dat['Q_ADD'][a, 0][0] for a in self.dat['A_ACTIVITIES']], default=0))
        self.dat.update(MAX_REV = self.dat['Q_MAX'] * self.dat['P_UPPER_BOUND'])
        self.dat.update(OBJ_BOUND =
                        sum(self.dat['discount_factor'][tau]*(self.dat['P_UPPER_BOUND']*0.75)*self.dat['Q_ADD'][(a,0)][tau]
                           for tau in range(self.dat['num_stages']-1) for a in self.dat['A_ACTIVITIES'])
                        + sum(self.dat['discount_factor'][tau]*self.dat['P_UPPER_BOUND']*0.75*self.dat['Q_BASE'][tau]
                              for tau in range(self.dat['num_stages']-1))
                        #- sum(self.dat['C_ACT'][a] for a in self.dat['A_ACTIVITIES'])
                        #- sum(self.dat['discount_factor'][tau]*self.dat['C_OPEX'][0]
                        #      for tau in range(self.dat['num_stages']-1) )
                        - self.dat['C_DECOM'][0]*self.dat['discount_factor'][self.dat['num_stages']-1]
                        )
        #what is the maximum revenue that one can get?? This should not be used!! It can give deceiving values.
        #Better to run a relaxation to get the bound!!!
        self.dat.update(MAX_COST = max([self.dat['C_ACT'][a] for a in self.dat['A_ACTIVITIES']], default=0) +
                                    max(self.dat['C_OPEX']) + max(self.dat['C_DECOM']))

        self.dat.update(var_std_normal = 1)
        self.dat.update(covariance = self.dat['rho'] * numpy.sqrt(self.dat['var_std_normal'] * self.dat['var_std_normal']))

        self.dat.update(delta_t = list(numpy.arange(1, max(self.dat['num_years_per_stage']) + 1)))
        self.dat.update(eps_xi_par = [self.dat['sigma_xi'] * numpy.sqrt(t) for t in self.dat['delta_t']])
        self.dat.update(eps_chi_par = [
            round(self.dat['sigma_chi'] * numpy.sqrt((1 - numpy.exp(-2 * self.dat['kappa'] * t)) /
                                                      (2 * self.dat['kappa'])), 2)
            for t in self.dat['delta_t']])
        # These are actually the variances.. (Linear transformation of a st. normal. variable)
        self.dat.update(beta_chi = [round(numpy.exp(-self.dat['kappa'] * t), 2) for t in self.dat['delta_t']])
        self.dat.update(alpha_chi = [round(-(1 - numpy.exp(-self.dat['kappa'] * t)) *
                                            (self.dat['lambda_chi'] / self.dat['kappa']), 2)
                                      for t in self.dat['delta_t']])
        self.dat.update(alpha_xi = [self.dat['mu_xi'] * t for t in self.dat['delta_t']])

        # lower and upper bounds
        prob_error = 0.001  # in 1 out of 100000 cases, the bounds may be wrong
        num_years = sum(self.dat['num_years_per_stage'])
        standard_deviation = [numpy.sqrt(self.dat['eps_xi_par'][0] * num_years),
                              numpy.sqrt(self.dat['eps_chi_par'][0] * num_years)]
        self.dat.update(error_low = [0, 0])
        for i in range(2):
            self.dat['error_low'][i] = norm.ppf(prob_error, loc=0, scale=standard_deviation[i])

        # this determines the number of binaries in the model for TS approach
        self.dat.update(lower_bound_eq_factor = numpy.floor(
            min(self.dat['xi_0'] + self.dat['error_low'][0],
                self.dat['xi_0'] + self.dat['alpha_xi'][0] * num_years + self.dat['error_low'][0])))
        self.dat.update(upper_bound_eq_factor = numpy.ceil(
            max(self.dat['xi_0'] - self.dat['error_low'][0],
                self.dat['xi_0'] + self.dat['alpha_xi'][0] * num_years - self.dat['error_low'][0])))
        self.dat.update(lower_bound_dev_factor = numpy.floor(
            min(self.dat['chi_0'] + self.dat['error_low'][1],
                -self.dat['chi_0'] + self.dat['error_low'][1])))
        self.dat.update(upper_bound_dev_factor = -self.dat['lower_bound_dev_factor'])
        # funker ikke når disse ikke er runda!! Super rar også

        ###############################
        ## Piecewise linear function ##
        ###############################

        self.dat['num_segments'] = self.dat['num_breakpoints'] - 1
        self.dat.update(slopes = [])
        self.dat.update(intercepts = [])
        for i in range(self.dat['num_segments']):
            slope = (self.dat['breakpoints_price'][i + 1] - self.dat['breakpoints_price'][i]) / (
                    self.dat['breakpoints_factor'][i + 1] - self.dat['breakpoints_factor'][i])  # dy/dx
            self.dat['slopes'].append(slope)
            intercept = self.dat['breakpoints_price'][i] - self.dat['breakpoints_factor'][i] * slope
            self.dat['intercepts'].append(intercept)
        # DO SOME ROUNDING HERE!

        ###########
        ## QUERY ##
        ###########

        self.dat.update(x_act_combinations=[])
        self.dat.update(query=[])
        self.dat['query'].append('y_sd')
        self.dat['query'].append('price')
        for a in self.dat['A_ACTIVITIES']:
            for t in range(self.dat['last_stage_to_start_activity'][a]):
                self.dat['x_act_combinations'].append((t,a))
                self.dat['query'].append("x_act[{},{}]".format(t, a))

    def update_parameter(self,variable,value):
        self.dat[variable] = value
        self.dat['updated_variables'].append(variable)
