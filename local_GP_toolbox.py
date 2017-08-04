
# coding: utf-8

# In[1]:

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib as matplotlib
get_ipython().magic(u'matplotlib inline')


# In[2]:

def log_marginal_likelihood(covariance, data):
    number_datapoints = len(data)
    data = np.matrix(np.reshape(data,(number_datapoints,1)))
    covariance = np.matrix(covariance)
    cholesky = np.matrix(lin.cholesky(covariance))
    log_determinant = 2*np.sum(np.diag(cholesky))
    modified_data = lin.solve(cholesky, data)
    log_likelihood = -0.5*log_determinant -0.5*modified_data.T*modified_data
    log_likelihood = log_likelihood[0,0]
    return log_likelihood


# In[3]:

def normalize_probability(log_probability, log):
    if log is not "yes":
        log_probability = np.log(log_probability)
    alpha = np.amax(log_probability)
    normalization = np.sum(np.exp(log_probability - alpha))
    probability = np.divide(np.exp(log_probability - alpha), normalization)
    return probability


# In[4]:

def matrix_division(divider, divided, side, cholesky):
    X = np.matrix(divided)
    if cholesky is "yes":
        M = np.matrix(np.linalg.cholesky(divider))
        if side is "right":
            first_division = np.linalg.solve(M,X.T).T
            result = np.linalg.solve(M.T,first_division.T).T
        elif side is "left":
            first_division = np.linalg.solve(M,X)
            result = np.linalg.solve(M.T,first_division)
        else:
            print "The side should be either left or right"
            return
    else:
        M = np.matrix(divider)
        if side is "right":
            result = np.linalg.solve(M.T,X.T).T
        elif side is "left":
            result = np.linalg.solve(M,X)
        else:
            print "The side should be either left or right"
            return
    return result


# In[5]:

def construct_covariance_matrices(training_time, covariance_function_and_parameters, **named_input):
    #
    training_grid_x, training_grid_y = np.meshgrid(training_time, training_time)
    training_difference_matrix = training_grid_x - training_grid_y
    training_covariance = np.matrix(covariance_function_and_parameters["function"](training_difference_matrix, covariance_function_and_parameters["parameters"]))
    if named_input.has_key("target_time"):
        target_time = named_input["target_time"]
        target_grid_x, target_grid_y = np.meshgrid(target_time, target_time)
        target_difference_matrix = target_grid_x - target_grid_y
        extrapolation_grid_x, extrapolation_grid_y = np.meshgrid(target_time, training_time)
        extrapolation_difference_matrix = extrapolation_grid_x - extrapolation_grid_y
        target_covariance = np.matrix(covariance_function_and_parameters["function"](target_difference_matrix, covariance_function_and_parameters["parameters"]))
        extrapolation_covariance = np.matrix(covariance_function_and_parameters["function"](extrapolation_difference_matrix, covariance_function_and_parameters["parameters"]))
        return {"training": training_covariance,"target": target_covariance, "extrapolation": extrapolation_covariance}
    else:
        return {"training": training_covariance}


# In[6]:

class locally_coupled_GP_analysis(object):
    #
    def __init__(self):
        self.data = data()
        self.likelihood = likelihood()
        self.prior_HMM = prior_HMM()
        self.posterior_HMM = posterior_HMM()
        self.local_prior_GP = local_prior_GP()
        self.global_prior_GP = global_prior_GP()
        self.posterior_GP = posterior_GP()
        self.windows = windows()
        self.archive_covariance_functions = {"white_noise": lambda tau, parameters: parameters["standard_deviation"]**2*np.exp(-np.abs(tau)*10**5),
                                             "squared_exponential": lambda tau, parameters: parameters["amplitude"]*np.exp(-(tau/parameters["time_scale"])**2/2.),
                                             "oscillatory": lambda tau, parameters: parameters["amplitude"]*np.exp(-(tau/parameters["width"])**2/2.)*np.cos(2*np.pi*parameters["frequency"]*tau),
                                             "multistate": lambda tau, parameters: parameters["state"]*parameters["first_state"](tau) + (1 - parameters["state"])*parameters["second_state"](tau)}
        self.archive_expectation_functions = {"constant": lambda t, parameters: parameters["expectation"] + 0*t}
        self.archive_window_functions = {"gaussian": lambda t, center, parameters: np.exp(-((t - center)/parameters["width"])**2/2.),
                                         "generalized_gaussian": lambda t, center, parameters: np.exp(-((t - center)/parameters["width"])**parameters["power"]/2.)}
    
    def initialize_windows(self, function_and_parameters, spacing):
        # This method initializes the window functions
        self.windows.window_function = function_and_parameters
        self.windows.spacing = spacing
        print "Constructing window functions"
        self.windows.construct_windows(self.data.training_time) 
        self.windows.initialized = "yes"
        print "The window functions have been initialized"
        
    def initialize_local_prior_GP(self, covariance_function_and_parameters, expectation_function_and_parameters):
        # This method initializes the prior of the local Gaussian processes
        self.local_prior_GP.expectation_function = expectation_function_and_parameters
        self.local_prior_GP.covariance_function = covariance_function_and_parameters
        self.local_prior_GP.construct_covariance_matrices(self.data.training_time, self.data.target_time)
        print "Constructing covariance matrix"
        self.local_prior_GP.construct_expectations(self.data.training_time, self.data.target_time)
        print "Constructing expectations"
        self.local_prior_GP.initialized = "yes"
        print "The local Gaussian process priors have been initialized"
        
    def initialize_likelihood(self, noise_covariance_function_and_parameters):
        self.likelihood.noise_covariance_function_and_parameters = noise_covariance_function_and_parameters
        print "Constructing noise covariance matrix"
        self.likelihood.compute_global_noise_covariance(self.data.training_time)
        self.likelihood.initialized = "yes"
        print "The likelihood (noise model) has been initialized"
        
    def initialize_prior_HMM(self, expectation, autoregressive_coefficient, innovation_variance, initial_expectation, initial_variance):
        # This method initializes the prior distribution of the autoregressive hidden markov model
        if self.local_prior_GP.initialized is not "yes":
            print "The local GP prior needs to be initialized before the initialization of the HMM prior"
            return
        self.prior_HMM.expectation = expectation
        self.prior_HMM.autoregressive_coefficient = autoregressive_coefficient
        self.prior_HMM.innovation_variance = innovation_variance
        self.prior_HMM.initial_distribution_expectation = initial_expectation
        self.prior_HMM.initial_distribution_variance = initial_variance
        print "Constructing transfer matrix"
        self.prior_HMM.construct_transfer_matrix(hidden_state_range = self.local_prior_GP.covariance_function["nonstationary_parameter"]["range"])
        self.prior_HMM.construct_initial_probability(hidden_state_range = self.local_prior_GP.covariance_function["nonstationary_parameter"]["range"])
        self.prior_HMM.initialized = "yes"
        print "The hidden Markov model prior has been initialized"
        
    def initialize_analysis(self, **named_input):
        # This method initializes the analysis with default configurations
        if self.data.initilized is not "yes":
            print "You have to input the data before initialising the analysis"
            return
        # Initialize windows
        if named_input.has_key("windows_spacing"):
            spacing =  named_input["windows_spacing"]
            number_windows = int(np.max(self.data.training_time)/spacing)
            print "Using a personalized windows spacing of " + str(spacing) + " (" + str(number_windows) + " windows)"
        else:
            spacing = np.max(self.data.training_time)/10.
            print "Using a default windows spacing of " + str(spacing) + " (ten windows)"
        if named_input.has_key("window_function_type"):
            window_function_type = named_input["window_function_type"]
            print "Using " + window_function_type + " as window function"
        else:
            window_function_type = "gaussian"
            print "Using the default Gaussian window function"
        if named_input.has_key("window_parameters"):
                window_parameters = named_input["window_parameters"]
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The personalized windows parameters are: " + str(parameters_list)
        else:
            if window_function_type is "gaussian":
                window_width = spacing
                window_parameters = {"width": window_width}
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The default windows parameters are: " + str(parameters_list)
            else:
                print "You need to input the parameters of the personalized window function"
                return
        window_function = {"function": self.archive_window_functions[window_function_type], "parameters": window_parameters}
        self.initialize_windows(function_and_parameters = window_function, spacing = windows_spacing)
        # Initialize local GP prior
        if named_input.has_key("covariance_function_type"):
            covariance_function_type = named_input["covariance_function_type"]
            print "Using " + covariance_function_type + " as covariance function"
        else:
            covariance_function_type = "squared_exponential"
            print "Using the default squared exponential as covariance function"
        if named_input.has_key("nonstationary_parameter_type"):
            nonstationary_parameter_type = named_input.has_key["nonstationary_parameter_type"]
            if named_input.has_key("nonstationary_parameter_range"):
                nonstationary_parameter_range = named_input["nonstationary_parameter_range"]
                print "Using " + nonstationary_parameter_type + " as nonstationary parameter of the " + covariance_function_type + " covariance_function"
            else:
                print "You have to specifify the range of " + nonstationary_parameter_type + "(input nonstationary_parameter_range)"
                return
        else:
            if covariance_function_type is "squared_exponential":
                nonstationary_parameter_type = "time_scale"
                print "Using " + nonstationary_parameter_type + "as nonstationary parameter of the squared exponential covariance function"
                time_scale_min = 0.
                time_scale_max = 1.
                time_scale_step = 0.05
                nonstationary_parameter_range = np.arange(time_scale_min, time_scale_max, time_scale_step)
            else:
                print "you have to specify the type and the range of the nonstationary parameter"
                return
        if named_input.has_key("covariance_parameters"):
                parmateters = named_input["covariance_parameters"]
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The personalized covariance parameters are: " + str(parameters_list)
        else:
            if covariance_function_type is "squared_exponential":
                parmateters = {"covariance": 1}
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The default covariance parameters are: " + str(parameters_list)
            else:
                print "You need to input the parameters of the personalized covariance function"
                return
        nonstationary_parameter = {"type": nonstationary_parameter_type, "range": nonstationary_parameter_range}
        covariance_function = {"function": self.archive_covariance_functions[covariance_function_type], "parmateters":parmateters, "nonstationary_parameter": nonstationary_parameter}
        if named_input.has_key("expectation_function_type"):
            expectation_function_type = named_input["expectation_function_type"]
            print "Using " + covariance_function_type + " as expectation function"
        else:
            covariance_function_type = "constant"
            print "Using the default squared exponential as covariance function"
        if named_input.has_key("expectation_parameters"):
                expectation_parameters = named_input["expectation_parameters"]
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in expectation_parameters.iteritems()])
                print "The personalized expectation parameters are: " + str(parameters_list)
        else:
            if expectation_function_type is "constant":
                expectation_parmateters = {"expectation": 0}
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in expectation_parameters.iteritems()])
                print "The default covariance parameters are: " + str(parameters_list)
            else:
                print "You need to input the parameters of the personalized expectation function"
                return
        expectation_function = {"function": self.archive_expectation_functions[expectation_function_type], "parameters": expectation_parmateters}
        self.initialize_local_prior_GP(covariance_function_and_parameters = covariance_function, expectation_function_and_parameters = expectation_function)
        # Initialization global likelihood model
        if named_input.has_key("noise_covariance_function_type"):
            noise_covariance_function_type = named_input["noise_covariance_function_type"]
            print "Using " + covariance_function_type + " as noise covariance function"
        else:
            covariance_function_type = "white_noise"
            print "Using the default white noise likelihood"
        if named_input.has_key("noise_parameters"):
                noise_parmateters = named_input["noise_parameters"]
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The personalized noise parameters are: " + str(parameters_list)
        else:
            if noise_covariance_function_type is "white_noise":
                parmateters = {"standard_deviation": 0.1}
                parameters_list = ', '.join([ key + " = " + str(value) for key, value in parameters.iteritems()])
                print "The default noise covariance parameters are: " + str(parameters_list)
            else:
                print "You need to input the parameters of the personalized covariance function"
                return
            
        noise_covariance_function_and_parameters = {"function": self.archive_covariance_functions[noise_covariance_function_type], "parmateters":noise_parmateters}
        self.initialize_likelihood(noise_covariance_function_and_parameters)
        # Initialize HMM prior model
        if named_input.has_key("HMM_expectation"):
            HMM_expectation = named_input["HMM_expectation"] 
            print "Using a personalized expectation of the hidden Markov model of " + str(HMM_expectation)
        else:
            HMM_expectation = 0
            print "Using a default expectation of the hidden Markov model of " + str(HMM_expectation)
        if named_input.has_key("HMM_autoregressive_coefficient"):
            HMM_autoregressive_coefficient = named_input["HMM_autoregressive_coefficient"]
            print "Using a personalized autoregressive coefficient of the hidden Markov model of " + str(HMM_autoregressive_coefficient)
        else:
            HMM_autoregressive_coefficient = 1
            print "Using a random walk hidden Markov model (the autoregressive coefficient is " + str(HMM_autoregressive_coefficient) + ")"
        if named_input.has_key("HMM_innovation_variance"):
            HMM_innovation_variance = named_input["HMM_innovation_variance"]
            print "Using a personalized innovation variance of the hidden Markov model of " + str(HMM_innovation_variance)
        else:
            HMM_innovation_variance = 3*np.median(np.diff(self.local_prior_GP.covariance_function["nonstationary_parameter"]["range"]))
            print "Using a default innovation variance of the hidden Markov model of " + str(HMM_innovation_variance) + ")"
        if named_input.has_key("HMM_initial_expectation"):
            HMM_initial_expectation = named_input["HMM_initial_expectation"] 
            print "Using a personalized initial expectation of the hidden Markov model of " + str(HMM_expectation)
        else:
            HMM_initial_expectation = 0
            print "Using a default initial expectation of the hidden Markov model of " + str(HMM_expectation)
        if named_input.has_key("HMM_initial_variance"):
            HMM_initial_variance = named_input["HMM_initial_variance"] 
            print "Using a personalized initial variance of the hidden Markov model of " + str(HMM_expectation)
        else:
            HMM_initial_variance = 10**2
            print "Using a default initial variance of the hidden Markov model of " + str(HMM_expectation)
        self.initialize_prior_HMM(HMM_expectation, HMM_autoregressive_coefficient, HMM_innovation_variance, HMM_initial_expectation, HMM_initial_variance)
        print "The analysis has been fully initialized."
        
    def input_data(self, training_data, **named_input):
        self.data.training_data = training_data
        if named_input.has_key("training_time"):
            time_parameters = named_input["training_time"]
            if time_parameters.has_key("time_min") and time_parameters.has_key("time_max") and time_parameters.has_key("time_step"):
                self.data.training_time = np.arange(time_parameters["time_min"],time_parameters["time_max"],time_parameters["time_step"])
            else:
                print "The input 'training_time' should contain the following keys: 'time_min', 'time_min' and 'time_step'"
        else:
            print "You need to specify the training time"
            return
        if named_input.has_key("target_time"):
            time_parameters = named_input["target_time"]
            if time_parameters.has_key("time_min") and time_parameters.has_key("time_max") and time_parameters.has_key("time_step"):
                self.data.target_time = np.arange(time_parameters["time_min"],time_parameters["time_max"],time_parameters["time_step"])
            else:
                print "The input 'target_time' should contain the following keys: 'time_min', 'time_min' and 'time_step'"
        else:
            self.data.target_time = self.data.training_time 
            print "The target time has not been specified, using the training time as target time"
        self.data.initialized = "yes"
        print "The data have been correctly inputted"
        return
        
    def perform_hierarchical_analysis(self):
        # Check whether the objects are initialized
        if self.data.initialized is not "yes":
            print "You need to input the training data before computing the posterior"
            return
        elif self.prior_HMM.initialized is not "yes":
            print "You need to initialize the prior of the hidden Markov model before computing the posterior"
            return
        elif self.local_prior_GP.initialized is not "yes":
            print "You need to initialize the prior of the local Gaussian processes before computing the posterior"
            return
        elif self.windows.initialized is not "yes":
            print "You need to initialize the window functions before computing the posterior"
            return
        elif self.likelihood.initialized is not "yes":
            print "You need to initialize the global likelihood before computing the posterior"
            return
        # Compute the local likelihoods
        print "Computing the local likelihoods"
        self.likelihood.compute_local_likelihood(self.windows.windows, self.local_prior_GP.local_covariance_matrices["training"], self.data.training_data)
        # Construct the posterior of the latent variables (HMM)
        print "Computing the posterior of the hidden Markov model"
        self.posterior_HMM.apply_forward_algorithm(self.likelihood.local_likelihoods, self.prior_HMM.transfer_matrix, self.prior_HMM.initial_distribution, self.windows.windows)
        self.posterior_HMM.apply_backward_algorithm(self.likelihood.local_likelihoods, self.prior_HMM.transfer_matrix, self.windows.windows)
        self.posterior_HMM.compute_point_estimate(self.local_prior_GP.covariance_function["nonstationary_parameter"]["range"])
        self.posterior_HMM.initialized = "yes"
        # Construct the posterior of the global GP process
        print "Constructing the prior of the global Gaussian process"
        self.global_prior_GP.construct_covariance(self.local_prior_GP.local_covariance_matrices, self.windows.windows, self.posterior_HMM.expectation_index)
        self.global_prior_GP.initialized = "yes"
        print "Computing the posterior of the global Gaussian process"
        self.posterior_GP.apply_GP_regression(self.data.training_data, self.global_prior_GP.covariance_matrices, self.likelihood.global_noise_covariance)
        self.posterior_GP.initialized = "yes"
        print "The locally coupled GP analysis has been succesfully completed"
    #def plot_GP_posterior():
        #
        
    #def plot_HMM_posterior():
        #
        


# In[7]:

class likelihood(object):
    #
    def __init__(self):
        self.noise_covariance_function_and_parameters = []
        self.local_likelihoods = []
        self.global_noise_covariance = []
        self.initialized = "no"
        
    def compute_local_likelihood(self, windows, covariance_matrices, data):
        local_log_likelihoods_array = []
        for covariance_matrix in covariance_matrices:
            local_log_likelihoods_row = []
            for window in windows:
                observation_model = np.matrix(np.diag(window.flatten()))
                local_covariance = observation_model*covariance_matrix*observation_model.T + self.global_noise_covariance
                local_log_likelihood = log_marginal_likelihood(local_covariance, data)
                local_log_likelihoods_row = local_log_likelihoods_row + [local_log_likelihood]
            local_log_likelihoods_array = local_log_likelihoods_array + [local_log_likelihoods_row]
        local_log_likelihoods_array = np.array(local_log_likelihoods_array)
        # Normalization
        probability_array = np.zeros(local_log_likelihoods_array.shape)
        for index in range(0,len(windows)):
            probability = normalize_probability(local_log_likelihoods_array[:, index], log = "yes")
            probability_array[:, index] = probability
        self.local_likelihoods = probability_array
        return
    
    def compute_global_noise_covariance(self, training_time):
        noise_covariance_matrices = construct_covariance_matrices(training_time, self.noise_covariance_function_and_parameters)
        noise_covariance_matrix = noise_covariance_matrices["training"] 
        self.global_noise_covariance = noise_covariance_matrix
        return                    


# In[8]:

class posterior_GP(object):
    #
    def __init__(self):
        self.expectation = []
        self.covariance_matrix = []
        self.variance = []
        self.initialized = "no"
        
    def apply_GP_regression(self, data, covariance_matrices, noise_covariance):
        data = np.matrix(np.reshape(data,(len(data),1)))
        modified_data = matrix_division(divider = covariance_matrices["training"] + noise_covariance, divided = data, side = "left", cholesky = "yes")
        self.expectation = covariance_matrices["extrapolation"]*modified_data
        self.covariance_matrix = covariance_matrices["target"] - covariance_matrices["extrapolation"]*matrix_division(divider = covariance_matrices["training"] + noise_covariance, divided = covariance_matrices["extrapolation"].T, side = "left", cholesky = "yes")
        self.variance = np.diag(self.covariance_matrix)
        return


# In[9]:

class posterior_HMM(object):
    #
    def __init__(self):
        self.posterior_probability = []
        self.expectation = []
        self.expectation_index = []
        self.estimate_type = "expectation"
        self.initialized = "no"
        self.foward_algorithm = "no"
        self.backward_algorithm = "no"
        
    def apply_forward_algorithm(self, marginal_likelihoods, transfer_matrix, initial_distribution, windows):
        if self.foward_algorithm is "yes":
            print "The forward algorithm has already been applied"
            return
        probability_array = np.zeros(marginal_likelihoods.shape)
        likelihood = np.reshape(marginal_likelihoods[:,0], (len(marginal_likelihoods[:,0]),1))
        initial_distribution = np.reshape(initial_distribution, (len(initial_distribution),1))
        probability = normalize_probability(np.multiply(likelihood, initial_distribution), log = "no")
        probability_array[:,0] = probability.flatten()
        for window_time_index in range(1, len(windows)):
            likelihood = np.reshape(marginal_likelihoods[:,window_time_index], (len(marginal_likelihoods[:,window_time_index]),1))
            predicted_probability = transfer_matrix*np.matrix(probability)
            probability = normalize_probability(np.multiply(likelihood, predicted_probability), log = "no")
            probability_array[:,window_time_index] = probability.flatten()
        self.posterior_probability = probability_array
        self.foward_algorithm = "yes"
        return
    
    def apply_backward_algorithm(self, marginal_likelihoods, transfer_matrix, windows):
        if self.foward_algorithm is not "yes":
            print "You have to apply the forward algorithm first"
            return
        elif self.backward_algorithm is "yes":
            print "The backward algorithm has already been applied"
            return   
        probability_array = self.posterior_probability
        parameter_range_length = len(marginal_likelihoods[:,-1])
        beta = np.ones(shape = (parameter_range_length,1))
        for window_time_index in reversed(range(-len(windows),0)):
            beta = normalize_probability(beta, log = "no")
            previous_probability = np.reshape(probability_array[:,window_time_index], newshape = (parameter_range_length,1))
            probability = normalize_probability(np.multiply(previous_probability,beta), log = "no")
            probability_array[:,window_time_index] = probability.flatten()
            likelihood = np.reshape(marginal_likelihoods[:,window_time_index], (len(marginal_likelihoods[:,window_time_index]),1))
            beta = transfer_matrix*np.matrix(np.multiply(beta,likelihood))
        self.posterior_probability = probability_array
        self.backward_algorithm = "yes"
        return
    
    def compute_point_estimate(self, parameter_range):
        if self.foward_algorithm is not "yes":
            print "You have to compute the posterior distribution first"
            return
        if self.estimate_type is "expectation":
            parameter_range = np.reshape(parameter_range, (len(parameter_range),1))
            parameter_posterior_expectation = np.sum(np.multiply(self.posterior_probability, parameter_range),0)
            self.expectation = parameter_posterior_expectation
            expectation_index = []
            for expectation in parameter_posterior_expectation:
                closest_match = np.argmin(np.abs(parameter_range - expectation))
                expectation_index = expectation_index + [closest_match]
            self.expectation_index = expectation_index
        elif self.estimate_type is "mode":
            self.expectation_index = np.argmax(self.posterior_probability, axis=0) 
            self.expectation = []
            for index in self.expectation_index:
                self.expectation = self.expectation + [parameter_range[index]]
            self.expectation = np.array(self.expectation)                                      
        else:
            print "The required point estimate is currently not implemented"
        return


# In[10]:

class global_prior_GP(object):
    #
    def __init__(self):
        self.expectation = []
        self.covariance_matrices = []
        self.initialized = "no"
    
    def construct_covariance(self, local_covariance_matrices, windows, estimated_parameter_indices):
        window_index = 0
        training_covariance = 0
        target_covariance = 0
        extrapolation_covariance = 0
        for estimated_parameter_index in estimated_parameter_indices:
            observation_model = np.matrix(np.diag(windows[window_index].flatten()))
            local_training_covariance = observation_model*local_covariance_matrices["training"][estimated_parameter_index]*observation_model.T
            local_target_covariance = local_covariance_matrices["target"][estimated_parameter_index]  
            local_extrapolation_covariance = observation_model*local_covariance_matrices["extrapolation"][estimated_parameter_index]*observation_model.T
            training_covariance = training_covariance + local_training_covariance
            target_covariance = target_covariance + local_target_covariance
            extrapolation_covariance = extrapolation_covariance + local_extrapolation_covariance
            window_index = window_index + 1
        self.covariance_matrices = {"training": training_covariance,"target": target_covariance,"extrapolation": extrapolation_covariance}
        return
    
    def construct_expectation():
        print "TO DO!!"
        return


# In[11]:

class local_prior_GP(object):
    #
    def __init__(self):
        self.expectation_function = {"function": [], "parameters": []}
        self.covariance_function = {"function": [], "parameters": [], "nonstationary_parameter": []}
        self.local_covariance_matrices = []
        self.local_expectations = []
        self.initialized = "no"
    
    def construct_covariance_matrices(self, training_time, target_time):
        local_training_covariance_matrices = []
        local_target_covariance_matrices = []
        local_extrapolation_covariance_matrices = []
        for parameter_value in self.covariance_function["nonstationary_parameter"]["range"]:
            covariance_function = self.covariance_function["function"]
            parameters = (self.covariance_function["parameters"]).copy()
            parameters.update({self.covariance_function["nonstationary_parameter"]["type"]: parameter_value})
            covariance_function_and_parameters = {"function": covariance_function,"parameters": parameters}
            matrices = construct_covariance_matrices(training_time = target_time ,target_time = target_time, covariance_function_and_parameters = covariance_function_and_parameters)
            local_training_covariance_matrices = local_training_covariance_matrices + [matrices["training"]]
            local_target_covariance_matrices = local_target_covariance_matrices + [matrices["target"]]
            local_extrapolation_covariance_matrices = local_extrapolation_covariance_matrices + [matrices["extrapolation"]]
        self.local_covariance_matrices = {"training": local_training_covariance_matrices, "target": local_target_covariance_matrices, "extrapolation": local_extrapolation_covariance_matrices}
            
    def construct_expectations(self, training_time, target_time):
        training_expectations = self.expectation_function["function"](training_time, self.expectation_function["parameters"])
        target_expectations = self.expectation_function["function"](target_time, self.expectation_function["parameters"])                                                            
        self.local_expectations = {"training": training_expectations,"target": target_expectations}


# In[12]:

class prior_HMM(object):
    #
    def __init__(self):
        self.expectation = []
        self.transfer_matrix = []
        self.autoregressive_coefficient = []
        self.innovation_variance = []
        self.initial_distribution_expectation = []
        self.initial_distribution_variance = []
        self.initial_distribution = []
        self.initialized = "no"
    
    def construct_transfer_matrix(self, hidden_state_range):
        transfer_matrix = []
        normal_pdf = lambda x,m,v: np.sqrt(2*np.pi*v)**(-1)*np.exp(-(x - m)**2/(2*v))
        for next_state in hidden_state_range:
            transfer_matrix_row = []
            for previous_state in hidden_state_range:
                conditional_expectation = self.expectation + self.autoregressive_coefficient*previous_state
                probability = normal_pdf(next_state ,conditional_expectation, self.innovation_variance)
                transfer_matrix_row = transfer_matrix_row + [probability]
            transfer_matrix = transfer_matrix + [transfer_matrix_row]
        transfer_matrix = np.matrix(transfer_matrix)
        normalization = transfer_matrix.sum(axis=0)
        transfer_matrix = transfer_matrix / np.array(normalization)
        self.transfer_matrix = transfer_matrix
        
    def construct_initial_probability(self, hidden_state_range):
        initial_probability = []
        normal_pdf = lambda x,m,v: np.sqrt(2*np.pi*v)**(-1)*np.exp(-(x - m)**2/(2*v))
        initial_distribution = normal_pdf(hidden_state_range, self.initial_distribution_expectation, self.initial_distribution_variance)
        self.initial_distribution = np.reshape(initial_distribution, (len(initial_distribution),1))/np.sum(initial_distribution)
            


# In[13]:

class windows(object):
    #
    def __init__(self):
        self.window_function = {"function": [], "parameters": []}
        self.spacing = []
        self.windows = []
        self.initialized = "no"
    
    def construct_windows(self, time):
        time_length = len(time)
        initial_time = np.amin(time)
        final_time = np.amax(time)
        window_centers = np.arange(initial_time, final_time, self.spacing)
        number_windows = len(window_centers)
        windows = []
        normalization = 0
        for window_index in range(0,number_windows):
            window = self.window_function["function"](time, window_centers[window_index], self.window_function["parameters"])
            windows = windows + [np.reshape(window,(time_length,1))]
            normalization = normalization + np.power(window,2)
        normalization = np.sqrt(normalization)
        windows = [np.divide(window, np.reshape(normalization,windows[0].shape)) for window in windows ]
        self.windows = windows
   


# In[14]:

class data(object):
    #
    def __init__(self):
        self.training_data = []
        self.training_time = []
        self.target_time = []
        self.initialized = "no"