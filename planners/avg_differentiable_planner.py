from math import exp
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from planners.extended_sampling_costs_planner import ExtendedSamplingCostsPlanner
from planners.sampling_planner import SamplingPlanner
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig

NUM_DESIRED_WAYPOINTS = 10
DISPLAY_GRADIENTS = True  
DUMMY_SC = [8.5, 18.95, 3.14]

class AvgDifferentiablePlanner(NNPlanner):
    """ A sampling-based planner that is differentiable with respect 
    to its inputs and internal parameters which selects an optimal spline
    plan using a trained (costmap) neural network.
    Given the sampled plans, will take a weighted average depending on the
    samples' costs and uncertainties.
    """

    def __init__(self, simulator, params):
        super(AvgDifferentiablePlanner, self).__init__(simulator, params)
        self.theta = self.params.diff_planner_uncertainty_weight
        self.len_costmap = self.params.len_costmap
        self.finite_differencing_delta = self.params.diff_planner_finite_diff_delta
        self.plotting_clip_value = self.params.diff_planner_plotting_clip_value 
        self.data_path = self.params.diff_planner_data_path

        # For the optimal waypt chosen
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)  
        
        # Uncertainty and noise schemes
        self.uncertainty_amount = 0.1
        self.noise = 0.0
        self.pre_determined_uncertainties = \
            tf.random_uniform([self.len_costmap], dtype=tf.double) * self.uncertainty_amount
        self.pre_determined_noise = np.random.rand(self.len_costmap)*self.noise

        # For taking gradients of the planner wrt uncertainty
        tfe = tf.contrib.eager
        self.costs = tfe.Variable(np.zeros((self.len_costmap)))
        self.uncertainties = tfe.Variable(np.zeros((self.len_costmap)))
        self.candidate_waypoints = tfe.Variable(np.zeros((self.len_costmap, 3)))

        # For getting an extended costmap of the environment
        self.sampling_costs_planner = ExtendedSamplingCostsPlanner(simulator, params)
        self.sampling_costs_planner_called = False  # Call it only once
        self.sampling_planner = SamplingPlanner(simulator, params)
        # For controlling the size of that costmap
        self.full_costmap_indices = []

        # Part of that costmap will be sampled to get the ground truth NN costmap
        self.nn_costmap_subsampling_indices = []

        # Printing to files
        self.analytical_gradient_computation = True 
        if self.analytical_gradient_computation:
            try:  # If file exists, clear it 
                f = open("analytical_gradient.txt")
                f.close()
                f = open("analytical_gradient.txt", "w")  # Clear contents of file
            except FileNotFoundError:
                f = open("analytical_gradient.txt","a")  # Create file
            
            self.analytical_file = f
            self.analytical_i = 0
            self.max_len_analytical_file = 100
        
        self.one_pt_gradient = True 
        if self.one_pt_gradient:
            try:  # If file exists, clear it 
                f = open("one_pt_gradient.txt")
                f.close()
                f = open("one_pt_gradient.txt", "w")  # Clear contents of file
            except FileNotFoundError:
                f = open("one_pt_gradient.txt","a")  # Create file
            
            self.one_pt_gradient_file = f
            self.one_pt_gradient_i = 0
            self.max_len_one_pt_gradient_file = 100

    def get_uncertainties(self, mode='random', nn_costs=None):
        '''
        For now a dummy function that will just output some hand-coded numbers, 
        should later be a function defined by the model.
        '''
        if mode == 'random':
            uncertainties = self.pre_determined_uncertainties
        elif mode == 'uniform':
            uncertainties = \
                tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.double)
        elif mode == 'high_on_gt':
            uncertainties = \
                tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=tf.double)
        elif mode == 'low_on_gt':
            uncertainties = \
                tf.constant([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0], dtype=tf.double)
        elif mode == 'high_on_non_gt':
            uncertainties = \
                tf.constant([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.double)
        elif mode == 'high_for_low_cost':
            assert nn_costs is not None
            multiplier = 1.0
            uncertainties = np.array([multiplier/elem for elem in nn_costs])
            uncertainties = tf.constant(uncertainties, dtype=tf.double)
        elif mode == 'proportional_to_cost':
            assert nn_costs is not None
            multiplier = 1.0
            uncertainties = np.array([multiplier * elem for elem in nn_costs])
            uncertainties = tf.constant(uncertainties, dtype=tf.double)
        else:
            raise Exception("Unknown uncertainty mode!")
        return uncertainties
    
    def get_cost_of_a_waypoint(self, start_config, waypoint):
        '''
        From the given start_config, which is an array, to the
        waypoint, which is in egocentric coordinates and is also
        an array
        '''
        # Create the SystemConfig object from this start_config array
        pos_nk2 = np.reshape(start_config[:2], (1, 1, 2)) 
        pos_nk2 = tf.convert_to_tensor(pos_nk2, dtype=tf.float32) 
        head_nk1 = np.reshape(start_config[2], (1, 1, 1))
        head_nk1 = tf.convert_to_tensor(head_nk1, dtype=tf.float32)
        start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                        position_nk2=pos_nk2,
                                        heading_nk1=head_nk1)

        # First transform waypoint to World Coordinates
        pos_nk2 = np.reshape(waypoint[:, :2], (1, 1, 2))
        pos_nk2 = tf.convert_to_tensor(pos_nk2, dtype=tf.float32)
        head_nk1 = np.reshape(waypoint[:, 2], (1, 1, 1))
        head_nk1 = tf.convert_to_tensor(head_nk1, dtype=tf.float32)
        waypoint_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                        position_nk2=pos_nk2,
                                        heading_nk1=head_nk1)
        # waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)
        # self.params.system_dynamics.to_world_coordinates(start_sys_config,
        #                                                 waypoint_config,
        #                                                 waypoint_world_config)
        
        # Now retrieve Control Pipeline data
        # obj_val, data = self.eval_objective(start_sys_config, waypoint_world_config)
        obj_val, data = self.eval_objective(start_sys_config, waypoint_config)

        return obj_val.numpy(), data

    def get_gradient_cost_wrt_plan(self, waypoint, start_config):
        '''
        Given the robot 'start_config' and the planner's generated plan
        'waypoint', samples some waypoints near 'waypoint' and gets their 
        costs traveling from 'start_config'. Then, performs finite
        differencing to approximate the derivative of the task cost
        with respect to the planner's plan. 

        The finite differencing looks like 
         [cost(waypoint + delta) - cost(waypoint - delta)] / 2*|delta|
        for delta being a perturbation in x, y, and theta directions.
        '''
        # Generate all the perturbations to 'waypoint' -- in x, y, and theta
        deltas = [np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]]
        deltas = [self.finite_differencing_delta * elem for elem in deltas] 
        deltas = [elem.reshape((3, 1)) for elem in deltas]
        perturbed_waypoints = []
        for delta in deltas:
            perturbed_waypoints.append(waypoint - delta)
            perturbed_waypoints.append(waypoint + delta)
        
        # Generate the costs for all perturbations
        perturbed_costs = []
        for perturbed_waypoint in perturbed_waypoints:
            obj_val, _ = self.get_cost_of_a_waypoint(start_config, np.reshape(perturbed_waypoint, (1, 3)))
            perturbed_costs.append(obj_val)

        cost_grads = []
        i = 0
        while i < len(perturbed_costs):  # Should be 6 elements in perturbed_costs
            numerator = perturbed_costs[i + 1] - perturbed_costs[i]
            denominator = self.finite_differencing_delta * 2
            cost_grads.append(numerator/denominator)
            i += 2
        
        cost_grad = tf.convert_to_tensor(np.reshape(cost_grads, (1, 3)), dtype=tf.double)  # [1, 3]

        return cost_grad, perturbed_waypoints, perturbed_costs 

    def planner_loss(self, start_config, costmap, uncertainties):
        '''
        We're not actually training the planner but this is important
        for the definition of the planner gradient. 
        start_config: the current pose of the robot -- needed to estimate the
                    derivative of the cost wrt the generated plan
        costmap: a W x 4 array, in which W is the number of sample trajectories
                outputted by the neural network and 4 represents the size of the
                waypoint (3) + 1 for the cost of the waypoint
        uncertainties: uncertainty values per sample trajectory/waypoint

        First computes the derivative of the output plan with respect to uncertainty.
        Then multiplies this value by the derivative of the task cost with respect
        to the plan. The final derivative is of the task cost with respect to 
        uncertainty.
        '''
        self.costs.assign(costmap[:, 3])
        self.uncertainties.assign(uncertainties)
        self.candidate_waypoints.assign(costmap[:, :3])

        # Compute the same thing 3 times, once for x, y, and theta dimensions
        # Why: TF v1 has no jacobian operator implemented -- have to take gradients 3 times
        # Also tape.gradient cannot be used multiple times within a context
        with tf.GradientTape() as tape:
            # Take the weighted average of the candidate waypoints
            weights = self.costs + self.theta * self.uncertainties
            inverse_weights = tf.reshape(tf.divide(1.0, weights), [self.len_costmap, 1])
            numerator = tf.matmul(tf.transpose(self.candidate_waypoints), inverse_weights)
            denominator = tf.reduce_sum(inverse_weights)
            waypoint = tf.divide(numerator, denominator)
        
            #grads = tape.gradient(waypoint, [self.candidate_waypoints, self.costs, self.uncertainties])
            grads_x = tape.gradient(waypoint[0], self.uncertainties)  # (10,)
        
        with tf.GradientTape() as tape:
            # Take the weighted average of the candidate waypoints
            weights = self.costs + self.theta * self.uncertainties
            inverse_weights = tf.reshape(tf.divide(1.0, weights), [self.len_costmap, 1])
            numerator = tf.matmul(tf.transpose(self.candidate_waypoints), inverse_weights)
            denominator = tf.reduce_sum(inverse_weights)
            waypoint = tf.divide(numerator, denominator)
        
            grads_y = tape.gradient(waypoint[1], self.uncertainties)  # (10,)
        
        with tf.GradientTape() as tape:
            # Take the weighted average of the candidate waypoints
            weights = self.costs + self.theta * self.uncertainties
            inverse_weights = tf.reshape(tf.divide(1.0, weights), [self.len_costmap, 1])
            numerator = tf.matmul(tf.transpose(self.candidate_waypoints), inverse_weights)
            denominator = tf.reduce_sum(inverse_weights)
            waypoint = tf.divide(numerator, denominator)
        
            grads_theta = tape.gradient(waypoint[2], self.uncertainties)  # (10,)

        jacobian = tf.stack([grads_x, grads_y, grads_theta])  # (3, 10)
        cost_grad, perturbed_waypoints, perturbed_costs = \
            self.get_gradient_cost_wrt_plan(waypoint.numpy(), start_config)
        final_grads = tf.matmul(cost_grad, jacobian)  # (1, len_costmap)
        
        if self.analytical_gradient_computation and self.analytical_i < self.max_len_analytical_file: 
            # Analytical gradient for derivative of output plan wrt uncertainty only 
            self.analytical_file.write("\n\nDouble-checking gradient calculation analytically")
            inverse_weights_squared = tf.multiply(inverse_weights, inverse_weights) 
            waypoint_repeated = tf.tile(waypoint, [1, self.len_costmap])  # [3, 10]
            difference_term = waypoint_repeated - tf.transpose(self.candidate_waypoints)  # [3, 10]
            multiplication_term = tf.divide(self.theta * inverse_weights_squared, denominator) 
            self.analytical_file.write("\nMultiplication term: " + str(multiplication_term))
            self.analytical_file.write("\nDifference term: " + str(difference_term))
            final_value = tf.multiply(difference_term, tf.squeeze(multiplication_term, axis=1))
            self.analytical_file.write("\nFinal value: " + str(final_value))

            self.analytical_i += 1  # Don't want too many elements in this file 
        
        return waypoint.numpy(), jacobian, cost_grad, final_grads, perturbed_waypoints, perturbed_costs 

    def get_data_from_pickle(self, file_num='1', batch_index=0):
        """
        From the costmap pickle files, return the start config and
        associated ground truth costmap. This is a dummy function.
        Eventually should be replaced by a call to the NN costmap model
        and/or an ExtendedSamplingCostsPlanner that will give the ground 
        truth costmap.
        """
        import pickle
        path = self.data_path + 'file' + file_num + '.pkl'
        d = pickle.load(open(path, 'rb'))
        start_configs = d['vehicle_state_nk3'][batch_index, 0, :]  # (3,)
        costmaps = d['costmap'][batch_index, :, :]  # (len_costmap, 4)

        return start_configs, costmaps
    
    def get_true_costmap(self, dummy_start_config, num_desired_waypoints, len_costmap):
        '''
        Given a start config for the robot, generates the full ground truth costmap with
        shape (num_desired_waypoints, 4) (4 for 3 + 1, where 3 is the dim of the waypoint
        and 1 is the cost). It then subsamples len_costmap amount of those waypoints 
        to provide as the NN ground truth.
        '''
        # First make the dummy_start_config a SystemConfig
        pos_nk2 = np.reshape(dummy_start_config[:2], (1, 1, 2))  
        pos_nk2 = tf.convert_to_tensor(pos_nk2, dtype=tf.float32)
        head_nk1 = np.reshape(dummy_start_config[2], (1, 1, 1))
        head_nk1 = tf.convert_to_tensor(head_nk1, dtype=tf.float32)
        dummy_start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)
        
        # data = self.sampling_planner.optimize(dummy_start_sys_config)
        # data = self.sampling_costs_planner.optimize(dummy_start_sys_config, num_desired_waypoints)

        obj_vals, data = self.eval_objective(dummy_start_sys_config)
        obj_vals = obj_vals.numpy()
        optimal_cost = np.min(obj_vals)
        optimal_cost_ind = np.argmin(obj_vals)
        # print("\n\nObj vals", obj_vals)
        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data
        waypts = waypts.position_and_heading_nk3().numpy()
        optimal_waypoint = np.squeeze(waypts[optimal_cost_ind])
        # print(waypts)
        if len(self.full_costmap_indices) == 0:
            self.full_costmap_indices = np.random.choice(len(waypts), num_desired_waypoints, replace=False)
        full_costmap = np.squeeze(waypts[self.full_costmap_indices])
        # print("Full costmap", full_costmap[:30, :])
        costs = obj_vals[self.full_costmap_indices]
        costs = np.expand_dims(costs, axis=1)
        full_costmap_n4 = np.hstack([full_costmap, costs])
        # print("Full costmap n4", full_costmap_n4[:30, :])

        if len(self.nn_costmap_subsampling_indices) == 0:
            self.nn_costmap_subsampling_indices = np.random.choice(num_desired_waypoints, len_costmap, replace=False)
        nn_costmap_true = full_costmap_n4[self.nn_costmap_subsampling_indices]

        return full_costmap_n4, nn_costmap_true, optimal_cost, optimal_waypoint

    
    def get_gradient_one_data_point(self, dummy_start_config, uncertainty_mode='random'):
        '''
        This function is given a start config for the robot and generates the 
        ground truth costmap for that config. It then simulates the neural network 
        output from that and then computes the planner gradient. 
        '''
        num_desired_waypoints = NUM_DESIRED_WAYPOINTS
        full_costmap_n4, true_costmap_n4, optimal_cost, optimal_waypoint = \
            self.get_true_costmap(dummy_start_config, num_desired_waypoints, self.len_costmap)
        
        best_cost_costmap = np.min(true_costmap_n4[:, 3])
        best_cost_ind_costmap = np.argmin(true_costmap_n4[:, 3])
        best_waypoint_costmap = true_costmap_n4[best_cost_ind_costmap, :3]

        # Simulated network output -- ground truth costs + noise
        nn_output_n4 = np.copy(true_costmap_n4)
        nn_output_n4[:, 3] = nn_output_n4[:, 3] + self.pre_determined_noise 

        uncertainties = self.get_uncertainties(uncertainty_mode, nn_costs=nn_output_n4[:, 3])
        plan, jacobian, cost_grad, final_grads, perturbed_waypoints, perturbed_costs = \
            self.planner_loss(dummy_start_config, nn_output_n4, uncertainties)
        
        plan_cost, _ = self.get_cost_of_a_waypoint(dummy_start_config, np.reshape(plan, (1,3)))
        final_grads = np.squeeze(final_grads.numpy())

        if self.one_pt_gradient and self.one_pt_gradient_i < self.max_len_one_pt_gradient_file:    
            self.one_pt_gradient_file.write("\n\nStart config: " + str(dummy_start_config))
            self.one_pt_gradient_file.write("\nTrue costmap: " + str(true_costmap_n4))
            self.one_pt_gradient_file.write("\nNN Costmap: " + str(nn_output_n4))
            self.one_pt_gradient_file.write("\nUncertainties: " + str(uncertainties))
            self.one_pt_gradient_file.write("\nPlan: " + str(plan))
            self.one_pt_gradient_file.write("\nPlan Cost: " + str(plan_cost[0]))
            self.one_pt_gradient_file.write("\nOptimal Cost: " + str(optimal_cost))
            self.one_pt_gradient_file.write("\nOptimal Plan: " + str(optimal_waypoint))
            self.one_pt_gradient_file.write("\nOptimal Cost from NN Costmap: " + str(best_cost_costmap))
            self.one_pt_gradient_file.write("\nOptimal Waypoint from NN Costmap: " + str(best_waypoint_costmap))
            self.one_pt_gradient_file.write("\nPercent Difference Between Plan Cost and Optimal Cost from NN Costmap: " +
                                                str((plan_cost - best_cost_costmap)/best_cost_costmap))
            self.one_pt_gradient_file.write("\nPlanner gradients: " + str(jacobian))
            self.one_pt_gradient_file.write("\nCost gradient: " + str(cost_grad))
            self.one_pt_gradient_file.write("\nFinal gradients: " + str(final_grads))

        return dummy_start_config, full_costmap_n4, true_costmap_n4, nn_output_n4, uncertainties, \
                plan, jacobian, cost_grad, final_grads, perturbed_waypoints, perturbed_costs, plan_cost

    # def get_gradients_dataset(self, num_data_points, per_file, num_files=70, 
    #                         uncertainty_mode='random', desired_gradient='loss_grads'):
    #     '''
    #     Calls get_gradient_one_data_point on multiple file and batch numbers
    #     num_data_points: The total number of data points the gradient 
    #                     computation is desired for
    #     per_file: How many data points should come from each file
    #     num_files: The total number of files available
    #     uncertainty_mode: The type of hand-coded uncertainty scheme
    #     '''
    #     losses = []
    #     gradients_list = []
    #     gradient_norms = []
    #     gradient_on_gts = []
    #     gradient_on_non_gts = []
    #     num_files_to_sample_from = int(num_data_points/per_file) 
    #     file_indices = np.random.choice(np.arange(1, num_files + 1), num_files_to_sample_from, replace=False)
    #     num_batches = 1000   # hard-coded value -- saves the effort of loading every pickle file to check
    #     for file_index in file_indices:
    #         batch_indices = np.random.choice(num_batches, per_file, replace=False)
    #         for batch_index in batch_indices:
    #             _, _, _, _, _, loss, gradients, norm_of_grad_uncertainty = \
    #                 self.get_gradient_one_data_point(str(file_index), batch_index, \
    #                                                 uncertainty_mode, desired_gradient)
    #             # Assumes that ground truth index is -1!
    #             losses.append(loss.numpy())
    #             gradients_list.append(gradients[-1])  # Gradient wrt uncertainty
    #             gradient_norms.append(norm_of_grad_uncertainty)
    #             gradient_on_gts.append(gradients[-1][-1]) # Gradient wrt uncertainty, ground truth index
    #             gradient_on_non_gts.append(np.linalg.norm(gradients[-1][:-1]))
        
    #     return losses, gradients_list, gradient_norms, gradient_on_gts, gradient_on_non_gts

    def optimize(self, start_config):
        """ 
        Expects that the neural network will output a set of spline
        trajectory samples (represented by their endpoints), their 
        corresponding costs, and uncertainties. The planner incorporates 
        this cost and uncertainty to choose a plan. 
        """
        # For now, start_config is unused!

        dummy_sc = DUMMY_SC 
        dummy_start_config, full_costmap_n4, true_costmap_n4, nn_output_n4, uncertainties, \
                plan, gradients, cost_grad, final_grads, perturbed_waypoints, perturbed_costs, plan_cost = \
                self.get_gradient_one_data_point(dummy_start_config=dummy_sc)

        perturbed_costs = list(np.stack([cost[0] for cost in perturbed_costs]))
        additional_waypoints = perturbed_waypoints.copy()
        additional_waypoints.extend(full_costmap_n4[:, :3])
        additional_costs = perturbed_costs.copy()
        additional_costs.extend(np.squeeze(full_costmap_n4[:, 3]))

        display_uncertainties = (len(full_costmap_n4) <= 5*self.len_costmap)  # too many points to display
        self.visualize_waypoints(dummy_start_config, nn_output_n4[:, :3], uncertainties, plan,
                                additional_waypoints, additional_costs, final_grads, display_uncertainties)
        
        # Visualize gradient wrt uncertainty
        # self.visualize_gradients(true_costmap_n4[:, 3], nn_output_n4[:, 3], uncertainties, 
        #                             planner_internal_costs, gradients[-1])
        
        # num_data_points = 1000
        # per_file = 50
        # losses, gradients_list, gradient_norms, gradient_on_gts, gradient_on_non_gts = \
        #                     self.get_gradients_dataset(num_data_points, per_file, \
        #                                                 uncertainty_mode='proportional_to_cost')

        # self.visualize_dataset_gradients(losses, gradients_list, gradient_norms, 
        #                                 gradient_on_gts, gradient_on_non_gts)

        
        # Get the optimal trajectory
        obj_val, data = self.get_cost_of_a_waypoint(dummy_start_config, np.reshape(plan, (1,3)))
        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        idx = 0  # 0 bc there's only one waypoint
        self.opt_waypt.assign_from_config_batch_idx(waypts, idx)  # Just here for debugging
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, idx)  

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[idx, 0]/self.params.dt).numpy())

        # To output in the metadata
        pos_nk2 = np.reshape(dummy_start_config[:2], (1, 1, 2))  
        head_nk1 = np.reshape(dummy_start_config[2], (1, 1, 1))
        dummy_start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)
        pos_nk2 = np.reshape(plan[:2, :], (1, 1, 2))  
        pos_nk2 = tf.convert_to_tensor(pos_nk2, dtype=tf.float32)
        head_nk1 = np.reshape(plan[2, :], (1, 1, 1))  
        head_nk1 = tf.convert_to_tensor(head_nk1, dtype=tf.float32)
        plan_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                                position_nk2=pos_nk2,
                                                heading_nk1=head_nk1)

        data = {'system_config': dummy_start_sys_config,
                'waypoint_config': SystemConfig.copy(plan_config),
                'planner_gradients': tf.cast(final_grads, dtype=tf.double),
                'trajectory': Trajectory.copy(self.opt_traj),
                'cost_of_trajectory': obj_val,
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][idx:idx + 1],
                'k_nkf1': controllers['k_nkf1'][idx:idx + 1]}

        return data
    
    def visualize_waypoints(self, start_config, costmap, uncertainties, plan,
                            additional_waypoints, additional_costs, gradients, 
                            display_uncertainties=True, display_gradients=DISPLAY_GRADIENTS):
        '''
        Plot a heatmap-style plot of various candidate waypoints and the plan
        provided by the planner along with associated uncertainties and costs.
        start_config: The current pose of the robot, needed to compute costs
        costmap: (len_costmap, 3) The candidate waypoints provided by the NN
        uncertainties: The uncertainties of the above candidate waypoints
        plan: A waypoint. Provided by the planner
        additional_waypoints
        additional_costs (of the above additional_waypoints)
        '''
        plan = np.reshape(plan, (1,3))
        additional_waypoints = [np.reshape(waypoint, (3,)) for waypoint in additional_waypoints]
        additional_waypoints = np.stack(additional_waypoints)

        # First get the costs of the candidate waypoints
        candidate_waypoints_costs = []
        for waypoint in costmap:
            obj_val, _ = self.get_cost_of_a_waypoint(start_config, np.reshape(waypoint, (1,3)))
            candidate_waypoints_costs.append(obj_val[0])
        
        plan_cost, _ = self.get_cost_of_a_waypoint(start_config, plan)

        candidate_waypoints_costs = np.stack(candidate_waypoints_costs)
        plan_cost = plan_cost[0]

        total_waypoints = np.vstack([costmap, plan, additional_waypoints])
        total_costs = np.hstack([candidate_waypoints_costs, plan_cost, additional_costs])
        # print("Total costs", total_costs)

        viridis = cm.get_cmap('coolwarm', len(total_costs))
        sorted_costs = np.sort(total_costs)
        sorted_costs_small = sorted_costs[sorted_costs < 100]
        normalized_sorted_costs = np.copy(sorted_costs)
        #normalized_sorted_costs[sorted_costs < 100] /= sum(sorted_costs_small)
        ind_sorted_costs = np.argsort(total_costs)
        sorted_waypoints = total_waypoints[ind_sorted_costs]

        plt.figure(figsize=(10, 11))
        plt.plot(start_config[0], start_config[1], color='g', marker='o', markersize=20)
        for i in range(len(sorted_waypoints)):
            waypoint = sorted_waypoints[i]
            if sorted_costs[i] > 100:
                plt.plot(waypoint[0], waypoint[1], color='k', marker='o')
            else:
                plt.plot(waypoint[0], waypoint[1], color=viridis(normalized_sorted_costs[i]), marker='o')
        if display_uncertainties:
            # ax = plt.axes()
            # plt.axis([0, 3, 0, 3])
            # circle = plt.Circle((1.0, 2.0), 0.5, fill = False)
            # plt.gcf().add_patch(circle)
            # ax.add_artist(circle)
            # ax.autoscale()
            for i in range(len(costmap)):
                waypoint = costmap[i]
                uncertainty = uncertainties[i].numpy()
                uncertainty_gradient = gradients[i]
                # Plot a circle
                angle = np.linspace(0, 2 * np.pi, 150) 
                radius = uncertainty
                to_adjust_radius = uncertainty - uncertainty_gradient
                x = radius * np.cos(angle) + waypoint[0]
                y = radius * np.sin(angle) + waypoint[1]
                plt.plot(x, y, 'k')
                adjust_radius_clip_value = 0.75
                if display_gradients and to_adjust_radius > 0:
                    to_adjust_radius = min(to_adjust_radius, adjust_radius_clip_value)
                    x = to_adjust_radius * np.cos(angle) + waypoint[0]
                    y = to_adjust_radius * np.sin(angle) + waypoint[1]
                    plt.plot(x, y, 'g')
                # circle = plt.Circle((waypoint[0], waypoint[1]), 0.25, fill = False)
                # plt.gca().add_patch(circle)
        plt.savefig('waypoints_heatmap.png')

    def visualize_gradients(self, true_costmap, nn_costmap, uncertainties, 
                            planner_internal_costs, uncertainty_gradients):
        true_costmap = np.clip(true_costmap, 0, self.plotting_clip_value)
        nn_costmap = np.clip(nn_costmap, 0, self.plotting_clip_value)
        planner_internal_costs = np.clip(planner_internal_costs, 0, self.plotting_clip_value)

        indices = np.arange(self.len_costmap)

        fontsize = 20
        plt.rc('font', size=fontsize)          # controls default text sizes
        plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
        plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
        plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title

        plt.figure(figsize=(10, 11))
        plt.bar(indices, true_costmap)
        plt.title("True Costmap")
        plt.xlabel("Indices")
        plt.ylabel("Costs")
        plt.savefig('true_costmap.png')

        plt.figure(figsize=(10, 11))
        plt.bar(indices, nn_costmap)
        plt.title("NN Costmap")
        plt.xlabel("Indices")
        plt.ylabel("Costs")
        plt.savefig('nn_costmap.png')

        plt.figure(figsize=(10, 11))
        plt.bar(indices, planner_internal_costs)
        plt.title("Planner Internal Costs")
        plt.xlabel("Indices")
        plt.ylabel("Costs")
        plt.savefig('planner_internal_costs.png')

        plt.figure(figsize=(10, 11))
        plt.bar(indices, uncertainties)
        plt.title("Uncertainties")
        plt.xlabel("Indices")
        plt.ylabel("Values")
        plt.savefig('uncertainties.png')

        plt.figure(figsize=(10, 11))
        plt.bar(indices, uncertainty_gradients)
        plt.title("Gradients of Loss wrt Uncertainties")
        plt.xlabel("Indices")
        plt.ylabel("Values")
        plt.savefig('uncertainty_gradients.png')
    
    def visualize_dataset_gradients(self, losses, gradients_list, gradient_norms, gradient_on_gts, 
                                    gradient_on_non_gts):
        
        plt.figure(figsize=(10, 11))
        plt.hist(losses)
        plt.title("Losses")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('losses.png')

        all_ind_grad_mean = np.mean(gradients_list, axis=0)
        all_ind_grad_std = np.std(gradients_list, axis=0)

        plt.figure(figsize=(10, 11))
        plt.bar(np.arange(len(all_ind_grad_mean)), all_ind_grad_mean)
        plt.title("Gradients Per Index, Mean")
        plt.xlabel("Indices")
        plt.ylabel("Gradient Mean")
        plt.savefig('all_ind_grad_mean.png')

        plt.figure(figsize=(10, 11))
        plt.bar(np.arange(len(all_ind_grad_mean)), all_ind_grad_std)
        plt.title("Gradients Per Index, Standard Deviation")
        plt.xlabel("Indices")
        plt.ylabel("Gradient Std")
        plt.savefig('all_ind_grad_std.png')

        plt.figure(figsize=(10, 11))
        plt.hist(gradient_norms)
        plt.title("Gradient Norms")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('gradient_norms.png')

        plt.figure(figsize=(10, 11))
        plt.hist(gradient_on_gts)
        plt.title("Gradients on Ground Truth")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('gradient_on_gt.png')

        plt.figure(figsize=(10, 11))
        plt.hist(gradient_on_non_gts)
        plt.title("Gradients on Non Ground Truth")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('gradient_on_non_gts.png')

        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        grad_mean = np.mean(gradient_norms)
        grad_std = np.std(gradient_norms)
        gradient_on_gts_mean = np.mean(gradient_on_gts)
        gradient_on_gts_std = np.std(gradient_on_gts)
        gradient_on_non_gts_mean = np.mean(gradient_on_non_gts)
        gradient_on_non_gts_std = np.std(gradient_on_non_gts)
        print("\nLOSS MEAN", loss_mean)
        print("\nLOSS STD", loss_std)
        print("\nALL INDICES GRADIENT MEAN", all_ind_grad_mean)
        print("\nALL INDICES GRADIENT STD", all_ind_grad_std)
        print("\nGRADIENT MEAN", grad_mean)
        print("\nGRADIENT STD", grad_std)
        print("\nGRADIENT ON GT MEAN", gradient_on_gts_mean)
        print("\nGRADIENT ON GT STD", gradient_on_gts_std)
        print("\nGRADIENT ON NON GTS MEAN", gradient_on_non_gts_mean)
        print("\nGRADIENT ON NON GTS STD", gradient_on_non_gts_std)

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'system_config': [],
                'waypoint_config': [],
                'planner_gradients': [],
                'trajectory': [],
                'cost_of_trajectory': [],
                'spline_trajectory': [],
                'planning_horizon': [],
                'K_nkfd': [],
                'k_nkf1': []}
        return data
    
    @staticmethod
    def mask_and_concat_data_along_batch_dim(data, k):
        """Keeps the elements in data which were produced
        before time index k. Concatenates each list in data
        along the batch dim after masking. Also returns data
        from the first segment not in the valid mask."""

        # Extract the Index of the Last Data Segment
        data_times = np.cumsum([traj.k for traj in data['trajectory']])
        valid_mask = (data_times <= k)
        data_last = {}
        last_data_idxs = np.where(np.logical_not(valid_mask))[0]

        # Take the first last_data_idx
        if len(last_data_idxs) > 0:
            last_data_idx = last_data_idxs[0]
            last_data_valid = True
        else:
            # Take the last element as it is not valid anyway
            last_data_idx = len(valid_mask) - 1
            last_data_valid = False

        # Get the last segment data
        data_last['system_config'] = data['system_config'][last_data_idx]
        data_last['waypoint_config'] = data['waypoint_config'][last_data_idx]
        data_last['planner_gradients'] = data['planner_gradients'][last_data_idx]
        data_last['trajectory'] = data['trajectory'][last_data_idx]
        data_last['cost_of_trajectory'] = data['cost_of_trajectory'][last_data_idx]
        data_last['spline_trajectory'] = data['spline_trajectory'][last_data_idx]
        data_last['planning_horizon_n1'] = [data['planning_horizon'][last_data_idx]] 
        data_last['K_nkfd'] = data['K_nkfd'][last_data_idx]
        data_last['k_nkf1'] = data['k_nkf1'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['planner_gradients'] = tf.boolean_mask(tf.concat(data['planner_gradients'], axis=0), valid_mask)
        data['trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['trajectory'])[valid_mask])
        data['cost_of_trajectory'] = np.array(data['cost_of_trajectory'])[valid_mask]
        data['spline_trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['spline_trajectory'])[valid_mask])
        data['planning_horizon_n1'] = np.array(data['planning_horizon'])[valid_mask][:, None]
        data['K_nkfd'] = tf.boolean_mask(tf.concat(data['K_nkfd'], axis=0), valid_mask)
        data['k_nkf1'] = tf.boolean_mask(tf.concat(data['k_nkf1'], axis=0), valid_mask)
        return data, data_last, last_data_valid
    
    @staticmethod
    def convert_planner_data_to_numpy_repr(data):
        """
        Convert any tensors into numpy arrays in a
        planner data dictionary.
        """
        if len(data.keys()) == 0:
            return data
        data_numpy = {}
        data_numpy['system_config'] = data['system_config'].to_numpy_repr()
        data_numpy['waypoint_config'] = data['waypoint_config'].to_numpy_repr()
        data_numpy['planner_gradients'] = data['planner_gradients']
        data_numpy['trajectory'] = data['trajectory'].to_numpy_repr()
        data_numpy['cost_of_trajectory'] = data['cost_of_trajectory']
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()

        return data_numpy
