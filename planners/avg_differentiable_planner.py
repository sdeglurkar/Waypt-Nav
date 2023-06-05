from math import exp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


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
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)
        self.uncertainty_amount = 1.0
        self.noise = 0.0

        tfe = tf.contrib.eager
        self.costs = tfe.Variable(np.zeros((self.len_costmap)))
        self.uncertainties = tfe.Variable(np.zeros((self.len_costmap)))
        self.candidate_waypoints = tfe.Variable(np.zeros((self.len_costmap, 3)))

        self.pre_determined_uncertainties = \
            tf.random_uniform([self.len_costmap], dtype=tf.double) * self.uncertainty_amount
        self.pre_determined_noise = np.random.rand(self.len_costmap)*self.noise

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
        # Get the SystemConfig for the start_config
        pos_nk2 = np.reshape(start_config[:2], (1, 1, 2))  
        head_nk1 = np.reshape(start_config[2], (1, 1, 1))
        start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)

        # Generate all the perturbations to 'waypoint' -- in x, y, and theta
        deltas = [np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]]
        deltas = [self.finite_differencing_delta * elem for elem in deltas] 
        perturbed_waypoints = []
        for delta in deltas:
            perturbed_waypoints.append(waypoint + delta)
            perturbed_waypoints.append(waypoint - delta)
        
        # Generate the costs for all perturbations
        perturbed_costs = []
        for perturbed_waypoint in perturbed_waypoints:
            pos_nk2 = np.reshape(perturbed_waypoint[:2, :], (1, 1, 2))  
            head_nk1 = np.reshape(perturbed_waypoint[2, :], (1, 1, 1))  
            perturbed_waypoint_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                                    position_nk2=pos_nk2,
                                                    heading_nk1=head_nk1)
            perturbed_waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)
            self.params.system_dynamics.to_world_coordinates(start_sys_config,
                                                         perturbed_waypoint_config,
                                                         perturbed_waypoint_world_config)
            obj_val, data = self.eval_objective(start_sys_config, perturbed_waypoint_world_config)
            # waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data
            perturbed_costs.append(obj_val.numpy())

        cost_grads = []
        i = 0
        while i < len(perturbed_costs):  # Should be 6 elements in perturbed_costs
            numerator = perturbed_costs[i + 1] - perturbed_costs[i]
            denominator = self.finite_differencing_delta * 2
            cost_grads.append(numerator/denominator)
            i += 2
        
        cost_grad = tf.convert_to_tensor(cost_grads)  # [1, 3]
        return cost_grad

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
        to the plan.
        '''
        with tf.GradientTape() as tape:
            self.costs.assign(costmap)
            self.uncertainties.assign(uncertainties)
            self.candidate_waypoints.assign(costmap[:, :3])

            # Take the weighted average of the candidate waypoints
            weights = self.costs + self.theta * self.uncertainties
            inverse_weights = tf.divide(1.0, weights)
            numerator = tf.matmul(tf.transpose(self.candidate_waypoints), inverse_weights)
            denominator = tf.reduce_sum(inverse_weights)
            waypoint = tf.divide(numerator, denominator)

            grads = tape.gradient(waypoint, [self.candidate_waypoints, self.costs, self.uncertainties])
            cost_grad = self.get_gradient_cost_wrt_plan(waypoint.numpy(), start_config)
            final_grads = tf.matmul(cost_grad, grads[-1])
        
        if self.analytical_gradient_computation and self.analytical_i < self.max_len_analytical_file: 
            # Analytical gradient for derivative of output plan wrt uncertainty only 
            self.analytical_file.write("\n\nDouble-checking gradient calculation analytically")
            self.analytical_file.write("\nEach row of the Jacobian should be the same")
            inverse_weights_squared = tf.divide(1.0, inverse_weights) 
            waypoint_repeated = tf.tile(waypoint, [1, self.len_costmap])
            difference_term = waypoint_repeated - tf.transpose(self.candidate_waypoints)
            multiplication_term = tf.divide(self.theta * inverse_weights_squared, denominator) 
            self.analytical_file.write("\nMultiplication term: " + str(multiplication_term))
            self.analytical_file.write("\nDifference term: " + str(difference_term))
            final_value = tf.multiply(difference_term, multiplication_term)
            self.analytical_file.write("\nFinal value: " + str(final_value))

            self.analytical_i += 1  # Don't want too many elements in this file 
        
        return waypoint.numpy(), grads, cost_grad, final_grads 

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
    
    def get_gradient_one_data_point(self, file_num='1', batch_index=0, 
                                    uncertainty_mode='random'):
        '''
        This function is given one data point from the dataset of ground 
        truth costmaps. It simulates the neural network output from that
        and then computes the planner gradient. This is a dummy implementation;
        eventually the simulated NN output should be replaced by the real NN
        output.
        '''
        dummy_start_config, true_costmap_n4 = self.get_data_from_pickle(file_num, batch_index)

        # Simulated network output -- ground truth costs + noise
        nn_output_n4 = np.copy(true_costmap_n4)
        nn_output_n4[:, 3] = nn_output_n4[:, 3] + self.pre_determined_noise 

        uncertainties = self.get_uncertainties(uncertainty_mode, nn_costs=nn_output_n4[:, 3])
        plan, grads, cost_grad, final_grads = \
            self.planner_loss(dummy_start_config, nn_output_n4, uncertainties)
        
        gradients = np.array([np.array(elem) for elem in grads])

        if self.one_pt_gradient and self.one_pt_gradient_i < self.max_len_one_pt_gradient_file:    
            self.one_pt_gradient_file.write("\n\nGot data from pickle: Start config: " + \
                                            str(dummy_start_config))
            self.one_pt_gradient_file.write("\nTrue costmap: " + str(true_costmap_n4))
            self.one_pt_gradient_file.write("\nNN Costmap: " + str(nn_output_n4))
            self.one_pt_gradient_file.write("\nUncertainties: " + str(uncertainties))
            self.one_pt_gradient_file.write("\nPlan: " + str(plan))
            self.one_pt_gradient_file.write("\nPlanner gradients: " + str(gradients))
            self.one_pt_gradient_file.write("\nCost gradient: " + str(cost_grad))
            self.one_pt_gradient_file.write("\nFinal gradients: " + str(final_grads))

        return dummy_start_config, true_costmap_n4, nn_output_n4, uncertainties, \
                plan, gradients, cost_grad, final_grads

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
        corresponding costs, and uncertainties. Incorporates this cost 
        and uncertainty into the planner internal cost function and 
        chooses the trajectory with the lowest planner internal cost.
        """
        # For now, start_config is unused!

        dummy_start_config, true_costmap_n4, nn_output_n4, uncertainties, \
                plan, gradients, cost_grad, final_grads = self.get_gradient_one_data_point()
        
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
        # First transform waypoints to World Coordinates
        pos_nk2 = np.reshape(plan[:2, :], (1, 1, 2))  
        head_nk1 = np.reshape(plan[2, :], (1, 1, 1))  
        plan_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                                position_nk2=pos_nk2,
                                                heading_nk1=head_nk1)
        pos_nk2 = np.reshape(dummy_start_config[:2], (1, 1, 2))  
        head_nk1 = np.reshape(dummy_start_config[2], (1, 1, 1))
        dummy_start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)
        self.params.system_dynamics.to_world_coordinates(dummy_start_sys_config,
                                                         plan_config,
                                                         self.waypoint_world_config)

        # Now retrieve Control Pipeline data
        _, data = self.eval_objective(dummy_start_sys_config, self.waypoint_world_config)
        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        idx = 0  # 0 bc there's only one waypoint
        self.opt_waypt.assign_from_config_batch_idx(waypts, idx)  # Just here for debugging
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, idx)  

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[idx, 0]/self.params.dt).numpy())

        
        data = {'system_config': dummy_start_sys_config,
                'waypoint_config': SystemConfig.copy(plan_config),
                'planner_gradients': final_grads,
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][idx:idx + 1],
                'k_nkf1': controllers['k_nkf1'][idx:idx + 1]}

        return data

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
        data_last['spline_trajectory'] = data['spline_trajectory'][last_data_idx]
        data_last['planning_horizon_n1'] = [data['planning_horizon'][last_data_idx]] 
        data_last['K_nkfd'] = data['K_nkfd'][last_data_idx]
        data_last['k_nkf1'] = data['k_nkf1'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['planner_gradients'] = np.array(data['planner_gradients'])[valid_mask]
        data['trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['trajectory'])[valid_mask])
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
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()

        return data_numpy
