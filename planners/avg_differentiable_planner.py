from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from planners.nn_planner import NNPlanner
import sys 
import tensorflow as tf
import time
from trajectory.trajectory import Trajectory, SystemConfig

NUM_DESIRED_WAYPOINTS = 1000  # How many points in the costmap to visualize for 1 robot pose
DUMMY_SC = [8.5, 10.5, 0.0] # Dummy start config -- getting the gradient at 1 data point
SIZE_DATASET = 1000  # Number of points to evaluate gradients for
OBSTACLE_COST = 100  # Defining the cost of an obstacle 
FAILED_PLAN_THRES = 1000  # Don't analyze criticalities for "failure" points
DISPLAY_MULT = 5  # Multiplier defining how many waypoints can be visualized
PER_POINT_VIZ = 35  # If size dataset is < this value, some debugging is done
IMAGE_PATH = 'images/'
TXT_PATH = 'txt/'
MAX_LEN_TXT_FILE = 100


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

        # For the optimal waypt chosen
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)  
        
        # Uncertainty and noise schemes
        self.uncertainty_amount = self.params.diff_planner_uncertainty_amount
        self.noise = self.params.diff_planner_noise_amount
        self.pre_determined_uncertainties = \
            tf.random_uniform([self.len_costmap], dtype=tf.double) * self.uncertainty_amount
        self.pre_determined_noise = np.random.rand(self.len_costmap)*self.noise

        # For taking gradients of the planner wrt uncertainty using Tensorflow
        tfe = tf.contrib.eager
        self.costs = tfe.Variable(np.zeros((self.len_costmap)))
        self.uncertainties = tfe.Variable(np.zeros((self.len_costmap)))
        self.candidate_waypoints = tfe.Variable(np.zeros((self.len_costmap, 3)))

        # Controlling the size of the extended costmap of the environment 
        self.full_costmap_indices = []
        # Part of that costmap will be sampled to get the ground truth NN costmap
        self.nn_costmap_subsampling_indices = []
        # Controlling the location of the 10 points in the NN costmap in terms of vectors: 
        # (angle from robot, length of vector)
        self.nn_costmap_point_locs = [(np.pi/4, 0.4), (-np.pi/4, 0.4), 
                                        (np.pi/4, 0.8), (0, 0.8), (-np.pi/4, 0.8), 
                                        (np.pi/6, 1.35), (0, 1.35), (-np.pi/6, 1.35), 
                                        (np.pi/18, 1.75), (-np.pi/18, 1.75)]
        self.use_controlled_locations = True

        # Used for one of the task criticality metrics -- radius within which
        # points are defined to be close to the robot
        self.close_distance = 0.6

        # Printing to files
        self.analytical_gradient_computation = True 
        if self.analytical_gradient_computation:
            try:  # If file exists, clear it 
                f = open(TXT_PATH + "analytical_gradient.txt")
                f.close()
                f = open(TXT_PATH + "analytical_gradient.txt", "w")  # Clear contents of file
            except FileNotFoundError:
                f = open(TXT_PATH + "analytical_gradient.txt","a")  # Create file
            
            self.analytical_file = f
            self.analytical_i = 0
            self.max_len_analytical_file = MAX_LEN_TXT_FILE
        
        self.one_pt_gradient = True 
        if self.one_pt_gradient:
            try:  # If file exists, clear it 
                f = open(TXT_PATH + "one_pt_gradient.txt")
                f.close()
                f = open(TXT_PATH + "one_pt_gradient.txt", "w")  # Clear contents of file
            except FileNotFoundError:
                f = open(TXT_PATH + "one_pt_gradient.txt","a")  # Create file
            
            self.one_pt_gradient_file = f
            self.one_pt_gradient_i = 0
            self.max_len_one_pt_gradient_file = MAX_LEN_TXT_FILE
        
        self.dataset_gradients = True 
        if self.dataset_gradients:
            try:  # If file exists, clear it 
                f = open(TXT_PATH + "dataset_gradients.txt")
                f.close()
                f = open(TXT_PATH + "dataset_gradients.txt", "w")  # Clear contents of file
            except FileNotFoundError:
                f = open(TXT_PATH + "dataset_gradients.txt","a")  # Create file
            
            self.dataset_gradients_file = f
    
    def convert_state_arr_to_config(self, state):
        '''
        Helper function: Create the SystemConfig object from this
        3D state array [x, y, theta].
        '''
        pos_nk2 = np.reshape(state[:2], (1, 1, 2)) 
        pos_nk2 = tf.convert_to_tensor(pos_nk2, dtype=tf.float32) 
        head_nk1 = np.reshape(state[2], (1, 1, 1))
        head_nk1 = tf.convert_to_tensor(head_nk1, dtype=tf.float32)
        state_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                        position_nk2=pos_nk2,
                                        heading_nk1=head_nk1)
        return state_config

    def get_uncertainties(self, mode='random', nn_costs=None):
        '''
        For now a dummy function that will just output some hand-coded numbers, 
        should later be a function defined by the model.
        '''
        if mode == 'random':
            uncertainties = self.pre_determined_uncertainties
        elif mode == 'uniform':
            uncertainties = \
                tf.constant(np.ones(self.len_costmap), dtype=tf.double)
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
        waypoint, which is in world coordinates and is also
        an array
        '''
        start_sys_config = self.convert_state_arr_to_config(start_config)
        waypoint_config = self.convert_state_arr_to_config(waypoint[0])
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
        waypoint = np.reshape(waypoint, (1, 3))
        # Generate all the perturbations to 'waypoint' -- in x, y, and theta
        deltas = self.finite_differencing_delta * np.eye(3)
        perturbed_waypoints = []
        for delta in deltas:
            perturbed_waypoints.append(waypoint - delta)
            perturbed_waypoints.append(waypoint + delta)
        
        # Generate the costs for all perturbations
        perturbed_costs = []
        for perturbed_waypoint in perturbed_waypoints:
            obj_val, _ = self.get_cost_of_a_waypoint(start_config, perturbed_waypoint)
            perturbed_costs.append(obj_val)

        cost_grads = []
        i = 0
        while i < len(perturbed_costs):  # Should be 6 elements in perturbed_costs
            numerator = perturbed_costs[i + 1] - perturbed_costs[i]
            denominator = self.finite_differencing_delta * 2
            cost_grads.append(numerator/denominator)
            i += 2
        
        cost_grad = tf.convert_to_tensor(np.reshape(cost_grads, (1, 3)), dtype=tf.double)  # [1, 3]

        # Also compute a percent cost gradient in case that is useful later
        plan_cost, _ = self.get_cost_of_a_waypoint(start_config, waypoint)
        plan_cost = plan_cost[0]

        percent_cost_grad = []
        i = 0
        while i < len(perturbed_costs):  # Should be 6 elements in perturbed_costs
            difference = perturbed_costs[i + 1] - perturbed_costs[i]
            percent = difference/plan_cost
            percent_cost_grad.append(percent[0])
            i += 2

        return cost_grad, perturbed_waypoints, perturbed_costs, percent_cost_grad, plan_cost

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
        cost_grad, perturbed_waypoints, perturbed_costs, percent_cost_grad, plan_cost = \
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
        
        return waypoint.numpy(), jacobian, cost_grad, percent_cost_grad, final_grads, \
                perturbed_waypoints, perturbed_costs, plan_cost 
    
    def get_true_costmap(self, dummy_start_config, num_desired_waypoints, len_costmap):
        '''
        Given a start config for the robot, generates the full ground truth costmap with
        shape (num_desired_waypoints, 4) (4 for 3 + 1, where 3 is the dim of the waypoint
        and 1 is the cost). It then subsamples len_costmap amount of those waypoints 
        to provide as the NN ground truth.
        '''
        dummy_start_sys_config = self.convert_state_arr_to_config(dummy_start_config)
        obj_vals, data = self.eval_objective(dummy_start_sys_config)
        obj_vals = obj_vals.numpy()
        waypts, _, _, _, _ = data
        optimal_cost = np.min(obj_vals)
        optimal_cost_ind = np.argmin(obj_vals)
        optimal_waypoint = np.squeeze(waypts[optimal_cost_ind])

        # What percent of these costs are obstacle costs?
        obstacle_obj_vals = obj_vals[np.where(obj_vals > OBSTACLE_COST)]
        percent_obstacle = len(obstacle_obj_vals)/len(obj_vals)
        
        # What percent of costs for points near the robot are obstacle costs?
        waypts = waypts.position_and_heading_nk3().numpy()
        distances_from_start = np.linalg.norm(np.squeeze(waypts)[:, :2] -  \
                                np.reshape(dummy_start_config[:2], (1, 2)), axis=1)
        close_points_ind = np.where(distances_from_start < self.close_distance)
        close_points_obj_vals = obj_vals[close_points_ind]  # Costs of nearby points
        obstacle_obj_vals_close = close_points_obj_vals[np.where(close_points_obj_vals > OBSTACLE_COST)]
        percent_obstacle_close = len(obstacle_obj_vals_close)/len(close_points_obj_vals)
        
        if len(self.full_costmap_indices) == 0:  # Generate self.full_costmap_indices once
            self.full_costmap_indices = np.random.choice(len(waypts), num_desired_waypoints, replace=False)
        full_costmap = np.squeeze(waypts[self.full_costmap_indices])
        costs = obj_vals[self.full_costmap_indices]
        costs = np.expand_dims(costs, axis=1)
        full_costmap_n4 = np.hstack([full_costmap, costs])

        if self.use_controlled_locations: 
            nn_costmap_true = []
            for elem in self.nn_costmap_point_locs:
                angle = elem[0] + dummy_start_config[2]
                length = elem[1]
                vector = length * np.array([np.cos(angle), np.sin(angle), 0])
                loc = dummy_start_config + vector
                loc[2] = angle
                loc_sys_config = self.convert_state_arr_to_config(loc)
                cost, _ = self.eval_objective(dummy_start_sys_config, loc_sys_config)
                cost = cost.numpy()
                point = np.hstack([loc, cost])
                nn_costmap_true.append(point) 
            nn_costmap_true = np.stack(nn_costmap_true)
            # For visualization reasons
            if num_desired_waypoints == self.len_costmap:
                full_costmap_n4 = nn_costmap_true  
        else: # Get the NN ground truth points randomly
            if len(self.nn_costmap_subsampling_indices) == 0:  
                # Generate self.nn_costmap_subsampling_indices once
                self.nn_costmap_subsampling_indices = \
                    np.random.choice(num_desired_waypoints, len_costmap, replace=False)
            nn_costmap_true = full_costmap_n4[self.nn_costmap_subsampling_indices]

        return_dict = {'full_costmap_n4': full_costmap_n4,
                        'nn_costmap_true': nn_costmap_true,
                        'optimal_cost': optimal_cost,
                        'optimal_waypoint': optimal_waypoint,
                        'percent_obstacle': percent_obstacle,
                        'percent_obstacle_close': percent_obstacle_close
                        }
        return return_dict 

    def get_gradient_one_data_point(self, dummy_start_config, 
                                    num_desired_waypoints=NUM_DESIRED_WAYPOINTS, 
                                    uncertainty_mode='random'):
        '''
        This function is given a start config for the robot and generates the 
        ground truth costmap for that config. It then simulates the neural network 
        output from that and then computes the planner gradient. 
        '''
        data = self.get_true_costmap(dummy_start_config, num_desired_waypoints, self.len_costmap)
        full_costmap_n4 = data['full_costmap_n4'] 
        true_costmap_n4 = data['nn_costmap_true'] 
        optimal_cost = data['optimal_cost'] 
        optimal_waypoint = data['optimal_waypoint']
        percent_obstacle = data['percent_obstacle']
        percent_obstacle_close = data['percent_obstacle_close']
        
        # The optimal waypoint in the NN ground truth costmap
        best_cost_costmap = np.min(true_costmap_n4[:, 3])
        best_cost_ind_costmap = np.argmin(true_costmap_n4[:, 3])
        best_waypoint_costmap = true_costmap_n4[best_cost_ind_costmap, :3]

        # Simulated network output -- ground truth costs + noise
        nn_output_n4 = np.copy(true_costmap_n4)
        nn_output_n4[:, 3] = nn_output_n4[:, 3] + self.pre_determined_noise 

        uncertainties = self.get_uncertainties(uncertainty_mode, nn_costs=nn_output_n4[:, 3])
        plan, jacobian, cost_grad, percent_cost_grad, final_grads, \
            perturbed_waypoints, perturbed_costs, plan_cost = \
            self.planner_loss(dummy_start_config, nn_output_n4, uncertainties)

        if self.one_pt_gradient and self.one_pt_gradient_i < self.max_len_one_pt_gradient_file:    
            self.one_pt_gradient_file.write("\n\nStart config: " + str(dummy_start_config))
            self.one_pt_gradient_file.write("\nTrue costmap: " + str(true_costmap_n4))
            self.one_pt_gradient_file.write("\nNN Costmap: " + str(nn_output_n4))
            self.one_pt_gradient_file.write("\nUncertainties: " + str(uncertainties))
            self.one_pt_gradient_file.write("\nPlan: " + str(plan))
            self.one_pt_gradient_file.write("\nPlan Cost: " + str(plan_cost))
            self.one_pt_gradient_file.write("\nOptimal Cost: " + str(optimal_cost))
            self.one_pt_gradient_file.write("\nOptimal Plan: " + str(optimal_waypoint))
            self.one_pt_gradient_file.write("\nOptimal Cost from NN Costmap: " + str(best_cost_costmap))
            self.one_pt_gradient_file.write("\nOptimal Waypoint from NN Costmap: " + str(best_waypoint_costmap))
            self.one_pt_gradient_file.write("\nPercent Obstacle: " + str(percent_obstacle))
            self.one_pt_gradient_file.write("\nPercent Obstacle Nearby: " + str(percent_obstacle_close))
            self.one_pt_gradient_file.write("\nPercent Difference Between Plan Cost and Optimal Cost from NN Costmap: " +
                                                str((plan_cost - best_cost_costmap)/best_cost_costmap))
            self.one_pt_gradient_file.write("\nSame as above wrt plan cost: " +
                                                str((plan_cost - best_cost_costmap)/plan_cost))
            self.one_pt_gradient_file.write("\nPercent Difference Between Plan Cost and Optimal Cost: " +
                                                str((plan_cost - optimal_cost)/optimal_cost))
            self.one_pt_gradient_file.write("\nPercent Cost Grad: " + str(percent_cost_grad))
            self.one_pt_gradient_file.write("\nPlanner gradients: " + str(jacobian))
            self.one_pt_gradient_file.write("\nCost gradient: " + str(cost_grad))
            self.one_pt_gradient_file.write("\nFinal gradients: " + str(final_grads))

        return_dict = {'dummy_start_config': dummy_start_config,
                        'full_costmap_n4': full_costmap_n4,
                        'true_costmap_n4': true_costmap_n4,
                        'nn_output_n4': nn_output_n4,
                        'uncertainties': uncertainties,
                        'plan': plan,
                        'jacobian': jacobian,
                        'cost_grad': cost_grad,
                        'final_grads': final_grads,
                        'percent_cost_grad': percent_cost_grad,
                        'perturbed_waypoints': perturbed_waypoints,
                        'perturbed_costs': perturbed_costs,
                        'plan_cost': plan_cost,
                        'optimal_cost': optimal_cost,
                        'optimal_waypoint': optimal_waypoint,
                        'percent_obstacle': percent_obstacle,
                        'percent_obstacle_close': percent_obstacle_close,
                        'best_cost_costmap': best_cost_costmap,
                        'best_waypoint_costmap': best_waypoint_costmap}
        return return_dict

    def get_gradients_dataset(self, size_dataset=SIZE_DATASET, range_x=[2,16], range_y=[6,20], 
                            fixed_theta=None, uncertainty_mode='random'):
        '''
        For size_dataset number of randomly generated locations on the map within the x- and 
        y-ranges range_x and range_y respectively, call get_gradient_one_data_point() and 
        analyze criticality and uncertainty "goodness" data. 

        fixed_theta is set to a value if all points should have the same heading. Otherwise,
        the headings are generated randomly between -pi and pi.
        '''
        def generate_pose():
            start_config_x = np.random.rand() * (range_x[1]-range_x[0]) + range_x[0]
            start_config_y = np.random.rand() * (range_y[1]-range_y[0]) + range_y[0]
            if fixed_theta is not None:
                start_config_theta = fixed_theta
            else:
                start_config_theta = np.random.rand() * (2*np.pi) - np.pi
            start_config = np.array([start_config_x, start_config_y, start_config_theta])
            perturb_config = start_config + 0.01*np.ones(3)
            obj_val, _ = self.get_cost_of_a_waypoint(start_config, np.reshape(perturb_config, (1, 3)))
            obj_val = obj_val[0]
            return obj_val, start_config

        print("\nComputing costs and gradients for a dataset of robot poses:")
        print("Point no. :")
        dataset_info = []
        for i in range(size_dataset):
            print(i)
            obj_val, start_config = generate_pose()
            # Make sure that start_config is not in an obstacle
            while obj_val > OBSTACLE_COST:  # Regenerate start_config while it is in an obstacle
                obj_val, start_config = generate_pose()

            data = self.get_gradient_one_data_point(start_config, uncertainty_mode)
            dummy_start_config = data['dummy_start_config']
            jacobian = data['jacobian']
            cost_grad = data['cost_grad']
            final_grads = data['final_grads']
            percent_cost_grad = data['percent_cost_grad']
            plan_cost = data['plan_cost']
            best_cost_costmap = data['best_cost_costmap'] 
            percent_obstacle = data['percent_obstacle']
            percent_obstacle_close = data['percent_obstacle_close']

            cost_criticality = tf.norm(cost_grad, ord=np.inf).numpy()
            plan_criticality = tf.abs(tf.reduce_max(jacobian)).numpy()
            criticality = np.linalg.norm(final_grads, ord=np.inf)
            decision_criticality = np.linalg.norm(percent_cost_grad, ord=np.inf)
            badness = (plan_cost - best_cost_costmap)/plan_cost
            analysis_data = {'dummy_start_config': dummy_start_config, 
                            'cost_criticality': cost_criticality, 
                            'plan_criticality': plan_criticality, 
                            'decision_criticality': decision_criticality,
                            'criticality': criticality, 
                            'badness': badness,
                            'cost_of_start_config': obj_val,
                            'plan_cost': plan_cost,
                            'percent_obstacle': percent_obstacle,
                            'percent_obstacle_close': percent_obstacle_close}
            dataset_info.append(analysis_data)
        
        print("Finished the dataset!")

        return dataset_info

    def optimize(self, start_config):
        """ 
        Expects that the neural network will output a set of spline
        trajectory samples (represented by their endpoints), their 
        corresponding costs, and uncertainties. The planner incorporates 
        this cost and uncertainty to choose a plan. 
        """
        # For now, start_config is unused! It is fed into the planner 
        # by other parts of the codebase

        dummy_sc = DUMMY_SC 
        data = self.get_gradient_one_data_point(dummy_start_config=dummy_sc)
        dummy_start_config = data['dummy_start_config']
        full_costmap_n4 = data['full_costmap_n4']
        nn_output_n4 = data['nn_output_n4']
        uncertainties = data['uncertainties']
        plan = data['plan']
        final_grads = data['final_grads']
        perturbed_waypoints = data['perturbed_waypoints']
        perturbed_costs = data['perturbed_costs']
        plan_cost = data['plan_cost']

        # Visualize costmap for dummy_start_config along with the plan and the perturbed
        # points around the plan
        perturbed_costs = list(np.stack([cost[0] for cost in perturbed_costs]))
        additional_waypoints = perturbed_waypoints.copy()
        additional_waypoints.extend(full_costmap_n4[:, :3])
        additional_costs = perturbed_costs.copy()
        additional_costs.extend(np.squeeze(full_costmap_n4[:, 3]))

        display_uncertainties = (len(full_costmap_n4) <= DISPLAY_MULT*self.len_costmap)  # too many points to display
        self.visualize_waypoints(dummy_start_config, nn_output_n4, uncertainties, plan, plan_cost,
                                additional_waypoints, additional_costs, display_uncertainties)
        
        # Analyze gradients for a dataset of robot poses
        dataset_info = self.get_gradients_dataset(fixed_theta=0.0)
        data = self.process_dataset_info(dataset_info, 
                                        visualize_critical_pts_threshold=True, 
                                        plot_relationships=True)

        sys.exit()
        # All of the rest of this function is needed for this planner to work with the 
        # rest of the codebase
        # Get the optimal trajectory
        obj_val, data = self.get_cost_of_a_waypoint(dummy_start_config, np.reshape(plan, (1, 3)))
        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        idx = 0  # 0 bc there's only one waypoint
        self.opt_waypt.assign_from_config_batch_idx(waypts, idx)  # Just here for debugging
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, idx)  

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[idx, 0]/self.params.dt).numpy())

        # To output in the metadata
        dummy_start_sys_config = self.convert_state_arr_to_config(dummy_start_config)
        plan_config = self.convert_state_arr_to_config(np.reshape(plan, (3,)))

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
    
    def visualize_waypoints(self, start_config, costmap, uncertainties, plan, plan_cost,
                            additional_waypoints, additional_costs, 
                            display_uncertainties=True):
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
        total_waypoints = np.vstack([costmap[:, :3], plan, additional_waypoints])
        total_costs = np.hstack([costmap[:, 3], plan_cost, additional_costs])

        viridis = cm.get_cmap('coolwarm', len(total_costs))
        sorted_costs = np.sort(total_costs)
        ind_sorted_costs = np.argsort(total_costs)
        sorted_waypoints = total_waypoints[ind_sorted_costs]

        fig, ax = plt.subplots(figsize=(10, 11))
        plot_quiver = display_uncertainties
        plt.plot(start_config[0], start_config[1], color='g', marker='o', markersize=20)
        for i in range(len(sorted_waypoints)):
            waypoint = sorted_waypoints[i]
            point_config = self.convert_state_arr_to_config(waypoint)
            if sorted_costs[i] > OBSTACLE_COST:
                point_config.render(ax, batch_idx=0, plot_quiver=plot_quiver,
                                 marker='o', color='k')
            else:
                point_config.render(ax, batch_idx=0, plot_quiver=plot_quiver,
                                 marker='o', color=viridis(sorted_costs[i]))
        if display_uncertainties:
            for i in range(len(costmap)):
                waypoint = costmap[i]
                uncertainty = uncertainties[i].numpy()
                # Plot a circle
                angle = np.linspace(0, 2 * np.pi, 150) 
                radius = uncertainty
                x = radius * np.cos(angle) + waypoint[0]
                y = radius * np.sin(angle) + waypoint[1]
                plt.plot(x, y, 'k')
        fig.savefig(IMAGE_PATH + 'waypoints_heatmap.png')

    def process_dataset_info(self, dataset_info, visualize_critical_pts_threshold=False,
                                plot_relationships=False):
        '''
        Plot plan and task critical points. Additionally plot the relationships between 
        various variables.
        '''
        points = np.stack([data['dummy_start_config'] for data in dataset_info])
        badnesses = [data['badness'] for data in dataset_info]
        cost_criticalities = [data['cost_criticality'] for data in dataset_info]
        plan_criticalities = [data['plan_criticality'] for data in dataset_info]
        decision_criticalities = [data['decision_criticality'] for data in dataset_info]
        criticalities = [data['criticality'] for data in dataset_info]
        costs_of_points = [data['cost_of_start_config'] for data in dataset_info]
        plan_costs = [data['plan_cost'] for data in dataset_info]
        percent_obstacles = [data['percent_obstacle'] for data in dataset_info]
        percent_obstacles_close = [data['percent_obstacle_close'] for data in dataset_info]

        badnesses = np.clip(badnesses, -10, 100)
        cost_criticalities = np.clip(cost_criticalities, 0, 200)
        criticalities = np.clip(criticalities, 0, 1000)
        decision_criticalities = np.clip(decision_criticalities, 0, 200)

        # Failed points are points for which the plan is in an obstacle - don't analyze
        # those
        failed_plan_ind = np.argwhere(np.array(plan_costs) > FAILED_PLAN_THRES)
        failed_plan_ind = np.squeeze(failed_plan_ind, axis=1)

        # Non-failed stats
        if len(failed_plan_ind) != 0:
            non_failed_ind = list(set(list(np.arange(len(points)))) - set(list(failed_plan_ind)))
        else:
            non_failed_ind = np.arange(len(points))
        
        non_failed_points = points[non_failed_ind]
        non_failed_decision_criticalities = np.array(decision_criticalities)[non_failed_ind]
        non_failed_cost_criticalities = np.array(cost_criticalities)[non_failed_ind]
        non_failed_badnesses = np.array(badnesses)[non_failed_ind]
        non_failed_percent_obstacles = np.array(percent_obstacles)[non_failed_ind]
        non_failed_percent_obstacles_close = np.array(percent_obstacles_close)[non_failed_ind]

        # Plots - relationships between variables
        def scatter_plotter(var1, var2, title, xlabel, ylabel, figname):
            fontsize = 20
            plt.rc('font', size=fontsize)          # controls default text sizes
            plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
            plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
            plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title

            plt.figure(figsize=(10, 11))
            plt.scatter(var1, var2, marker='o')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(IMAGE_PATH + figname)

        if plot_relationships:
            scatter_plotter(plan_criticalities, criticalities, "Plan Criticality vs Criticality", 
                            "Plan Criticality", "Criticality", 'plan_criticality_vs_criticality.png')
            scatter_plotter(cost_criticalities, criticalities, "Cost Criticality vs Criticality", 
                            "Cost Criticality", "Criticality", 'cost_criticality_vs_criticality.png')
            scatter_plotter(badnesses, criticalities, "Uncertainty Badness vs Criticality", 
                            "Badness", "Criticality", 'badness_vs_criticality.png')
            scatter_plotter(badnesses, cost_criticalities, "Uncertainty Badness vs Cost Criticality", 
                            "Badness", "Cost Criticality", 'badness_vs_cost_criticality.png')
            scatter_plotter(badnesses, plan_criticalities, "Uncertainty Badness vs Plan Criticality", 
                            "Badness", "Plan Criticality", 'badness_vs_plan_criticality.png')
            scatter_plotter(decision_criticalities, plan_criticalities, 
                            "Decision Criticality vs Plan Criticality", "Decision Criticality", 
                            "Plan Criticality", 'decision_vs_plan_criticality.png')
            scatter_plotter(costs_of_points, criticalities, "Costs of Points vs Criticality", 
                            "Cost of Point", "Criticality", 'cost_of_point_vs_criticality.png')

        # Visualize critical points if the criticality is above a threshold
        if visualize_critical_pts_threshold:
            plan_critical_points, cost_critical_points, decision_critical_points, critical_points, \
            badness_points, percent_obstacle_points, percent_obstacle_close_points = \
                self.visualize_critical_points_above_threshold(points, plan_criticalities, 
                                                    non_failed_cost_criticalities, 
                                                    non_failed_decision_criticalities, 
                                                    criticalities,
                                                    non_failed_badnesses,
                                                    non_failed_percent_obstacles,
                                                    non_failed_percent_obstacles_close)
        
        # Visualize all critical points in a heatmap format
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: plan")
        self.visualize_critical_points(points, plan_criticalities, 
                                        figname='plan_critical_points_heatmap', plot_quiver=False)
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: decision")
        self.visualize_critical_points(non_failed_points, non_failed_decision_criticalities, 
                                        figname='decision_critical_points_heatmap', plot_quiver=False)
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: cost")
        self.visualize_critical_points(non_failed_points, non_failed_cost_criticalities, 
                                        figname='cost_critical_points_heatmap', plot_quiver=False)
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: badness")
        self.visualize_critical_points(non_failed_points, np.abs(non_failed_badnesses), 
                                        figname='decision_critical_badness_points_heatmap', plot_quiver=False)
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: percent obstacle")
        self.visualize_critical_points(non_failed_points, np.abs(non_failed_percent_obstacles), 
                                        figname='percent_obstacle_heatmap', plot_quiver=False)
        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\n\nCritical points heatmap: percent obstacle close")
        self.visualize_critical_points(non_failed_points, np.abs(non_failed_percent_obstacles_close), 
                                        figname='percent_obstacle_close_heatmap', plot_quiver=False)
        
        return_dict = {'points': points,
                        'non_failed_points': non_failed_points,
                        'badnesses': badnesses,
                        'cost_criticalities': cost_criticalities,
                        'plan_criticalities': plan_criticalities,
                        'criticalities': criticalities,
                        'decision_criticalities': decision_criticalities,
                        'non_failed_decision_criticalities': non_failed_decision_criticalities,
                        'non_failed_cost_criticalities': non_failed_cost_criticalities,
                        'non_failed_badnesses': non_failed_badnesses,
                        'non_failed_percent_obstacles': non_failed_percent_obstacles,
                        'non_failed_percent_obstacles_close': non_failed_percent_obstacles_close}
        
        if visualize_critical_pts_threshold:
            return_dict['plan_critical_points'] = plan_critical_points
            return_dict['cost_critical_points'] = cost_critical_points
            return_dict['critical_points'] = critical_points 
            return_dict['decision_critical_points'] = decision_critical_points 
        
        return return_dict

    def visualize_critical_points_above_threshold(self, points, plan_criticalities, 
                                                cost_criticalities, decision_criticalities, 
                                                criticalities, badnesses,
                                                percent_obstacles,
                                                percent_obstacles_close,
                                                percentile_threshold=70):
        '''
        Plot points on the map that have high `criticality' values for each of plan_criticalities,
        cost_criticalities, decision_criticalities, and criticalities. It is assumed that these are
        all criticalities for points with non-failed plans. Criticality values are high if they are
        above a percentile threshold.
        '''
        # Helper function to plot robot poses on the obstacle map
        def plot_points_on_map(points, figname, markersize=14, 
                                plotting_grid_steps=150, plot_quiver=True):
            fig, ax = plt.subplots(figsize=(10, 11))
            self.simulator._render_obstacle_map(ax, plotting_grid_steps)
            self.simulator.goal_config.render(ax, batch_idx=0, plot_quiver=False,
                                    marker='*', color='y', markersize=markersize)
            for point in points:
                point_config = self.convert_state_arr_to_config(point)
                point_config.render(ax, batch_idx=0, plot_quiver=plot_quiver,
                                    marker='o', color='b', markersize=markersize)
            fig.savefig(IMAGE_PATH + figname)
        
        # Helper function to get the points that have high criticalities
        def get_critical_points_above_thres(criticalities, print_statement):
            critical_threshold = np.percentile(criticalities, percentile_threshold)
            critical_ind = np.argwhere(np.array(criticalities) >= critical_threshold)
            critical_ind = np.squeeze(critical_ind, axis=1)
            critical_points = []
            if len(critical_ind) != 0:
                critical_points = points[critical_ind]
                if self.dataset_gradients:    
                    self.dataset_gradients_file.write(print_statement + str(critical_points))
            return critical_points

        # Plan Critical Points
        plan_critical_points = get_critical_points_above_thres(plan_criticalities, "\nPlan Critical Points: ")
        plot_points_on_map(plan_critical_points, 'plan_critical_points.png')
        
        # Cost Critical Points 
        cost_critical_points = get_critical_points_above_thres(cost_criticalities, "\nCost Critical Points: ")
        plot_points_on_map(cost_critical_points, 'cost_critical_points.png')
        
        # Decision Critical Points 
        decision_critical_points = get_critical_points_above_thres(decision_criticalities, "\nDecision Critical Points: ")
        plot_points_on_map(decision_critical_points, 'decision_critical_points.png')

        # Overall Critical Points 
        critical_points = get_critical_points_above_thres(criticalities, "\nOverall Critical Points: ")
        plot_points_on_map(critical_points, 'critical_points.png')

        # Badness Values
        badness_points = get_critical_points_above_thres(badnesses, "\nBadness Points: ")
        plot_points_on_map(critical_points, 'badness_points.png')

        # Percent Obstacle Critical Points
        percent_obstacle_points = get_critical_points_above_thres(percent_obstacles, "\nPercent Obstacle Points: ")
        plot_points_on_map(percent_obstacle_points, 'percent_obstacle_points.png')

        # Percent Obstacle Nearby Critical Points
        percent_obstacle_close_points = get_critical_points_above_thres(percent_obstacles_close, 
                                                        "\nPercent Obstacle Nearby Points")
        plot_points_on_map(percent_obstacle_close_points, 'percent_obstacle_close_points.png')

        return plan_critical_points, cost_critical_points, decision_critical_points, critical_points, \
                badness_points, percent_obstacle_points, percent_obstacle_close_points

    def visualize_critical_points(self, points, criticalities, percentile_rng=[5, 95], 
                                    figname=None, plot_quiver=True, 
                                    min_color=np.array([0.6, 0.6, 0.6]), 
                                    max_color=np.array([1.0, 0.0, 0.0])):
        '''
        Plot a heatmap-style plot of various points according to how
        critical they are (gray -> red in order of greater criticality).
        '''
        # Visualize criticalities distribution
        fig, ax = plt.subplots(figsize=(10, 11))
        plt.hist(criticalities)
        if figname is None:
            fig.savefig(IMAGE_PATH + 'criticalities_histogram.png')
        else:
            fig.savefig(IMAGE_PATH + figname + '_histogram.png')
        
        rng_low = int(np.percentile(criticalities, percentile_rng[0]))
        rng_high = np.percentile(criticalities, percentile_rng[1])

        if self.dataset_gradients:    
            self.dataset_gradients_file.write("\nMinimum criticality: " + str(min(criticalities)))
            self.dataset_gradients_file.write("\nMaximum criticality: " + str(max(criticalities)))
            self.dataset_gradients_file.write("\nPercentiles: " + str(rng_low) + ", " + str(rng_high))
        
        # Linear interpolation to get colors wrt criticalities 
        slope = (max_color - min_color)/(rng_high - rng_low)
        def get_color(criticality): 
            if criticality <= rng_low: 
                return min_color
            elif criticality >= rng_high: 
                return max_color
            else:
                return min_color + slope * (criticality - min(criticalities))

        sorted_criticalities = np.sort(criticalities)
        ind_sorted_criticalities = np.argsort(criticalities)
        sorted_points = points[ind_sorted_criticalities]

        plotting_grid_steps = 150
        fig, ax = plt.subplots(figsize=(10, 11))
        self.simulator._render_obstacle_map(ax, plotting_grid_steps)
        self.simulator.goal_config.render(ax, batch_idx=0, plot_quiver=False,
                                    marker='*', color='y', markersize=14)
        for i in range(len(sorted_points)):
            point = sorted_points[i]
            point_config = self.convert_state_arr_to_config(point)
            point_config.render(ax, batch_idx=0, plot_quiver=plot_quiver,
                                 marker='o', color=get_color(sorted_criticalities[i]))
        if figname is None:
            fig.savefig(IMAGE_PATH + 'critical_points_heatmap.png')
        else:
            fig.savefig(IMAGE_PATH + figname + '.png')
        

        # For debugging
        if len(points) < PER_POINT_VIZ:
            # Visualize each point
            for i in range(len(points)):
                point = points[i]
                criticality = criticalities[i]
                print("Point and its criticality:", point, criticality)
                
                fig, ax = plt.subplots(figsize=(10, 11))
                self.simulator._render_obstacle_map(ax, plotting_grid_steps)
                point_config = self.convert_state_arr_to_config(point)
                point_config.render(ax, batch_idx=0, plot_quiver=True,
                                    marker='o', color=get_color(criticality), 
                                    markersize=17)
                fig.savefig('single_point_on_map.png')
                # time.sleep(5)

                data = self.get_gradient_one_data_point(dummy_start_config=point, num_desired_waypoints=10)
                dummy_start_config = data['dummy_start_config']
                full_costmap_n4 = data['full_costmap_n4']
                nn_output_n4 = data['nn_output_n4']
                uncertainties = data['uncertainties']
                plan = data['plan']
                final_grads = data['final_grads']
                perturbed_waypoints = data['perturbed_waypoints']
                perturbed_costs = data['perturbed_costs']
                plan_cost = data['plan_cost']

                perturbed_costs = list(np.stack([cost[0] for cost in perturbed_costs]))
                additional_waypoints = perturbed_waypoints.copy()
                additional_waypoints.extend(full_costmap_n4[:, :3])
                additional_costs = perturbed_costs.copy()
                additional_costs.extend(np.squeeze(full_costmap_n4[:, 3]))

                display_uncertainties = (len(full_costmap_n4) <= DISPLAY_MULT*self.len_costmap)  # too many points to display
                self.visualize_waypoints(dummy_start_config, nn_output_n4, uncertainties, plan, plan_cost,
                                        additional_waypoints, additional_costs, display_uncertainties)
                # time.sleep(5)



    ########## The rest of this file is for making this planner compatible with the rest of the 
    # codebase ##########
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
