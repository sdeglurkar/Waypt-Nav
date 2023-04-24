import numpy as np
import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class DifferentiablePlanner(NNPlanner):
    """ A sampling-based planner that is differentiable with respect 
    to its inputs and internal parameters which selects an optimal spline
    plan using a trained (costmap) neural network."""

    def __init__(self, simulator, params):
        print("\nINITIALIZING PLANNER")
        super(DifferentiablePlanner, self).__init__(simulator, params)
        self.theta = self.params.diff_planner_uncertainty_weight
        self.len_costmap = self.params.len_costmap
        self.uncertainty_amount = 1.0
        self.noise = 0.25
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)

        tfe = tf.contrib.eager
        self.costs = tfe.Variable(np.zeros((self.len_costmap)))
        self.uncertainties = tfe.Variable(np.zeros((self.len_costmap)))
        
    def get_uncertainties(self):
        '''
        For now a dummy function that will just output random numbers, 
        should later be a function defined by the model.
        '''
        return tf.random_uniform([self.len_costmap], dtype=tf.double) * self.uncertainty_amount

    def planner_internal_cost(self, costmap, uncertainties):
        '''
        Returns the planner internal cost of all candidate sampled
        spline trajectories, in which all candidate samples' NN-provided 
        costs and uncertainties are given by 'costmap' and 
        'uncertainties'.
        '''
        self.costs.assign(costmap)
        self.uncertainties.assign(uncertainties)
        denominator = tf.reduce_sum(self.costs) + self.theta * tf.reduce_sum(self.uncertainties)
        numerator = self.costs + self.theta * self.uncertainties
        return tf.divide(numerator, denominator)

    def planner_loss(self, gt_traj, costmap, uncertainties):
        '''
        We're not actually training the planner but this is important
        for the definition of the planner gradient.
        '''
        planner_internal_costs = self.planner_internal_cost(costmap, uncertainties)
        with tf.GradientTape() as tape:
            loss = tf.losses.softmax_cross_entropy(gt_traj, -planner_internal_costs)
        grads = tape.gradient(loss, [self.costs, self.uncertainties])
        return loss, grads

    def get_data_from_pickle(self, file_num='1', batch_index=0):
        """
        From the costmap pickle files, return the start config and
        associated ground truth costmap. This is a dummy function.
        Eventually should be replaced by a call to the NN costmap model
        and/or an ExtendedSamplingCostsPlanner that will give the ground 
        truth costmap.
        """
        import pickle
        path = '/home/ext_drive/sampada_deglurkar/costmap_data/file' + file_num + '.pkl'
        d = pickle.load(open(path, 'rb'))
        start_configs = d['vehicle_state_nk3'][batch_index, 0, :]  # (3,)
        costmaps = d['costmap'][batch_index, :, :]  # (len_costmap, 4)
        # Simulated network output -- ground truth costs + noise
        costmaps[:, 3] = costmaps[:, 3] + np.random.rand(self.len_costmap)*self.noise

        return start_configs, costmaps

    def optimize(self, start_config):
        """ 
        Expects that the neural network will output a set of spline
        trajectory samples (represented by their endpoints), their corresponding 
        costs, and uncertainties. Incorporates this cost into the planner 
        internal cost function and chooses the trajectory with 
        the lowest planner internal cost.
        """
        print("\nINSIDE OPTIMIZE")
        p = self.params

        model = p.model

        # For now, start_config is unused!

        # Get the costmap
        # raw_data = self._raw_data(start_config)
        # processed_data = model.create_nn_inputs_and_outputs(raw_data)
        # nn_output_114 = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
        #                                                             is_training=False)[:, None]

        dummy_start_config, nn_output_n4 = self.get_data_from_pickle()
        print("\nGOT DATA FROM PICKLE", dummy_start_config, nn_output_n4)
        
        uncertainties = self.get_uncertainties()
        planner_internal_costs = self.planner_internal_cost(nn_output_n4[:, 3], uncertainties)

        print("\nCOSTMAP", nn_output_n4[:, 3])
        print("\nUNCERTAINTIES", uncertainties)
        print("\nPLANNER INTERNAL COSTS", planner_internal_costs)

        # Minimize the planner internal cost
        min_idx = tf.argmin(planner_internal_costs)
        min_cost = planner_internal_costs[min_idx]  # Just here for debugging

        gt_min_idx = np.argmin(nn_output_n4[:, 3])
        gt_traj = np.zeros(self.len_costmap)
        gt_traj[gt_min_idx] = 1.0  # One-hot vector
        
        # Get the optimal trajectory
        # First transform waypoints to World Coordinates
        pos_nk2 = np.reshape(nn_output_n4[:, :2][min_idx], (1, 1, 2))  
        head_nk1 = np.reshape(nn_output_n4[:, 2:3][min_idx], (1, 1, 1))  
        waypoint_ego_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)
        pos_nk2 = np.reshape(dummy_start_config[:2], (1, 1, 2))  
        head_nk1 = np.reshape(dummy_start_config[2], (1, 1, 1))
        dummy_start_sys_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=pos_nk2,
                                           heading_nk1=head_nk1)
        self.params.system_dynamics.to_world_coordinates(dummy_start_sys_config,
                                                         waypoint_ego_config,
                                                         self.waypoint_world_config)

        # Now retrieve Control Pipeline data
        _, data = self.eval_objective(dummy_start_sys_config, self.waypoint_world_config)
        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        idx = 0  # 0 bc there's only one waypoint
        self.opt_waypt.assign_from_config_batch_idx(waypts, idx)  # Just here for debugging
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, idx)  

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[idx, 0]/self.params.dt).numpy())

        loss, grads = self.planner_loss(gt_traj, nn_output_n4[:, 3], uncertainties)


        data = {'system_config': dummy_start_sys_config,
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'min_planner_internal_cost': min_cost.numpy(),
                'planner_loss': loss.numpy(),
                'planner_gradients': np.array(grads),
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][idx:idx + 1],
                'k_nkf1': controllers['k_nkf1'][idx:idx + 1]}

        return data

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'system_config': [],
                'waypoint_config': [],
                'min_planner_internal_cost': [],
                'planner_loss': [],
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
        data_last['min_planner_internal_cost'] = data['min_planner_internal_cost'][last_data_idx]
        data_last['planner_loss'] = data['planner_loss'][last_data_idx]
        data_last['planner_gradients'] = data['planner_gradients'][last_data_idx]
        data_last['trajectory'] = data['trajectory'][last_data_idx]
        data_last['spline_trajectory'] = data['spline_trajectory'][last_data_idx]
        data_last['planning_horizon_n1'] = [data['planning_horizon'][last_data_idx]] 
        data_last['K_nkfd'] = data['K_nkfd'][last_data_idx]
        data_last['k_nkf1'] = data['k_nkf1'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['min_planner_internal_cost'] = np.array(data['min_planner_internal_cost'])[valid_mask]
        data['planner_loss'] = np.array(data['planner_loss'])[valid_mask]
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
        data_numpy['min_planner_internal_cost'] = data['min_planner_internal_cost']
        data_numpy['planner_loss'] = data['planner_loss']
        data_numpy['planner_gradients'] = data['planner_gradients']
        data_numpy['trajectory'] = data['trajectory'].to_numpy_repr()
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()

        return data_numpy
