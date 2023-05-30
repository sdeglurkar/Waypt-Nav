import numpy as np
import tensorflow as tf
from planners.planner import Planner
from trajectory.trajectory import Trajectory, SystemConfig


class ExtendedSamplingCostsPlanner(Planner):
    """ A planner which selects an optimal waypoint using
    a sampling based method. Given a fixed start_config,
    the planner
        1. Uses a control pipeline to plan paths from start_config
            to a fixed set of waypoint configurations
        2. Evaluates the objective function on the resulting trajectories
        3. Returns the minimum cost waypoint and associated trajectory as 
            well as all the waypoints and their costs.

    Basically a wrapper on the SamplingCostsPlanner with information on all 
    waypoints and their costs.  
    Used during data generation for the purpose of training the cost network.
    """

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.control_pipeline_params.pipeline.parse_params(p.control_pipeline_params)

        p.system_dynamics = p.control_pipeline_params.system_dynamics_params.system
        p.dt = p.control_pipeline_params.system_dynamics_params.dt
        p.planning_horizon = p.control_pipeline_params.planning_horizon
        return p
    
    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'system_config': [],
                'waypoint_config': [],  # Optimal waypoint
                'cost_of_waypoint': [],  # Cost of optimal waypoint
                'all_waypoints_number': [],
                'all_waypoint_configs': [],
                'all_waypoint_costs': [],
                'trajectory': [],
                'spline_trajectory': [],
                'planning_horizon': [],
                'K_nkfd': [],
                'k_nkf1': [],
                'img_nmkd': []}
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
        data_last['cost_of_waypoint'] = data['cost_of_waypoint'][last_data_idx]
        data_last['all_waypoints_number'] = data['all_waypoints_number'][last_data_idx]
        data_last['all_waypoint_configs'] = data['all_waypoint_configs'][last_data_idx]
        data_last['all_waypoint_costs'] = data['all_waypoint_costs'][last_data_idx]
        data_last['trajectory'] = data['trajectory'][last_data_idx]
        data_last['spline_trajectory'] = data['spline_trajectory'][last_data_idx]
        data_last['planning_horizon_n1'] = [data['planning_horizon'][last_data_idx]] 
        data_last['K_nkfd'] = data['K_nkfd'][last_data_idx]
        data_last['k_nkf1'] = data['k_nkf1'][last_data_idx]
        data_last['img_nmkd'] = data['img_nmkd'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['cost_of_waypoint'] = np.array(data['cost_of_waypoint'])[valid_mask]
        data['all_waypoints_number'] = np.array(data['all_waypoints_number'])[valid_mask]
        data['all_waypoint_configs'] = SystemConfig.concat_across_batch_dim(np.array(data['all_waypoint_configs'])[valid_mask])
        data['all_waypoint_costs'] = np.array(data['all_waypoint_costs'])[valid_mask]
        data['trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['trajectory'])[valid_mask])
        data['spline_trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['spline_trajectory'])[valid_mask])
        data['planning_horizon_n1'] = np.array(data['planning_horizon'])[valid_mask][:, None]
        data['K_nkfd'] = tf.boolean_mask(tf.concat(data['K_nkfd'], axis=0), valid_mask)
        data['k_nkf1'] = tf.boolean_mask(tf.concat(data['k_nkf1'], axis=0), valid_mask)
        data['img_nmkd'] = np.array(np.concatenate(data['img_nmkd'], axis=0))[valid_mask]
        
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
        data_numpy['cost_of_waypoint'] = data['cost_of_waypoint']
        data_numpy['all_waypoints_number'] = data['all_waypoints_number']
        data_numpy['all_waypoint_configs'] = data['all_waypoint_configs'].to_numpy_repr()
        data_numpy['all_waypoint_costs'] = data['all_waypoint_costs']
        data_numpy['trajectory'] = data['trajectory'].to_numpy_repr()
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()
        data_numpy['img_nmkd'] = data['img_nmkd']
        return data_numpy

    def optimize(self, start_config, num_desired_waypoints=None):
        """ Optimize the objective over a trajectory
        starting from start_config.
            1. Uses a control pipeline to plan paths from start_config
            2. Evaluates the objective function on the resulting trajectories
            3. Return the minimum cost waypoint, trajectory, and cost
        """
        obj_vals, data = self.eval_objective(start_config)
        min_idx = tf.argmin(obj_vals)
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0] / self.params.dt).numpy())

        # Select only num_candidate_waypoints amount of waypoints
        num_waypoints = waypts.n
        if num_desired_waypoints is None: 
            num_desired_waypoints = self.params.data_creation.num_candidate_waypoints
        waypoint_configs = Trajectory(dt=self.params.dt, 
                                    n=min(num_waypoints, num_desired_waypoints) + 1, # +1 for opt waypt 
                                    k=1, 
                                    variable=True)
        if num_waypoints > num_desired_waypoints:
            # Randomly select waypoints
            batch_indices = np.random.choice(num_waypoints, num_desired_waypoints, replace=False)
            while min_idx in batch_indices: # Make sure optimal waypoint is not included
                batch_indices = np.random.choice(num_waypoints, num_desired_waypoints, replace=False)
            list_batch_indices = list(batch_indices)
            list_batch_indices.append(min_idx) # Add the optimal waypoint
            batch_indices = np.array(list_batch_indices)
            waypoint_configs.assign_from_trajectory_batch_indices(waypts, batch_indices)
            obj_vals = tf.gather(obj_vals, batch_indices)

        # If the real LQR data has been discarded just take the first element
        # since it will be all zeros
        if self.params.control_pipeline_params.discard_LQR_controller_data:
            K_nkfd = controllers['K_nkfd'][0: 1]
            k_nkf1 = controllers['k_nkf1'][0: 1]
        else:
            K_nkfd = controllers['K_nkfd'][min_idx:min_idx + 1]
            k_nkf1 = controllers['k_nkf1'][min_idx:min_idx + 1]

        img_nmkd = self.simulator.get_observation(config=start_config)

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'cost_of_waypoint': min_cost.numpy(),
                'all_waypoints_number': min(num_waypoints, num_desired_waypoints) + 1,
                'all_waypoint_configs': waypoint_configs,
                'all_waypoint_costs': obj_vals.numpy(),
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': K_nkfd,
                'k_nkf1': k_nkf1,
                'img_nmkd': img_nmkd}

        return data