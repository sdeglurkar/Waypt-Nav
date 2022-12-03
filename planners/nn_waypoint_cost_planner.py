import numpy as np
import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointCostPlanner(NNPlanner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. Is basically a wrapper for 
    NNWaypointPlanner to account for the cost field in the
    data dictionary. """

    def __init__(self, simulator, params):
        super(NNWaypointCostPlanner, self).__init__(simulator, params)
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'system_config': [],
                'waypoint_config': [],
                'cost_of_waypoint': [],
                'predicted_cost_of_waypoint': [],
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
        data_last['predicted_cost_of_waypoint'] = data['predicted_cost_of_waypoint'][last_data_idx]
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
        data['predicted_cost_of_waypoint'] = np.array(data['predicted_cost_of_waypoint'])[valid_mask]
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
        data_numpy['predicted_cost_of_waypoint'] = data['predicted_cost_of_waypoint']
        data_numpy['trajectory'] = data['trajectory'].to_numpy_repr()
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()
        data_numpy['img_nmkd'] = data['img_nmkd']
        return data_numpy

    def _raw_data(self, start_config):
        """
        Return a dictionary of raw_data from the simulator.
        To be passed to model.create_nn_inputs_and_outputs
        """
        simulator = self.simulator
        data = {}

        # Convert Goal to Egocentric Coordinates
        self.params.system_dynamics.to_egocentric_coordinates(start_config,
                                                              simulator.goal_config,
                                                              self.goal_ego_config)

        # Image Data
        if hasattr(self.params.model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      self.params.model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}
        data['img_nmkd'] = simulator.get_observation(config=start_config, **kwargs)

        # Vehicle Data
        data['vehicle_state_nk3'] = start_config.position_and_heading_nk3().numpy()
        data['vehicle_controls_nk2'] = start_config.speed_and_angular_speed_nk2().numpy()

        # Goal Data
        data['goal_position_n2'] = simulator.goal_config.position_nk2().numpy()[:, 0, :]
        data['goal_position_ego_n2'] = self.goal_ego_config.position_nk2().numpy()[:, 0, :]

        # Dummy Labels
        data['optimal_waypoint_ego_n3'] = np.ones((1, 3), dtype=np.float32)
        data['waypoint_horizon_n1'] = np.ones((1, 1), dtype=np.float32)
        data['optimal_control_nk2'] = np.ones((1, 1, 2), dtype=np.float32)
        data['cost'] = np.ones((1,), dtype=np.float32)
        return data

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
        """
        p = self.params

        model = p.model

        raw_data = self._raw_data(start_config)
        processed_data = model.create_nn_inputs_and_outputs(raw_data)
        
        # Predict the NN output
        nn_output_114 = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                    is_training=False)[:, None]
        # If model predicts cost
        if nn_output_114.shape[2] == 4:
            predicted_waypoint_cost = nn_output_114[:, :, 3]
        else:
            predicted_waypoint_cost = nn_output_114[:, :, 0] * 0  # Dummy

        # Transform to World Coordinates
        waypoint_ego_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=nn_output_114[:, :, :2],
                                           heading_nk1=nn_output_114[:, :, 2:3])
        self.params.system_dynamics.to_world_coordinates(start_config,
                                                         waypoint_ego_config,
                                                         self.waypoint_world_config)

        # Evaluate the objective and retrieve Control Pipeline data
        obj_vals, data = self.eval_objective(start_config, self.waypoint_world_config)
        
        # The batch dimension is length 1 since there is only one waypoint
        min_idx = 0
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0]/self.params.dt).numpy())

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'cost_of_waypoint': min_cost.numpy(),
                'predicted_cost_of_waypoint': predicted_waypoint_cost.numpy(),
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][min_idx:min_idx + 1],
                'k_nkf1': controllers['k_nkf1'][min_idx:min_idx + 1],
                'img_nmkd': raw_data['img_nmkd']}

        return data
