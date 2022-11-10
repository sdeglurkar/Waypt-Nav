import numpy as np
import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointCostPlanner(NNPlanner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. Is basically a wrapper for 
    NNWaypointPlanner. """

    def __init__(self, simulator, params):
        super(NNWaypointCostPlanner, self).__init__(simulator, params)
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)

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
        data['costmap'] = np.ones((1,), dtype=np.float32)
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
        nn_output_113 = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                    is_training=False)[:, None]

        # Transform to World Coordinates
        waypoint_ego_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=nn_output_113[:, :, :2],
                                           heading_nk1=nn_output_113[:, :, 2:3])
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
                'waypoint_cost': min_cost,
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][min_idx:min_idx + 1],
                'k_nkf1': controllers['k_nkf1'][min_idx:min_idx + 1],
                'img_nmkd': raw_data['img_nmkd']}

        return data
