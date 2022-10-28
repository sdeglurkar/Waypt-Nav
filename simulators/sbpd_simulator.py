import numpy as np
from obstacles.sbpd_map import SBPDMap
from simulators.simulator import Simulator
from trajectory.trajectory import Trajectory


class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super(SBPDSimulator, self).__init__(params=params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        if hasattr(model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}

        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        **kwargs)
        return img_nmkd
    
    def generate_costmap(self, data_dict):
        '''
        Given the optimal waypoint, get the expert's cost of that waypoint from
        the MPC problem.
        '''
        costmap = []  # Just scalars
        # The map origin of the FMM map is [0, 0] and the optimal_waypoint_ego_n3 is in relative
        # coordinates with the ego position
        waypoints = data_dict['optimal_waypoint_n3']  # TODO (sdeglurkar): Should it really be ego n3 
        goal_positions = data_dict['goal_position_n2']
        for i in range(len(goal_positions)):
            goal_pos = goal_positions[i].reshape(1, 2)
            waypt_pos = waypoints[i, :2]
            waypt_heading = waypoints[i, 2]
            # Get the FMM map with this goal position
            fmm_map = self._init_fmm_map(goal_pos)
            self._update_obj_fn(fmm_map)
            n = 1  # Batch size
            k = 1  # 1-step "trajectories" -- just waypoints
            one_step_trajectories = Trajectory(dt=0, n=n, k=k, 
                                                position_nk2=waypt_pos.reshape((n, k, 2)), 
                                                heading_nk1=waypt_heading.reshape((n, k, 1)))
            one_step_trajectories.update_valid_mask_nk()  # Needed for taking a valid mean of objective values
            cost = self.obj_fn.evaluate_function(one_step_trajectories)  # Tensor
            cost = cost[0].numpy()
            costmap.append(cost)  
        
        return np.array(costmap)

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        if hasattr(self, 'fmm_map'):
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax, start_config=self.start_config,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
