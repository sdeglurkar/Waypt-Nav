import os
import pickle
import numpy as np
import tensorflow as tf

from data_sources.visual_navigation_data_source import VisualNavigationDataSource
from systems.dubins_car import DubinsCar
from utils import utils


class CostDataSource(VisualNavigationDataSource):
    '''
    Wrapper on VisualNavigationDataSource allowing for the storage
    of all waypoints considered by the expert supervision and their
    corresponding costs.
    '''

    @staticmethod
    def reset_data_dictionary(params):
        """
        Create a dictionary to store the data.
        """
        # Data dictionary to store the data
        data = {}

        # Start configuration information
        data['vehicle_state_nk3'] = []
        data['vehicle_controls_nk2'] = []

        # Goal configuration information
        data['goal_position_n2'] = []
        data['goal_position_ego_n2'] = []

        # Optimal waypoint configuration information
        data['optimal_waypoint_n3'] = []
        data['optimal_waypoint_ego_n3'] = []

        # All waypoints and their costs
        data['all_waypoints_n3'] = []
        data['all_waypoints_ego_n3'] = []
        data['all_waypoints_costs'] = []

        # The horizon of waypoint
        data['waypoint_horizon_n1'] = []

        # Optimal control information
        data['optimal_control_nk2'] = []

        # Episode type information
        data['episode_type_string_n1'] = []
        data['episode_number_n1'] = []
        
        # Last step information
        # Saved separately from other episode information
        # So that we can decide whether to train on this or not
        data['last_step_vehicle_state_nk3'] = []
        data['last_step_vehicle_controls_nk2'] = []
        data['last_step_goal_position_n2'] = []
        data['last_step_goal_position_ego_n2'] = []
        data['last_step_optimal_waypoint_n3'] = []
        data['last_step_optimal_waypoint_ego_n3'] = []
        data['last_all_waypoints_n3'] = []
        data['last_all_waypoints_ego_n3'] = []
        data['last_all_waypoints_costs'] = []
        data['last_step_optimal_control_nk2'] = []
        data['last_step_data_valid_n'] = []

        return data
    
    def _get_n(self, data):
        """
        Returns n, the batch size of the data inside
        this data dictionary.
        """
        return data['goal_position_n2'].shape[0]

    def append_data_to_dictionary(self, data, simulator):
        """
        Append the appropriate data from the simulator to the existing data dictionary.
        """
        # Batch Dimension
        n = simulator.vehicle_data['system_config'].n

        # Vehicle data
        data['vehicle_state_nk3'].append(simulator.vehicle_data['trajectory'].position_and_heading_nk3().numpy())

        # Convert to egocentric coordinates
        start_nk3 = simulator.vehicle_data['system_config'].position_and_heading_nk3().numpy()

        goal_n13 = np.broadcast_to(simulator.goal_config.position_and_heading_nk3().numpy(), (n, 1, 3))
        waypoint_n13 = simulator.vehicle_data['waypoint_config'].position_and_heading_nk3().numpy()
        all_waypoints_n13 = simulator.vehicle_data['all_waypoint_configs'].position_and_heading_nk3().numpy()

        # A single start position, goal position, and optimal waypoint is associated with a 
        # set of sampled waypoints
        all_waypoints_number = simulator.vehicle_data['all_waypoints_number']
        start_nk3 = np.repeat(start_nk3, all_waypoints_number, axis=0)
        goal_n13 = np.repeat(goal_n13, all_waypoints_number, axis=0)
        waypoint_n13 = np.repeat(waypoint_n13, all_waypoints_number, axis=0)
        # Same for the vehicle control
        vehicle_controls_repeat = np.repeat(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy(), 
                            all_waypoints_number, axis=0)
        data['vehicle_controls_nk2'].append(vehicle_controls_repeat)


        goal_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                 goal_n13)
        waypoint_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     waypoint_n13)
        all_waypoints_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                    all_waypoints_n13)

        all_waypoints_costs = simulator.vehicle_data['all_waypoint_costs'] # This is a list of np.arrays
        all_waypoints_costs = np.expand_dims(np.concatenate(all_waypoints_costs), axis=1)

        # Goal Data
        data['goal_position_n2'].append(goal_n13[:, 0, :2])
        data['goal_position_ego_n2'].append(goal_ego_n13[:, 0, :2])

        # Waypoint data
        data['optimal_waypoint_n3'].append(waypoint_n13[:, 0])
        data['optimal_waypoint_ego_n3'].append(waypoint_ego_n13[:, 0])
        data['all_waypoints_n3'].append(all_waypoints_n13[:, 0])
        data['all_waypoints_ego_n3'].append(all_waypoints_ego_n13[:, 0])

        # Costs
        data['all_waypoints_costs'].append(all_waypoints_costs)

        # Waypoint horizon
        data['waypoint_horizon_n1'].append(simulator.vehicle_data['planning_horizon_n1'])

        # Optimal control data
        data['optimal_control_nk2'].append(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy())

        # Episode Type Information
        data['episode_type_string_n1'].append([simulator.params.episode_termination_reasons[simulator.episode_type]]*n)
        data['episode_number_n1'].append([self.episode_counter]*n)

        data = self._append_last_step_info_to_dictionary(data, simulator)
        return data

    # TODO Varun T.: Clean up this code so the structure isnt repeating
    # the function below
    def _append_last_step_info_to_dictionary(self, data, simulator):
        """
        Append data from the last trajectory segment
        to the data dictionary.
        """
        data_last_step = simulator.vehicle_data_last_step
        n = data_last_step['system_config'].n

        data['last_step_vehicle_state_nk3'].append(simulator.vehicle_data_last_step['trajectory'].position_and_heading_nk3().numpy())

        last_step_goal_n13 = np.broadcast_to(simulator.goal_config.position_and_heading_nk3().numpy(), (n, 1, 3)) 
        last_step_waypoint_n13 = data_last_step['waypoint_config'].position_and_heading_nk3().numpy()
        last_all_waypoints_n13 = data_last_step['all_waypoint_configs'].position_and_heading_nk3().numpy()

        # Convert to egocentric coordinates
        start_nk3 = data_last_step['system_config'].position_and_heading_nk3().numpy()

        # A single start position, goal position, and optimal waypoint is associated with a 
        # set of sampled waypoints
        all_waypoints_number = data_last_step['all_waypoints_number']
        start_nk3 = np.repeat(start_nk3, all_waypoints_number, axis=0)
        last_step_goal_n13 = np.repeat(last_step_goal_n13, all_waypoints_number, axis=0)
        last_step_waypoint_n13 = np.repeat(last_step_waypoint_n13, all_waypoints_number, axis=0)
        # Same for the vehicle control
        vehicle_controls_repeat = np.repeat(simulator.vehicle_data_last_step['trajectory'].speed_and_angular_speed_nk2().numpy(), 
                            all_waypoints_number, axis=0)
        data['last_step_vehicle_controls_nk2'].append(vehicle_controls_repeat)

        goal_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                 last_step_goal_n13)
        waypoint_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     last_step_waypoint_n13)
        last_all_waypoints_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     last_all_waypoints_n13)

        last_all_waypoints_costs = data_last_step['all_waypoint_costs'] 

        data['last_step_goal_position_n2'].append(last_step_goal_n13[:, 0, :2])
        
        data['last_step_goal_position_ego_n2'].append(goal_ego_n13[:, 0, :2])
        
        data['last_step_optimal_waypoint_n3'].append(last_step_waypoint_n13[:, 0, :])
        data['last_step_optimal_waypoint_ego_n3'].append(waypoint_ego_n13[:, 0, :])
        data['last_all_waypoints_n3'].append(last_all_waypoints_n13[:, 0])
        data['last_all_waypoints_ego_n3'].append(last_all_waypoints_ego_n13[:, 0])

        data['last_all_waypoints_costs'].append(last_all_waypoints_costs)

        data['last_step_optimal_control_nk2'].append(simulator.vehicle_data_last_step['trajectory'].speed_and_angular_speed_nk2().numpy())
        data['last_step_data_valid_n'].append([simulator.last_step_data_valid])
        return data


