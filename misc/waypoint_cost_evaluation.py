'''
For the waypoint-cost model, does some evaluation based on the trajectory pickle
files that are generated during testing. Evaluates both the learned method and the
expert.
'''

import argparse 
import numpy as np
import os
import pickle


def parse_args():
    '''
    Get job directories from command line input.
    '''

    parser = argparse.ArgumentParser(description='Process the command line inputs')
    parser.add_argument("-j", "--job-dir", required=True,
                                help='the path to the job directory containing testing results')
    parser.add_argument("-t", "--trajectory", required=False,
                                help='the trajectory whose results are desired')
    parser.add_argument("-b", "--base-dir", required=False,
                                help='the base directory for the job directory')
            
    args = parser.parse_args()

    if args.base_dir == None:
        base_dir = "/home/sampada_deglurkar/Waypt-Nav/reproduce_LB_WayptNavResults/"
    else:
        base_dir = args.base_dir 

    testing_dir = base_dir + args.job_dir
    expert_dir = testing_dir + "/expert_simulator/trajectories/"
    method_dir = testing_dir + "/rgb_resnet50_nn_waypoint_simulator/trajectories/"

    # Same amount of trajectory pickle files
    assert(len(os.listdir(expert_dir)) == len(os.listdir(method_dir)))

    return expert_dir, method_dir, args.trajectory

def finite_differencing(linear, angular, dt):
    '''
    Used for getting higher-order derivatives in both linear and angular parts.
    '''
    assert(len(linear) == len(angular))
    linear_deriv = []
    angular_deriv = []
    i = 1
    while i < len(linear):
        linear_diff = linear[i] - linear[i - 1]
        linear_deriv.append(linear_diff/dt)
        angular_diff = angular[i] - angular[i - 1]
        angular_deriv.append(angular_diff/dt)

        i += 1
    
    return linear_deriv, angular_deriv


if __name__ == "__main__":
    expert_dir, method_dir, trajectory_id = parse_args()

    trajectory_files = []
    pred_minus_true_diff_in_costs = []
    true_costs_of_waypoints_list = []
    method_trajectory_times = []
    expert_trajectory_times = []
    method_avg_traj_costs = []
    method_max_traj_costs = []
    method_min_traj_costs = []
    expert_avg_traj_costs = []
    expert_max_traj_costs = []
    expert_min_traj_costs = []
    method_avg_linear_accels = []
    method_avg_angular_accels = []
    expert_avg_linear_accels = []
    expert_avg_angular_accels = []

    for file in os.listdir(method_dir):
        if file == "metadata.pkl": continue

        method_dict = pickle.load(open(method_dir + file, "rb"))
        expert_dict = pickle.load(open(expert_dir + file, "rb"))

        # Waypoints
        method_waypoints = method_dict['vehicle_data']['waypoint_config']['position_nk2']
        expert_waypoints = expert_dict['vehicle_data']['waypoint_config']['position_nk2']
        try:
            true_costs_of_waypoints = method_dict['vehicle_data']['cost_of_waypoint']
            costs_of_expert_waypoints = expert_dict['vehicle_data']['cost_of_waypoint']
        except KeyError: 
            true_costs_of_waypoints = np.array([-1])
            costs_of_expert_waypoints = np.array([-1])
        try:
            method_costs_of_waypoints = method_dict['vehicle_data']['predicted_cost_of_waypoint'].reshape(len(true_costs_of_waypoints))
        except KeyError:
            method_costs_of_waypoints = np.array([-1])
        

        # Time and Speed
        method_trajectory_time = method_dict['vehicle_trajectory']['k'] *  method_dict['vehicle_trajectory']['dt']
        method_linear_speed = method_dict['vehicle_trajectory']['speed_nk1']
        method_angular_speed = method_dict['vehicle_trajectory']['angular_speed_nk1']

        expert_trajectory_time = expert_dict['vehicle_trajectory']['k'] *  expert_dict['vehicle_trajectory']['dt']
        expert_linear_speed = expert_dict['vehicle_trajectory']['speed_nk1']
        expert_angular_speed = expert_dict['vehicle_trajectory']['angular_speed_nk1']

        # Cost of trajectory
        method_avg_traj_cost = method_dict['mean_obj_val']
        method_max_traj_cost = method_dict['max_obj_val']
        method_min_traj_cost = method_dict['min_obj_val']
        expert_avg_traj_cost = expert_dict['mean_obj_val']
        expert_max_traj_cost = expert_dict['max_obj_val']
        expert_min_traj_cost = expert_dict['min_obj_val']

        # Acceleration
        # Cannot just get acceleration from ['vehicle_trajectory'] field, it wasn't created
        method_linear_speed = np.squeeze(method_linear_speed)
        expert_linear_speed = np.squeeze(expert_linear_speed)
        method_angular_speed = np.squeeze(method_angular_speed)
        expert_angular_speed = np.squeeze(expert_angular_speed)

        # Simple finite differencing
        method_linear_accel, method_angular_accel = finite_differencing(method_linear_speed, 
                                                        method_angular_speed, 
                                                        method_dict['vehicle_trajectory']['dt'])
        expert_linear_accel, expert_angular_accel = finite_differencing(expert_linear_speed, 
                                                        expert_angular_speed, 
                                                        expert_dict['vehicle_trajectory']['dt'])

        # Append to lists
        if method_dict['episode_type_string'] == "Success":
            trajectory_files.append(file)
            pred_minus_true_diff_in_costs.append(np.mean(
                                            np.abs(method_costs_of_waypoints - true_costs_of_waypoints)))
            true_costs_of_waypoints_list.append(np.mean(true_costs_of_waypoints))
            method_trajectory_times.append(method_trajectory_time)
            method_avg_traj_costs.append(method_avg_traj_cost)
            method_max_traj_costs.append(method_max_traj_cost)
            method_min_traj_costs.append(method_min_traj_cost)
            method_avg_linear_accels.append(np.mean(method_linear_accel))
            method_avg_angular_accels.append(np.mean(method_angular_accel))
        if expert_dict['episode_type_string'] == "Success":
            expert_trajectory_times.append(expert_trajectory_time)
            expert_avg_traj_costs.append(expert_avg_traj_cost)
            expert_max_traj_costs.append(expert_max_traj_cost)
            expert_min_traj_costs.append(expert_min_traj_cost)
            expert_avg_linear_accels.append(np.mean(expert_linear_accel))
            expert_avg_angular_accels.append(np.mean(expert_angular_accel))


        # Prints for a particular trajectory
        if trajectory_id != None and file == "traj_" + trajectory_id + ".pkl":
            print("\n")
            print(file)
            print("Method Success:", method_dict['episode_type_string'])
            print("Expert Success:", expert_dict['episode_type_string'])

            print("Cost of Trajectory, Method: Mean, Max, Min", method_avg_traj_cost, \
                                                                method_max_traj_cost, \
                                                                method_min_traj_cost)
            print("Cost of Trajectory, Expert: Mean, Max, Min", expert_avg_traj_cost, \
                                                                expert_max_traj_cost, \
                                                                expert_min_traj_cost)

            distances_btwn_waypoints = []
            num_waypoints = min(len(method_waypoints), len(expert_waypoints))
            for i in range(num_waypoints):
                method_waypoint = method_waypoints[i]
                expert_waypoint = expert_waypoints[i]
                distances_btwn_waypoints.append(np.linalg.norm(method_waypoint - expert_waypoint))

            print("\nAmount of method waypoints", len(method_waypoints))
            print("Amount of expert waypoints", len(expert_waypoints))
            print("Distances Between Expert and Predicted Waypoints", distances_btwn_waypoints)
            
            print("\nPredicted Costs of Waypoints", method_costs_of_waypoints)
            print("Average Predicted Cost of Waypoints", np.mean(method_costs_of_waypoints))
            print("Maximum Predicted Cost of Waypoints", np.max(method_costs_of_waypoints))
            print("Minimum Predicted Cost of Waypoints", np.min(method_costs_of_waypoints))
            
            print("\nTrue Costs of Waypoints", true_costs_of_waypoints)
            print("Average True Cost of Waypoints", np.mean(true_costs_of_waypoints))
            print("Maximum True Cost of Waypoints", np.max(true_costs_of_waypoints))
            print("Minimum True Cost of Waypoints", np.min(true_costs_of_waypoints))
            
            print("\nCosts of Expert Waypoints", costs_of_expert_waypoints)
            print("Average cost of expert waypoints", np.mean(costs_of_expert_waypoints))
            print("Maximum cost of expert waypoints", np.max(costs_of_expert_waypoints))
            print("Minimum cost of expert waypoints", np.min(costs_of_expert_waypoints))
        
            avg_diff_cost = np.mean(np.abs(method_costs_of_waypoints - true_costs_of_waypoints))
            print("\nAverage difference in cost pred - true", avg_diff_cost)

            print("\nTime taken by method", method_trajectory_time, "sec")
            print("Time taken by expert", expert_trajectory_time, "sec")

            ################# Speed ####################

            print("\nAverage linear speed, method", np.mean(method_linear_speed))
            print("Maximum linear speed, method", np.max(method_linear_speed))
            print("Range of linear speed, method", np.max(method_linear_speed) - np.min(method_linear_speed))

            print("\nAverage linear speed, expert", np.mean(expert_linear_speed))
            print("Maximum linear speed, expert", np.max(expert_linear_speed))
            print("Range of linear speed, expert", np.max(expert_linear_speed) - np.min(expert_linear_speed))

            print("\nAverage angular speed, method", np.mean(method_angular_speed))
            print("Maximum angular speed, method", np.max(method_angular_speed))
            print("Range of angular speed, method", np.max(method_angular_speed) - np.min(method_angular_speed))

            print("\nAverage angular speed, expert", np.mean(expert_angular_speed))
            print("Maximum angular speed, expert", np.max(expert_angular_speed))
            print("Range of angular speed, expert", np.max(expert_angular_speed) - np.min(expert_angular_speed))

            ################## Acceleration ###############

            print("\nAverage linear acceleration, method", np.mean(method_linear_accel))
            print("Maximum linear acceleration, method", np.max(method_linear_accel))
            print("Range of linear acceleration, method", np.max(method_linear_accel) - np.min(method_linear_accel))

            print("\nAverage linear acceleration, expert", np.mean(expert_linear_accel))
            print("Maximum linear acceleration, expert", np.max(expert_linear_accel))
            print("Range of linear acceleration, expert", np.max(expert_linear_accel) - np.min(expert_linear_accel))

            print("\nAverage angular acceleration, method", np.mean(method_angular_accel))
            print("Maximum angular acceleration, method", np.max(method_angular_accel))
            print("Range of angular acceleration, method", np.max(method_angular_accel) - np.min(method_angular_accel))

            print("\nAverage angular acceleration, expert", np.mean(expert_angular_accel))
            print("Maximum angular acceleration, expert", np.max(expert_angular_accel))
            print("Range of angular acceleration, expert", np.max(expert_angular_accel) - np.min(expert_angular_accel))



    print("\n-------------------------------------------------------")
    print("Statistics for all successful trajectories:")
    print("Average abs difference between predicted and true waypoint costs", np.mean(pred_minus_true_diff_in_costs))
    print("Average true cost of waypoints", np.mean(true_costs_of_waypoints_list))
    print("Average time taken by method", np.mean(method_trajectory_times))
    print("Average time taken by expert", np.mean(expert_trajectory_times))
    print("Average mean trajectory cost for method", np.mean(method_avg_traj_costs))
    print("Large average mean method trajectory costs", [elem for elem in method_avg_traj_costs if elem > 3.0])
    indices = np.where(np.array(method_avg_traj_costs) > 3.0)
    trajectory_files = np.array(trajectory_files)
    print("Trajectories for the above", trajectory_files[indices])
    print("Average mean trajectory cost for expert", np.mean(expert_avg_traj_costs))
    print("Average max trajectory cost for method", np.mean(method_max_traj_costs))
    print("Average max trajectory cost for expert", np.mean(expert_max_traj_costs))
    print("Average min trajectory cost for method", np.mean(method_min_traj_costs))
    print("Average min trajectory cost for expert", np.mean(expert_min_traj_costs))
    print("Average mean linear acceleration for method", np.mean(method_avg_linear_accels))
    print("Average mean linear acceleration for expert", np.mean(expert_avg_linear_accels))
    print("Average mean angular acceleration for method", np.mean(method_avg_angular_accels))
    print("Average mean angular acceleration for expert", np.mean(expert_avg_angular_accels))