'''
For the waypoint-cost model, does some evaluation based on the trajectory pickle
files that are generated during testing. Evaluates both the learned method and the
expert.
'''

import argparse 
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from utils import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the command line inputs')
    parser.add_argument("-j", "--job-dir", required=True,
                                help='the path to the job directory containing testing results')
    parser.add_argument("-t", "--trajectory", required=True,
                                help='the trajectory whose results are desired')
            
    args = parser.parse_args()

    base_dir = "/home/sampada_deglurkar/Waypt-Nav/reproduce_LB_WayptNavResults/"
    testing_dir = base_dir + args.job_dir
    expert_dir = testing_dir + "/expert_simulator/trajectories/"
    method_dir = testing_dir + "/rgb_resnet50_nn_waypoint_simulator/trajectories/"

    # Same amount of trajectory pickle files
    assert(len(os.listdir(expert_dir)) == len(os.listdir(method_dir)))

    for file in os.listdir(method_dir):
        if file == "metadata.pkl": continue
        method_dict = pickle.load(open(method_dir + file, "rb"))
        expert_dict = pickle.load(open(expert_dir + file, "rb"))
        method_waypoints = method_dict['vehicle_data']['waypoint_config']['position_nk2']
        expert_waypoints = expert_dict['vehicle_data']['waypoint_config']['position_nk2']
        true_costs_of_waypoints = method_dict['vehicle_data']['cost_of_waypoint']
        method_costs_of_waypoints = method_dict['vehicle_data']['predicted_cost_of_waypoint'].reshape(len(true_costs_of_waypoints))
        costs_of_expert_waypoints = expert_dict['vehicle_data']['cost_of_waypoint']

        distances_btwn_waypoints = []
        num_waypoints = min(len(method_waypoints), len(expert_waypoints))
        for i in range(num_waypoints):
            method_waypoint = method_waypoints[i]
            expert_waypoint = expert_waypoints[i]
            distances_btwn_waypoints.append(np.linalg.norm(method_waypoint - expert_waypoint))
        
        if file == "traj_" + args.trajectory + ".pkl":
            print("\n")
            print(file)
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

