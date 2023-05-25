from math import exp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig

EPSILON = 1e-10
PLOTTING_CLIP_VALUE = 50

class DifferentiablePlanner(NNPlanner):
    """ A sampling-based planner that is differentiable with respect 
    to its inputs and internal parameters which selects an optimal spline
    plan using a trained (costmap) neural network."""

    def __init__(self, simulator, params):
        print("\nINITIALIZING PLANNER")
        super(DifferentiablePlanner, self).__init__(simulator, params)
        self.theta = self.params.diff_planner_uncertainty_weight
        self.softmax_beta = self.params.diff_planner_softmax_temperature
        self.len_costmap = self.params.len_costmap
        self.uncertainty_amount = 1.0
        self.noise = 0.0
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)

        tfe = tf.contrib.eager
        self.costs = tfe.Variable(np.zeros((self.len_costmap)))
        self.uncertainties = tfe.Variable(np.zeros((self.len_costmap)))

        self.pre_determined_uncertainties = \
            tf.random_uniform([self.len_costmap], dtype=tf.double) * self.uncertainty_amount
        self.pre_determined_noise = np.random.rand(self.len_costmap)*self.noise

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

    def planner_loss(self, gt_traj, costmap, uncertainties, desired_gradient, analytical=False):
        '''
        We're not actually training the planner but this is important
        for the definition of the planner gradient.
        Part of the computation involves computing the planner internal 
        cost of all candidate sampled spline trajectories, in which all 
        candidate samples' NN-provided costs and uncertainties are given by 
        'costmap' and 'uncertainties'.
        '''
        with tf.GradientTape() as tape:
            self.costs.assign(costmap)
            self.uncertainties.assign(uncertainties)
            planner_internal_costs = self.costs + self.theta * self.uncertainties
            softmax = tf.nn.softmax(-self.softmax_beta * planner_internal_costs)
            clipped_softmax = tf.clip_by_value(softmax, EPSILON, 1 - EPSILON)
            loss = -tf.reduce_sum(gt_traj * tf.log(clipped_softmax)) # softmax cross-entropy
        if desired_gradient == 'loss_grads':
            grads = tape.gradient(loss, [clipped_softmax, planner_internal_costs, self.costs, self.uncertainties])
        elif desired_gradient == 'softmax_grads':
            grads = tape.gradient(clipped_softmax, [planner_internal_costs, self.costs, self.uncertainties])
        elif desired_gradient == 'cost_grads':
            grads = tape.gradient(planner_internal_costs, [self.costs, self.uncertainties])
        
        if analytical:
            print("\nDouble-checking gradient calculation analytically")
            softmax_numerator = tf.exp(-self.softmax_beta * planner_internal_costs)
            denom = tf.reduce_sum(softmax_numerator)
            print("Softmax numerator", softmax_numerator)
            print("Denom", denom)
            gt_planner_internal_cost = tf.reduce_sum(gt_traj * planner_internal_costs)
            exp_gt_index = tf.exp(-self.softmax_beta * gt_planner_internal_cost)
            print("Exp gt index", exp_gt_index)
            gt_softmax = tf.reduce_sum(gt_traj * clipped_softmax)
            dloss_dsoftmax = -(1/gt_softmax)
            dsoftmax_dplannerintcost = (-self.softmax_beta * exp_gt_index) * \
                                        (denom - exp_gt_index) / (denom * denom)
            print("dloss/dsoftmax:", dloss_dsoftmax)
            print("dsoftmax/dplannerintcost:", dsoftmax_dplannerintcost)
            gradient_wrt_uncertainty_gt = self.theta * self.softmax_beta * \
                                        (denom - exp_gt_index)/denom  # Simplified
            gradient_wrt_uncertainty_non_gt = -self.theta * self.softmax_beta * \
                                        softmax_numerator / denom  # Simplified
            print("Gradient wrt uncertainty, ground truth: ", gradient_wrt_uncertainty_gt)
            print("And that should be the product of dloss/dsoftmax and dsoftmax/dplannerintcost and theta:", 
                            self.theta * dloss_dsoftmax * dsoftmax_dplannerintcost)
            print("Gradient wrt uncertainty, non ground truth: ", gradient_wrt_uncertainty_non_gt)

        
        # Loss_grads should be equal to d_loss/d_softmax * d_softmax/d_planner_internal_costs * 
        # d_planner_internal_costs/d_uncertainties
        return planner_internal_costs, loss, grads

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

        return start_configs, costmaps
    
    def get_gradient_one_data_point(self, file_num='1', batch_index=0, 
                                    uncertainty_mode='random', desired_gradient='loss_grads'):
        '''
        This function is given one data point from the dataset of ground 
        truth costmaps. It simulates the neural network output from that
        and then computes the planner gradient. This is a dummy implementation;
        eventually the simulated NN output should be replaced by the real NN
        output.
        '''
        dummy_start_config, true_costmap_n4 = self.get_data_from_pickle(file_num, batch_index)
        # print("\nGOT DATA FROM PICKLE: START CONFIG", dummy_start_config)
        # print("TRUE COSTMAP", true_costmap_n4)

        # Ground truth trajectory will be the one that minimizes the ground 
        # truth costs 
        gt_min_idx = np.argmin(true_costmap_n4[:, 3])
        gt_traj = np.zeros(self.len_costmap)
        gt_traj[gt_min_idx] = 1.0  # One-hot vector
        gt_traj = tf.convert_to_tensor(gt_traj)

        # print("\nGT TRAJ", gt_traj)

        # Simulated network output -- ground truth costs + noise
        nn_output_n4 = np.copy(true_costmap_n4)
        nn_output_n4[:, 3] = nn_output_n4[:, 3] + self.pre_determined_noise 

        uncertainties = self.get_uncertainties(uncertainty_mode, nn_costs=nn_output_n4[:, 3])
        planner_internal_costs, loss, grads = \
            self.planner_loss(gt_traj, nn_output_n4[:, 3], uncertainties, desired_gradient)
        
        gradients = np.array([np.array(elem) for elem in grads])
        norm_of_grad_uncertainty = np.linalg.norm(gradients[-1])  # Norm of gradient wrt uncertainty

        # print("\nNN COSTMAP", nn_output_n4[:, 3])
        # print("\nUNCERTAINTIES", uncertainties)
        # print("\nPLANNER INTERNAL COSTS", planner_internal_costs)
        # print("\nPLANNER LOSS", loss.numpy())
        # if desired_gradient == 'loss_grads':
        #     print('\nPLANNER GRADIENTS', gradients)
        # if desired_gradient == 'softmax_grads':
        #     print('\nSOFTMAX GRADIENTS', gradients)
        # if desired_gradient == 'cost_grads':
        #     print('\nCOST GRADIENTS', gradients)
        # print("\nNORM OF GRAD UNCERTAINTY", norm_of_grad_uncertainty)

        return dummy_start_config, true_costmap_n4, nn_output_n4, uncertainties, \
            planner_internal_costs, loss, gradients, norm_of_grad_uncertainty

    def get_gradients_dataset(self, num_data_points, per_file, num_files=70, 
                            uncertainty_mode='random', desired_gradient='loss_grads'):
        '''
        Calls get_gradient_one_data_point on multiple file and batch numbers
        num_data_points: The total number of data points the gradient 
                        computation is desired for
        per_file: How many data points should come from each file
        num_files: The total number of files available
        uncertainty_mode: The type of hand-coded uncertainty scheme
        '''
        losses = []
        gradients_list = []
        gradient_norms = []
        gradient_on_gts = []
        gradient_on_non_gts = []
        num_files_to_sample_from = int(num_data_points/per_file) 
        file_indices = np.random.choice(np.arange(1, num_files + 1), num_files_to_sample_from, replace=False)
        num_batches = 1000   # hard-coded value -- saves the effort of loading every pickle file to check
        for file_index in file_indices:
            batch_indices = np.random.choice(num_batches, per_file, replace=False)
            for batch_index in batch_indices:
                dummy_start_config, true_costmap_n4, nn_output_n4, \
                uncertainties, planner_internal_costs, loss, \
                gradients, norm_of_grad_uncertainty = \
                    self.get_gradient_one_data_point(str(file_index), batch_index, \
                                                    uncertainty_mode, desired_gradient)
                # print("\nPlanner Internal Costs", planner_internal_costs)
                # TODO (sdeglurkar): Unclean code -- assuming that GT index is -1
                losses.append(loss.numpy())
                gradients_list.append(gradients[-1])
                gradient_norms.append(norm_of_grad_uncertainty)
                gradient_on_gts.append(gradients[-1][-1])
                gradient_on_non_gts.append(np.linalg.norm(gradients[-1][:-1]))
        
        return losses, gradients_list, gradient_norms, gradient_on_gts, gradient_on_non_gts

    def optimize(self, start_config):
        """ 
        Expects that the neural network will output a set of spline
        trajectory samples (represented by their endpoints), their 
        corresponding costs, and uncertainties. Incorporates this cost 
        and uncertainty into the planner internal cost function and 
        chooses the trajectory with the lowest planner internal cost.
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

        dummy_start_config, true_costmap_n4, nn_output_n4, \
            uncertainties, planner_internal_costs, loss, \
            gradients, norm_of_grad_uncertainty = self.get_gradient_one_data_point()

        # Minimize the planner internal cost
        min_idx = tf.argmin(planner_internal_costs)
        min_cost = planner_internal_costs[min_idx]  # Just here for debugging
        
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

        # Visualize gradient wrt uncertainty
        self.visualize_gradients(true_costmap_n4[:, 3], nn_output_n4[:, 3], uncertainties, 
                                    planner_internal_costs, gradients[-1])
        
        num_data_points = 1000
        per_file = 50
        losses, gradients_list, gradient_norms, gradient_on_gts, gradient_on_non_gts = \
                            self.get_gradients_dataset(num_data_points, per_file, \
                                                        uncertainty_mode='proportional_to_cost')

        self.visualize_dataset_gradients(losses, gradients_list, gradient_norms, 
                                        gradient_on_gts, gradient_on_non_gts)
        
        data = {'system_config': dummy_start_sys_config,
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'min_planner_internal_cost': min_cost.numpy(),
                'planner_loss': loss.numpy(),
                'planner_gradients': gradients,
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][idx:idx + 1],
                'k_nkf1': controllers['k_nkf1'][idx:idx + 1]}

        return data

    def visualize_gradients(self, true_costmap, nn_costmap, uncertainties, 
                            planner_internal_costs, uncertainty_gradients):
        true_costmap = np.clip(true_costmap, 0, PLOTTING_CLIP_VALUE)
        nn_costmap = np.clip(nn_costmap, 0, PLOTTING_CLIP_VALUE)
        planner_internal_costs = np.clip(planner_internal_costs, 0, PLOTTING_CLIP_VALUE)

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
