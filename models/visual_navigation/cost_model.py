from models.visual_navigation.base import VisualNavigationModelBase
import numpy as np
import tensorflow as tf

class VisualNavigationCostModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image,
    a candidate waypoint, and other inputs, returns a cost.
    """
    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        
        if self.p.data_processing.input_processing_function is not None:
            raw_data = self.preprocess_nn_input(raw_data, is_training)

        # Get the input image (n, m, k, d)
        # batch size n x (m x k pixels) x d channels
        img_nmkd = raw_data['img_nmkd']

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        goal_position = self._goal_position(raw_data)
        vehicle_controls = self._vehicle_controls(raw_data)
        state_features_n4 = tf.concat([goal_position, vehicle_controls], axis=1)

        # Optimal Supervision
        optimal_labels_n = self._optimal_labels(raw_data)
        
        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [img_nmkd, state_features_n4]
        data['labels'] = optimal_labels_n
        return data

    def _optimal_labels(self, raw_data):
        """
        Supervision for the costs.
        """
        optimal_cost = raw_data['all_waypoints_costs']
        return optimal_cost 
