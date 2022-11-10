from models.visual_navigation.base import VisualNavigationModelBase
import numpy as np

class VisualNavigationWaypointCostModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image
    (and potentially other inputs), returns an optimal
    waypoint and its cost.
    """
    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal waypoints.
        """
        optimal_waypoints_n3 = raw_data['optimal_waypoint_ego_n3']
        optimal_cost = raw_data['costmap']
        optimal_cost = np.expand_dims(optimal_cost, axis=1)
        return np.hstack((optimal_waypoints_n3, optimal_cost))
