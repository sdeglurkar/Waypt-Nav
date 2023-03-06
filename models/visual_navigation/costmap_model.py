from models.visual_navigation.base import VisualNavigationModelBase
import numpy as np

class VisualNavigationCostmapModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image
    (and potentially other inputs), returns an optimal
    waypoint and its cost.
    """
    def _optimal_labels(self, raw_data):
        """
        Supervision for the costmap.
        """
        costmap = raw_data['costmap']
        return costmap
