from models.visual_navigation.waypoint_cost_model import VisualNavigationWaypointCostModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase


class RGBResnet50WaypointCostModel(Resnet50ModelBase, VisualNavigationWaypointCostModel):
    """
    A model that regresses upon optimal waypoints (in 3d space) and their costs
    given an rgb image.
    """
    name = 'RGB_Resnet50_Waypoint_Cost_Model'
