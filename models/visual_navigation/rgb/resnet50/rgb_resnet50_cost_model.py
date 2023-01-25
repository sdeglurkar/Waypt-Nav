from models.visual_navigation.waypoint_cost_model import VisualNavigationWaypointCostModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase


class RGBResnet50WaypointCostModel(Resnet50ModelBase, VisualNavigationWaypointCostModel):
    """
    A model that regresses upon costs given candidate waypoints and an rgb image.
    """
    name = 'RGB_Resnet50_Cost_Model'
