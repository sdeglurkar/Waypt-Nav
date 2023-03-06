from models.visual_navigation.costmap_model import VisualNavigationCostmapModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase


class RGBResnet50CostmapModel(Resnet50ModelBase, VisualNavigationCostmapModel):
    """
    A model that regresses upon optimal waypoints (in 3d space) and their costs
    given an rgb image.
    """
    name = 'RGB_Resnet50_Costmap_Model'
