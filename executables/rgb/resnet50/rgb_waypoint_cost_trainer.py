from training_utils.visual_navigation_trainer import VisualNavigationTrainer
from models.visual_navigation.rgb.resnet50.rgb_resnet50_waypoint_cost_model import RGBResnet50WaypointCostModel
import os


class RGBWaypointCostTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regresses on the optimal waypoint and its cost using rgb images.
    """
    simulator_name = 'RGB_Resnet50_NN_Waypoint_Simulator'

    def create_model(self, params=None):
        self.model = RGBResnet50WaypointCostModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNWaypointCostPlanner
        """
        from planners.nn_waypoint_cost_planner import NNWaypointCostPlanner

        p.planner_params.planner = NNWaypointCostPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_waypoint')


if __name__ == '__main__':
    RGBWaypointCostTrainer().run()
