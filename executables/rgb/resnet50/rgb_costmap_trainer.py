from training_utils.visual_navigation_trainer import VisualNavigationTrainer
from models.visual_navigation.rgb.resnet50.rgb_resnet50_costmap_model import RGBResnet50CostmapModel
import os


class RGBCostmapTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regresses on the optimal waypoint and its cost using rgb images.
    """
    simulator_name = 'RGB_Resnet50_NN_Waypoint_Simulator'

    def create_model(self, params=None):
        self.model = RGBResnet50CostmapModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNWaypointCostPlanner
        """
        from planners.nn_costmap_planner import NNCostmapPlanner

        p.planner_params.planner = NNCostmapPlanner
        p.planner_params.model = self.model
    
    # Custom data generation for this type of training - hence we need
    # to overwrite generate_data() and create_data_source().
    def generate_data(self, params=None):
        """
        Generate the data using the data source.
        """
        from planners.extended_sampling_costs_planner import ExtendedSamplingCostsPlanner

        self.p.simulator_params.planner_params.planner = ExtendedSamplingCostsPlanner

        super().generate_data(params)

    def create_data_source(self, params=None):
        from data_sources.costmap_data_source import CostmapDataSource
        self.data_source = CostmapDataSource(self.p)

        if hasattr(self, 'model'):
            # Give the visual_navigation data source access to the model.
            # May be needed to render training images, etc.
            self.data_source.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_waypoint')


if __name__ == '__main__':
    RGBCostmapTrainer().run()
