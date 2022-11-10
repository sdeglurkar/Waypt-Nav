import os
from dotmap import DotMap
from params.base_data_directory import base_data_dir


def create_rgb_trainer_params():
    from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params

    from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
    from params.model.resnet50_arch_v1_params import create_params as create_model_params

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is projected SBPD Grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875

    # Change episode horizon
    simulator_params.episode_horizon_s = 80.0

    # Ensure the renderer is using area3
    simulator_params.obstacle_map_params.renderer_params.building_name = 'area1'

    # Save trajectory data
    simulator_params.save_trajectory_data = True

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    return p


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 4  # (x, y, theta) waypoint, cost
    
    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]
    
    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    # TODO (sdeglurkar) This is a hack -- have to check exactly how much data is 
    # generated during data generation
    p.trainer.num_samples = 99900 #int(125e3)
    
    # Checkpoint settings
    p.trainer.ckpt_save_frequency = 1
    p.trainer.restore_from_ckpt = False

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'

    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
        p=0.1,  # Probability of distortion
        version='v1'
    )

    # Checkpoint directory
    p.trainer.ckpt_path = '/home/sampada_deglurkar/Waypt-Nav/reproduce_LB_WayptNavResults/session_2022-11-02_10-00-27/checkpoints/ckpt-100'
    
    # Change the data_dir
    p.data_creation.data_dir = ['/home/ext_drive/sampada_deglurkar/dummy_wayptnav_data']

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 500

    # Test the network only on goals where the expert succeeded
    p.test.expert_success_goals = DotMap(use=True,
                                         dirname=os.path.join(base_data_dir(), 'expert_success_goals/sbpd_projected_grid'))
    
    # Let's not look at the expert
    p.test.simulate_expert = True
    
    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )
    
    return p
