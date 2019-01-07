from dotmap import DotMap
from obstacles.turtlebot_map import TurtlebotMap


def create_params():
    p = DotMap()

    p.obstacle_map = TurtlebotMap

    return p
