from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

import numpy as np
import yaml


@dataclass
class SurveyParams:
    """Configuration for a rectangular or circular coverage survey."""

    area_width: float = 20
    area_height: float = 20
    area_origin_x: float = 0
    area_origin_y: float = 0
    waypoint_spacing: float = 0.99
    track_spacing: float = 2
    robot_body_height: float = 0.5
    robot_start_x: float = 0.0
    robot_start_y: float = 0.0
    pattern_yaw: float = 0.0

    # Additional parameters for snake-ordered, Dubins-smoothed coverage path
    robot_width: float = 0.53
    min_turning_radius: float = 1.
    headland_width: float = 1.5 * robot_width
    path_style: str = 'rectangular-snake'
    circle_diameter: float = 0.0
    circular_start_at_robot: bool = False

    def __post_init__(self):
        if self.headland_width is None:
            self.headland_width = 1.5 * self.robot_width


@dataclass
class SurveyWaypoint:
    index: int
    x: float
    y: float
    z: float


T = TypeVar('T')


def load_params(
    yaml_path: str,
    planner_name: str,
    params_class: Type[T],
) -> T:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    params = {
        **config.get('common', {}),
        **config.get(planner_name, {}),
    }

    return params_class(**params)


class AbstractPlanner(ABC):
    """Base interface for survey waypoint planners."""

    def __init__(self, params: SurveyParams):
        self.params = params

    @abstractmethod
    def plan(self, vertices: np.ndarray):
        pass
