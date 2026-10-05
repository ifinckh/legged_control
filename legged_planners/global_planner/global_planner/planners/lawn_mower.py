import os

import fields2cover as f2c
import numpy as np

from .abstract import (
    AbstractPlanner,
    load_params,
    SurveyParams,
    SurveyWaypoint,
)


def plot_plan(
    params: SurveyParams,
    waypoints: list[SurveyWaypoint],
) -> None:
    """Show the survey map and path in 2D, hiding waypoint z values."""
    matplotlib_config_dir = '/tmp/matplotlib'
    os.makedirs(matplotlib_config_dir, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', matplotlib_config_dir)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs = [waypoint.x for waypoint in waypoints]
    ys = [waypoint.y for waypoint in waypoints]

    fig, ax = plt.subplots(figsize=(9, 7))
    area = Rectangle(
        (params.area_origin_x, params.area_origin_y),
        params.area_width,
        params.area_height,
        facecolor='#d8f3dc',
        edgecolor='#1b4332',
        linewidth=2.0,
        label='Survey area',
    )
    ax.add_patch(area)
    ax.plot(xs, ys, color='#1d3557', linewidth=1.8, label='Lawn-mower path')

    waypoint_order = range(len(waypoints))
    scatter = ax.scatter(
        xs,
        ys,
        c=list(waypoint_order),
        cmap='viridis',
        s=28,
        edgecolors='white',
        linewidths=0.5,
        label='Waypoints (x, y)',
        zorder=3,
    )
    fig.colorbar(scatter, ax=ax, label='Waypoint order')

    ax.scatter(
        [params.robot_start_x],
        [params.robot_start_y],
        marker='*',
        s=180,
        color='#d62828',
        edgecolors='white',
        linewidths=0.8,
        label='Robot start',
        zorder=4,
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        marker='X',
        s=90,
        color='#f77f00',
        edgecolors='white',
        linewidths=0.8,
        label='Path end',
        zorder=4,
    )

    padding = max(params.area_width, params.area_height, 1.0) * 0.08
    ax.set_xlim(
        min(params.area_origin_x, params.robot_start_x) - padding,
        max(
            params.area_origin_x + params.area_width,
            params.robot_start_x,
        ) + padding,
    )
    ax.set_ylim(
        min(params.area_origin_y, params.robot_start_y) - padding,
        max(
            params.area_origin_y + params.area_height,
            params.robot_start_y,
        ) + padding,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Rectangular Lawn-Mower Survey Plan')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.45)
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.show()


class LawnMowerPlanner(AbstractPlanner):
    """Generate a lawn-mower coverage route with Fields2Cover."""

    def __init__(self, params):
        super().__init__(params)

    def plan(self, vertices: np.ndarray):

        # rand = f2c.Random(42)
        # field = rand.generateRandField(1e4, 5)
        # cells = field.getField()

        ring = f2c.LinearRing(
            f2c.VectorPoint(
                [
                    f2c.Point(vertices[i, 0], vertices[i, 1])
                    for i in range(len(vertices))
                ]
            )
        )
        cells = f2c.Cells(f2c.Cell(ring))

        # Robot configuration
        robot = f2c.Robot(self.params.robot_width, self.params.track_spacing)
        robot.setMinTurningRadius(self.params.min_turning_radius)

        headland_generator = f2c.HG_Const_gen()
        no_hl = headland_generator.generateHeadlands(
            cells,
            self.params.headland_width,
        )

        swath_generator = f2c.SG_BruteForce()
        swaths = swath_generator.generateSwaths(
            np.pi + self.params.pattern_yaw,
            robot.getCovWidth(),
            no_hl.getGeometry(0),
        )

        # Snake plan
        snake_sorter = f2c.RP_Snake()

        # candidates = [
        #     snake_sorter.genSortedSwaths(swaths, variant)
        #     for variant in range(4)
        # ]

        # def distance_from_robot(candidate):
        #     point = candidate.at(0).startPoint()
        #     return math.hypot(
        #         point.getX() - self.params.robot_start_x,
        #         point.getY() - self.params.robot_start_y,
        #     )

        # swaths = min(candidates, key=distance_from_robot)
                
        swaths = snake_sorter.genSortedSwaths(swaths, 1)

        # Dubins
        path_planner = f2c.PP_PathPlanning()
        dubins = f2c.PP_DubinsCurves()
        dubins.setDiscretization(0.1)

        smooth_path = path_planner.planPath(robot, swaths, dubins)
        smooth_path = smooth_path.discretizeSwath(
            self.params.waypoint_spacing
        )

        waypoints: list[SurveyWaypoint] = []
        z = self.params.robot_body_height

        for i in range(smooth_path.size()):
            state = smooth_path.getState(i)

            waypoints.append(
                SurveyWaypoint(
                    index=i,
                    x=state.point.getX(),
                    y=state.point.getY(),
                    z=z,
                )
            )

        # f2c.Visualizer.figure()
        # f2c.Visualizer.plot(no_hl)
        # f2c.Visualizer.plot(smooth_path)
        # f2c.Visualizer.plot(swaths)
        # f2c.Visualizer.show()

        # return smooth_path

        # for i in range(smooth_path.size()):
        #     state = smooth_path.getState(i)

        #     keep_state = True
        #     if state.type == f2c.PathSectionType_TURN:
        #         if (
        #             previous_state is not None
        #             and previous_state.type == f2c.PathSectionType_TURN
        #         ):
        #             turn_distance += abs(previous_state.len)
        #             keep_state = (
        #                 turn_distance >= self.params.waypoint_spacing
        #             )
        #             if keep_state:
        #                 turn_distance = 0.0
        #         else:
        #             turn_distance = 0.0
        #     else:
        #         turn_distance = 0.0

        #     if keep_state:
        #         waypoints.append(
        #             SurveyWaypoint(
        #                 x=state.point.getX(),
        #                 y=state.point.getY(),
        #                 z=z,
        #             )
        #         )

        #     previous_state = state

        return waypoints


if __name__ == '__main__':

    vertices = np.array([
        [0., 0.],
        [20., 0.],
        [20., 20.],
        [0., 20.]
    ])
    params = load_params('../config/planner.yaml', 'lawn_mower', SurveyParams)
    planner = LawnMowerPlanner(params)
    plan = planner.plan(vertices)
