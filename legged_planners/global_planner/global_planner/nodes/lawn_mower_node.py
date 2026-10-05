import math
from pathlib import Path
import select
import sys
import termios
import threading
import tty
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path as PathMessage
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ..planners.abstract import load_params, SurveyWaypoint
from ..planners.lawn_mower import LawnMowerPlanner, SurveyParams


def validate_positive(name: str, value: float) -> float:
    """Return a float parameter value that must be greater than zero."""
    number = float(value)
    if number <= 0.0:
        raise ValueError(f'{name} must be greater than 0.0')
    return number


def yaw_from_quaternion(orientation) -> float:
    """Return planar yaw from a ROS quaternion."""
    return math.atan2(
        2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        ),
        1.0 - 2.0 * (
            orientation.y * orientation.y
            + orientation.z * orientation.z
        ),
    )


class SurveyWaypointNode(Node):
    """Publish survey waypoints and advance using robot pose feedback."""

    def __init__(self) -> None:
        """Initialize publishers, subscriptions, and waypoint state."""
        super().__init__('survey_waypoint_node')

        self.declare_parameter('goal_topic', 'move_base_simple/goalddd')
        self.declare_parameter('waypoint_topic', 'survey_waypoints')
        self.declare_parameter('command_topic', 'survey_waypoint_command')
        self.declare_parameter(
            'robot_pose_topic',
            '/legged_odometry/pose_in_odom',
        )
        self.declare_parameter('robot_odom_topic', 'odom')
        self.declare_parameter('goal_frame_id', '')
        self.declare_parameter('distance_threshold', 0.8)
        self.declare_parameter('distance_log_period', 2.0)
        self.declare_parameter('publish_period', 1.0)
        self.declare_parameter('waypoint_path_publish_period', 1.0)
        self.declare_parameter('stamp_goals', False)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('keyboard_control', True)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('use_current_pose_as_origin', False)

        # RViz
        self.declare_parameter('rviz_waypoints_topic', 'survey_waypoint_markers')
        self.declare_parameter('goal_match_tolerance', 0.05)
        self.declare_parameter('line_width', 0.05)
        self.declare_parameter('arrow_length', 0.65)
        self.declare_parameter('arrow_z_offset', 0.15)

        self.distance_threshold = float(self.get_parameter(
            'distance_threshold',
        ).value)
        self.distance_log_period = max(
            float(self.get_parameter('distance_log_period').value),
            0.0,
        )
        publish_period = self.get_parameter('publish_period').value
        waypoint_path_publish_period = max(
            float(
                self.get_parameter('waypoint_path_publish_period').value
            ),
            0.1,
        )
        self.stamp_goals = self.get_parameter('stamp_goals').value
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.keyboard_control = bool(
            self.get_parameter('keyboard_control').value
        )
        # self.path_style = normalize_path_style(
        #     self.get_parameter('path_style').value
        # )
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.use_current_pose_as_origin = bool(
            self.get_parameter('use_current_pose_as_origin').value
        )
        goal_frame_id = self.get_parameter('goal_frame_id').value
        if goal_frame_id:
            self.frame_id = goal_frame_id

        goal_topic = self.get_parameter('goal_topic').value
        waypoint_topic = self.get_parameter('waypoint_topic').value
        command_topic = self.get_parameter('command_topic').value
        robot_pose_topic = self.get_parameter('robot_pose_topic').value
        robot_odom_topic = self.get_parameter('robot_odom_topic').value
        self.origin_capture_source = self.select_origin_capture_source(
            robot_pose_topic,
            robot_odom_topic,
        )

        self.goal_match_tolerance = float(
            self.get_parameter('goal_match_tolerance').value
        )

        self.line_width = float(self.get_parameter('line_width').value)
        self.arrow_length = float(self.get_parameter('arrow_length').value)
        self.arrow_z_offset = float(
            self.get_parameter('arrow_z_offset').value
        )

        self.waypoints: List[SurveyWaypoint] = []
        self.waypoint_path_message: Optional[PathMessage] = None
        self.current_waypoint_index = 0
        self.last_logged_waypoint_index = None
        self.last_distance_log_time = None
        self.warned_feedback_frame_ids = set()
        self.completed = False
        self.started = self.auto_start
        self.paused = not self.auto_start
        self.origin_captured = False
        self.keyboard_stop_event = threading.Event()
        self.keyboard_input = None
        self.keyboard_thread = None

        waypoint_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.waypoint_path_publisher = self.create_publisher(
            PathMessage,
            waypoint_topic,
            waypoint_qos,
        )
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            goal_topic,
            10,
        )
        self.command_subscription = None
        if command_topic:
            self.command_subscription = self.create_subscription(
                String,
                command_topic,
                self.command_callback,
                10,
            )
        self.pose_subscription = None
        if robot_pose_topic:
            self.pose_subscription = self.create_subscription(
                PoseWithCovarianceStamped,
                robot_pose_topic,
                self.robot_pose_callback,
                10,
            )
        self.odom_subscription = None
        if robot_odom_topic:
            self.odom_subscription = self.create_subscription(
                Odometry,
                robot_odom_topic,
                self.robot_odom_callback,
                10,
            )

        rviz_waypoints_topic = self.get_parameter('rviz_waypoints_topic').value
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            rviz_waypoints_topic,
            marker_qos,
        )

        if not self.use_current_pose_as_origin:
            self.build_waypoints(0.0, 0.0)
        elif self.origin_capture_source is None:
            self.get_logger().warning(
                'use_current_pose_as_origin is true, but both robot feedback '
                'topics are disabled. No survey plan can be generated.'
            )

        self.waypoint_path_timer = self.create_timer(
            waypoint_path_publish_period,
            self.publish_waypoint_path,
        )
        self.publish_waypoint_path()
        self.publish_timer = self.create_timer(
            max(publish_period, 0.05),
            self.publish_current_waypoint,
        )
        if self.started:
            self.publish_current_waypoint()
        if self.keyboard_control:
            self.start_keyboard_listener()

        if self.waypoints:
            self.get_logger().info(
                'Generated '
                f'{len(self.waypoints)} waypoints. Publishing goals on '
                f'"{goal_topic}".'
            )
        else:
            self.get_logger().info(
                'Waiting for the first '
                f'{self.origin_capture_source} feedback pose to generate '
                f'the survey plan. Goals will publish on "{goal_topic}".'
            )
        self.get_logger().info(
            'Publishing complete waypoint path on '
            f'"{waypoint_topic}" for visualizers and bridges.'
        )
        if command_topic:
            self.get_logger().info(
                f'Reading optional waypoint commands from "{command_topic}".'
            )
        if robot_pose_topic:
            self.get_logger().info(
                f'Reading PoseWithCovarianceStamped feedback from '
                f'"{robot_pose_topic}".'
            )
        if robot_odom_topic:
            self.get_logger().info(
                f'Reading Odometry feedback from "{robot_odom_topic}".'
            )
        if self.started:
            self.get_logger().info(
                'Waypoint planner is running. Press "p" to pause, '
                '"f" to skip forward, or "b" to skip backward.'
            )
        else:
            self.get_logger().info(
                'Waypoint planner is waiting. Press "s" to start, '
                '"f" to skip forward, or "b" to skip backward.'
            )

    def select_origin_capture_source(
        self,
        robot_pose_topic: str,
        robot_odom_topic: str,
    ) -> Optional[str]:
        """Choose the feedback stream used to capture planner zero."""
        if robot_pose_topic:
            return 'pose'
        if robot_odom_topic:
            return 'odom'
        return None

    def make_survey_params(
        self,
        robot_start_x: float,
        robot_start_y: float,
        pattern_yaw: float,
    ) -> SurveyParams:
        """Read and validate survey generation parameters."""
        waypoint_spacing = validate_positive(
            'waypoint_spacing',
            self.get_parameter('waypoint_spacing').value,
        )
        track_spacing = validate_positive(
            'track_spacing',
            self.get_parameter('track_spacing').value,
        )
        area_origin_x = float(self.get_parameter('area_origin_x').value)
        area_origin_y = float(self.get_parameter('area_origin_y').value)
        circle_diameter = validate_positive(
            'circle_diameter',
            self.get_parameter('circle_diameter').value,
        )
        circular_start_at_robot = (
            self.use_current_pose_as_origin
            # and self.path_style in {CIRCULAR_SNAKE, CIRCULAR_SPIRAL}
        )
        planner_yaw = pattern_yaw
        if self.use_current_pose_as_origin:
            if circular_start_at_robot:
                radius = circle_diameter / 2.0
                area_origin_x = (
                    robot_start_x + radius * math.cos(pattern_yaw) - radius
                )
                area_origin_y = (
                    robot_start_y + radius * math.sin(pattern_yaw) - radius
                )
                # if self.path_style == CIRCULAR_SNAKE:
                #     planner_yaw = pattern_yaw + math.pi / 2.0
                # else:
                #     planner_yaw = pattern_yaw + math.pi
                planner_yaw = pattern_yaw + math.pi
            else:
                area_origin_x += robot_start_x
                area_origin_y += robot_start_y

        return SurveyParams(
            area_width=validate_positive(
                'area_width',
                self.get_parameter('area_width').value,
            ),
            area_height=validate_positive(
                'area_height',
                self.get_parameter('area_height').value,
            ),
            area_origin_x=area_origin_x,
            area_origin_y=area_origin_y,
            waypoint_spacing=waypoint_spacing,
            track_spacing=track_spacing,
            robot_body_height=validate_positive(
                'robot_body_height',
                self.get_parameter('robot_body_height').value,
            ),
            robot_start_x=robot_start_x,
            robot_start_y=robot_start_y,
            pattern_yaw=planner_yaw,
            robot_width=validate_positive(
                'robot_width',
                self.get_parameter('robot_width').value,
            ),
            min_turning_radius=validate_positive(
                'min_turning_radius',
                self.get_parameter('min_turning_radius').value,
            ),
            headland_width=validate_positive(
                'headland_width',
                self.get_parameter('headland_width').value,
            ),
            path_style=self.path_style,
            circle_diameter=circle_diameter,
            circular_start_at_robot=circular_start_at_robot,
        )

    def build_waypoints(
        self,
        robot_start_x: float,
        robot_start_y: float,
        pattern_yaw: float = 0.0,
    ) -> None:
        """Generate the active survey waypoints from node parameters."""
        # TODO add rotation, remove hard coded code
        vertices = np.array([
                [0., 0.],
                [20., 0.],
                [20., 20.],
                [0., 20.]
            ])
        vertices[:, 0] += robot_start_x
        vertices[:, 1] += robot_start_y
        print(vertices)
        planner_config = (
            Path(get_package_share_directory('global_planner'))
            / 'config'
            / 'planner.yaml'
        )
        params = load_params(
            str(planner_config),
            'lawn_mower',
            SurveyParams,
        )
        self.path_style = params.path_style
        planner = LawnMowerPlanner(params)
        self.waypoints = planner.plan(vertices)
        if not self.waypoints:
            raise RuntimeError('Generated survey plan has no waypoints.')

        self.current_waypoint_index = min(
            self.current_waypoint_index,
            len(self.waypoints) - 1,
        )
        self.waypoint_path_message = self.create_waypoint_path_message()
        self.origin_captured = True
        self.get_logger().info(
            'Generated '
            f'{len(self.waypoints)} {self.path_style} waypoints from '
            f'planner origin ({robot_start_x:.3f}, {robot_start_y:.3f}) '
            f'and yaw {pattern_yaw:.3f} rad.'
        )
        if params.circular_start_at_robot:
            radius = params.circle_diameter / 2.0
            self.get_logger().info(
                'Aligned circular survey: '
                f'diameter={params.circle_diameter:.3f} m, '
                f'center=({params.area_origin_x + radius:.3f}, '
                f'{params.area_origin_y + radius:.3f}), '
                'with the captured robot pose on the boundary.'
            )
        self.publish_waypoint_path()

    def create_waypoint_path_message(self) -> PathMessage:
        """Create a Path message with the complete survey waypoint list."""
        message = PathMessage()
        message.header.frame_id = self.frame_id
        for waypoint_index, waypoint in enumerate(self.waypoints):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = waypoint.x
            pose.pose.position.y = waypoint.y
            pose.pose.position.z = waypoint.z
            self.set_route_orientation(pose.pose, waypoint_index)
            message.poses.append(pose)

        return message

    def set_route_orientation(self, pose, waypoint_index: int) -> None:
        """Orient a pose along the route at the requested waypoint."""
        yaw = self.route_yaw(waypoint_index)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw * 0.5)
        pose.orientation.w = math.cos(yaw * 0.5)

    def route_yaw(self, waypoint_index: int) -> float:
        """Return the yaw that points from a waypoint along the route."""
        direction_x, direction_y = self.route_direction(waypoint_index)
        return math.atan2(direction_y, direction_x)

    def route_direction(self, waypoint_index: int) -> tuple[float, float]:
        """Return a normalized route direction for a waypoint."""
        waypoint = self.waypoints[waypoint_index]

        for other in self.waypoints[waypoint_index + 1:]:
            dx = other.x - waypoint.x
            dy = other.y - waypoint.y
            length = math.hypot(dx, dy)
            if length >= 1e-6:
                return dx / length, dy / length

        for other in reversed(self.waypoints[:waypoint_index]):
            dx = waypoint.x - other.x
            dy = waypoint.y - other.y
            length = math.hypot(dx, dy)
            if length >= 1e-6:
                return dx / length, dy / length

        return 1.0, 0.0

    def publish_waypoint_path(self) -> None:
        """Publish the complete waypoint list for visualization and bridge."""
        if self.waypoint_path_message is None:
            return

        now = self.get_clock().now().to_msg()
        self.waypoint_path_message.header.stamp = now
        for pose in self.waypoint_path_message.poses:
            pose.header.stamp = now

        self.waypoint_path_publisher.publish(self.waypoint_path_message)
        self.publish_marker_array()

    def publish_current_waypoint(self, force: bool = False) -> None:
        """Publish the active waypoint as a PoseWithCovarianceStamped goal."""
        if self.completed or (not self.started and not force):
            return
        if not self.waypoints:
            return

        waypoint = self.waypoints[self.current_waypoint_index]
        message = PoseStamped()
        if self.stamp_goals:
            message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.pose.position.x = waypoint.x
        message.pose.position.y = waypoint.y
        message.pose.position.z = waypoint.z
        self.set_route_orientation(
            message.pose,
            self.current_waypoint_index,
        )

        self.goal_publisher.publish(message)
        self.publish_marker_array()
        if self.last_logged_waypoint_index != self.current_waypoint_index:
            self.last_logged_waypoint_index = self.current_waypoint_index
            self.get_logger().info(
                'Published waypoint '
                f'{waypoint.index}: x={waypoint.x:.3f}, '
                f'y={waypoint.y:.3f}, z={waypoint.z:.3f}, '
                f'yaw={self.route_yaw(self.current_waypoint_index):.3f}'
            )

    def robot_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        """Use a PoseWithCovarianceStamped message as robot pose feedback."""
        self.handle_robot_position(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
            'pose',
            message.header.frame_id,
        )

    def robot_odom_callback(self, message: Odometry) -> None:
        """Use an Odometry message as robot pose feedback."""
        self.handle_robot_position(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
            'odom',
            message.header.frame_id,
        )

    def command_callback(self, message: String) -> None:
        """Use a String message as a start/pause/skip command."""
        command = message.data.strip().lower()
        if not command:
            return

        command_aliases = {
            'start': 's',
            'resume': 's',
            'pause': 'p',
            'forward': 'f',
            'next': 'f',
            'back': 'b',
            'backward': 'b',
            'previous': 'b',
            'prev': 'b',
        }
        key = command_aliases.get(command, command[0])
        if key not in {'s', 'p', 'f', 'b'}:
            self.get_logger().warning(
                f'Ignoring unknown waypoint command "{message.data}". '
                'Use "s", "p", "f", or "b".'
            )
            return

        self.handle_keyboard_key(key)

    def handle_robot_position(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        feedback_source: str,
        feedback_frame_id: Optional[str],
    ) -> None:
        """Advance to the next waypoint when the robot is close enough."""
        if self.capture_origin_if_needed(
            robot_x,
            robot_y,
            robot_yaw,
            feedback_source,
            feedback_frame_id,
        ):
            return

        if self.completed or not self.started:
            return
        if not self.waypoints:
            return

        self.warn_once_for_frame_mismatch(feedback_source, feedback_frame_id)

        waypoint = self.waypoints[self.current_waypoint_index]
        distance = math.hypot(waypoint.x - robot_x, waypoint.y - robot_y)
        self.log_distance_status(
            waypoint,
            distance,
            robot_x,
            robot_y,
            feedback_source,
        )

        if self.paused:
            return

        if distance > self.distance_threshold:
            return

        self.get_logger().info(
            'Reached waypoint '
            f'{waypoint.index} at distance {distance:.3f} m'
        )
        self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(self.waypoints):
            self.completed = True
            self.get_logger().info('Survey waypoint sequence complete.')
            return

        self.publish_current_waypoint()

    def capture_origin_if_needed(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        feedback_source: str,
        feedback_frame_id: Optional[str],
    ) -> bool:
        """Generate the survey once the configured startup origin arrives."""
        if not self.use_current_pose_as_origin or self.origin_captured:
            return False
        if feedback_source != self.origin_capture_source:
            return False

        self.warn_once_for_frame_mismatch(feedback_source, feedback_frame_id)
        self.build_waypoints(robot_x, robot_y, robot_yaw)
        if self.started:
            self.publish_current_waypoint()
        return True

    def start_keyboard_listener(self) -> None:
        """Start a background listener for terminal start/pause commands."""
        if self.keyboard_thread is not None:
            return

        self.keyboard_input = self.open_keyboard_input()
        if self.keyboard_input is None:
            self.get_logger().warning(
                'Keyboard control requested, but no interactive terminal is '
                'available. Publish String commands on the configured command '
                'topic, or run the waypoint node from an interactive terminal.'
            )
            return

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            name='survey_waypoint_keyboard',
            daemon=True,
        )
        self.keyboard_thread.start()
        self.get_logger().info(
            'Reading keyboard control directly in this node: "s" start/resume, '
            '"p" pause, "f" skip forward, "b" skip backward.'
        )

    def open_keyboard_input(self):
        """Open the controlling terminal used for direct keyboard control."""
        if sys.stdin.isatty():
            return sys.stdin

        try:
            return open('/dev/tty', 'r', encoding='utf-8')
        except OSError:
            return None

    def keyboard_loop(self) -> None:
        """Read single-key terminal commands until the node shuts down."""
        stdin_file = self.keyboard_input
        if stdin_file is None:
            return

        old_settings = termios.tcgetattr(stdin_file)
        try:
            tty.setcbreak(stdin_file.fileno())
            while not self.keyboard_stop_event.is_set() and rclpy.ok():
                readable, _, _ = select.select([stdin_file], [], [], 0.1)
                if not readable:
                    continue

                self.handle_keyboard_key(stdin_file.read(1))
        finally:
            termios.tcsetattr(
                stdin_file,
                termios.TCSADRAIN,
                old_settings,
            )
            if stdin_file is not sys.stdin:
                stdin_file.close()

    def handle_keyboard_key(self, key: str) -> None:
        """Dispatch one terminal keyboard command."""
        key = key.lower()
        if key == 's':
            self.start_or_resume()
        elif key == 'p':
            self.pause()
        elif key == 'f':
            self.skip_forward()
        elif key == 'b':
            self.skip_backward()

    def start_or_resume(self) -> None:
        """Start or resume waypoint advancement from keyboard control."""
        if self.completed:
            self.get_logger().info(
                'Ignoring start command because the waypoint sequence is '
                'complete.'
            )
            return

        if not self.started:
            self.started = True
            self.paused = False
            self.last_distance_log_time = None
            if self.waypoints:
                self.get_logger().info(
                    'Waypoint planner started. Press "p" to pause, '
                    '"f" to skip forward, or "b" to skip backward.'
                )
            else:
                self.get_logger().info(
                    'Waypoint planner will start after the first robot pose '
                    'is received and the survey plan is generated.'
                )
            self.publish_current_waypoint()
            return

        if self.paused:
            self.paused = False
            self.last_distance_log_time = None
            self.get_logger().info(
                'Waypoint planner resumed. Press "p" to pause, '
                '"f" to skip forward, or "b" to skip backward.'
            )
            self.publish_current_waypoint()
            return

        self.get_logger().info('Waypoint planner is already running.')

    def pause(self) -> None:
        """Pause waypoint advancement while holding the current goal."""
        if self.completed:
            self.get_logger().info(
                'Ignoring pause command because the waypoint sequence is '
                'complete.'
            )
            return
        if not self.started:
            self.get_logger().info(
                'Waypoint planner is already waiting. Press "s" to start.'
            )
            return
        if self.paused:
            self.get_logger().info(
                'Waypoint planner is already paused. Press "s" to resume.'
            )
            return

        self.paused = True
        self.last_distance_log_time = None
        if not self.waypoints:
            self.get_logger().info(
                'Waypoint planner paused while waiting for the survey plan.'
            )
            return

        waypoint = self.waypoints[self.current_waypoint_index]
        self.get_logger().info(
            'Waypoint planner paused at waypoint '
            f'{waypoint.index}. Press "s" to resume.'
        )
        self.publish_current_waypoint()

    def skip_forward(self) -> None:
        """Move the active target to the next waypoint."""
        self.skip_waypoint(1, 'forward')

    def skip_backward(self) -> None:
        """Move the active target to the previous waypoint."""
        self.skip_waypoint(-1, 'backward')

    def skip_waypoint(self, delta: int, direction: str) -> None:
        """Move the active target by one waypoint from keyboard control."""
        if self.completed:
            self.get_logger().info(
                f'Ignoring skip {direction} command because the waypoint '
                'sequence is complete.'
            )
            return
        if not self.waypoints:
            self.get_logger().info(
                f'Cannot skip {direction} before the survey plan is '
                'generated.'
            )
            return

        next_index = self.current_waypoint_index + delta
        if next_index < 0:
            self.get_logger().info(
                'Cannot skip backward from the first waypoint.'
            )
            return
        if next_index >= len(self.waypoints):
            self.get_logger().info(
                'Cannot skip forward from the final waypoint.'
            )
            return

        self.current_waypoint_index = next_index
        self.last_logged_waypoint_index = None
        self.last_distance_log_time = None
        waypoint = self.waypoints[self.current_waypoint_index]
        self.get_logger().info(
            f'Skipped {direction} to waypoint {waypoint.index}.'
        )
        self.publish_current_waypoint(force=True)

    def stop_keyboard_listener(self) -> None:
        """Stop the terminal keyboard listener."""
        self.keyboard_stop_event.set()
        if self.keyboard_thread is not None:
            self.keyboard_thread.join(timeout=1.0)

    def warn_once_for_frame_mismatch(
        self,
        feedback_source: str,
        feedback_frame_id: Optional[str],
    ) -> None:
        """Warn when raw feedback coordinates are not in the waypoint frame."""
        if (
            not feedback_frame_id
            or not self.frame_id
            or feedback_frame_id == self.frame_id
            or feedback_frame_id in self.warned_feedback_frame_ids
        ):
            return

        self.warned_feedback_frame_ids.add(feedback_frame_id)
        self.get_logger().warning(
            f'{feedback_source} feedback frame is "{feedback_frame_id}", '
            f'but waypoint frame is "{self.frame_id}". Distances are '
            'computed directly from x/y, without a TF transform.'
        )

    def publish_marker_array(self) -> None:
        """Publish one RViz marker array containing path and active target."""
        if not self.waypoints:
            return

        message = MarkerArray()
        message.markers.append(self.create_path_marker())
        if 0 <= self.current_waypoint_index < len(self.waypoints):
            message.markers.append(self.create_current_waypoint_arrow())
        else:
            marker = self.base_marker(marker_id=1)
            marker.action = Marker.DELETE
            message.markers.append(marker)
        self.marker_publisher.publish(message)

    def create_path_marker(self) -> Marker:
        """Create a faded red line strip through all planned waypoints."""
        marker = self.base_marker(marker_id=0)
        marker.type = Marker.LINE_STRIP
        marker.scale.x = self.line_width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        marker.points = [
            Point(x=waypoint.x, y=waypoint.y, z=waypoint.z)
            for waypoint in self.waypoints
        ]
        return marker

    def base_marker(self, marker_id: int) -> Marker:
        """Create a valid RViz marker with the common header and namespace."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'survey_waypoints'
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def create_current_waypoint_arrow(self) -> Marker:
        """Create a saturated red arrow for the active waypoint."""
        waypoint = self.waypoints[self.current_waypoint_index]
        direction_x, direction_y = self.route_direction(
            self.current_waypoint_index,
        )
        tail = Point(
            x=waypoint.x,
            y=waypoint.y,
            z=waypoint.z + self.arrow_z_offset,
        )
        head = Point(
            x=waypoint.x + direction_x * self.arrow_length,
            y=waypoint.y + direction_y * self.arrow_length,
            z=waypoint.z + self.arrow_z_offset,
        )

        marker = self.base_marker(marker_id=1)
        marker.type = Marker.ARROW
        marker.scale.x = max(self.line_width * 1.6, 0.08)
        marker.scale.y = max(self.line_width * 4.5, 0.22)
        marker.scale.z = max(self.line_width * 4.0, 0.2)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [tail, head]
        return marker

    def log_distance_status(
        self,
        waypoint: SurveyWaypoint,
        distance: float,
        robot_x: float,
        robot_y: float,
        feedback_source: str,
    ) -> None:
        """Periodically log the active waypoint distance for debugging."""
        if self.distance_log_period <= 0.0:
            return

        now = self.get_clock().now()
        if self.last_distance_log_time is not None:
            elapsed = now - self.last_distance_log_time
            if elapsed.nanoseconds < self.distance_log_period * 1e9:
                return

        self.last_distance_log_time = now
        self.get_logger().info(
            'Waypoint status '
            f'{waypoint.index}: distance={distance:.3f} m, '
            f'threshold={self.distance_threshold:.3f} m, '
            f'robot=({robot_x:.3f}, {robot_y:.3f}), '
            f'target=({waypoint.x:.3f}, {waypoint.y:.3f}), '
            f'source={feedback_source}'
        )


def main(args=None) -> None:
    """Run the survey waypoint node."""
    rclpy.init(args=args)

    node = SurveyWaypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_keyboard_listener()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
