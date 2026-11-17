'''
Launches the ghost_tester nodes, optionally with a bag playing into them.

    ros2 launch ghost_tester ghost_tester.launch.py \\
        bag:=<bag> model:=<artifact_xyz.model.txt> [<action>:=<preset> ...]

    bag      recording to play (the lidar topic and tf, see bag_play in config/launch.json)
    model    exported model for artifact_filter_node. Without it the node uses
             "model_file" from the config, then ./models/artifact_xyz.model.txt

Actions and their presets are in config/ (see launch_utils for the format):

    artifact_filter   multiscan (default) | multiscan_debug | null
    test_node         enabled | null (default)       the PCL shadow point test
    bag_play          lidar_live (default) | lidar_loop | all | ...
    filter            lance1 | lance2 | debug        groups: also start the
                      Foxglove bridge and the robot's transforms
    foxglove_bridge   all | null (default)
    robot_tf          lance | lance2 | null (default)
'''
import os
import sys

from launch import LaunchDescription
from launch.actions import OpaqueFunction, LogInfo
from ament_index_python.packages import get_package_share_directory

# Get launch_utils share directory and add its src/ to sys.path
sys.path.append(os.path.join(get_package_share_directory('launch_utils'), 'src'))

from launch_utils.preprocess import preprocess_launch_json
from launch_utils.actions import get_util_actions, NodeAction
from launch_utils.common import try_load_json_from_args, parse_launch_args

PKG_PATH = get_package_share_directory('ghost_tester')
DEFAULT_JSON_PATH = os.path.join(PKG_PATH, 'config', 'launch.json')
DEFAULT_MODEL = os.path.join('models', 'artifact_xyz.model.txt')


def get_node_actions(config, launch_args):
    actions = []

    if 'artifact_filter' in config:
        block = config['artifact_filter']
        model = launch_args.get('model') or block.get('model_file') or DEFAULT_MODEL
        block['model_file'] = os.path.abspath(os.path.expanduser(model))
        if not os.path.isfile(block['model_file']):
            actions.append(LogInfo(msg=(
                f"[ghost_tester] model file {block['model_file']} does not exist: "
                "pass model:=<path to artifact_xyz.model.txt> (train a realtime "
                "model in the annotator app to get one). artifact_filter not started.")))
        else:
            actions.append(
                NodeAction(block).format_node(
                    package='ghost_tester',
                    executable='artifact_filter_node',
                    output='screen'
                )
            )

    if 'test_node' in config:
        actions.append(
            NodeAction(config['test_node']).format_node(
                package='ghost_tester',
                executable='test_node',
                output='screen'
            )
        )

    return actions


def launch(context, *args, **kwargs):
    actions = []

    launch_args = parse_launch_args(context.argv)
    json_data = try_load_json_from_args(launch_args, DEFAULT_JSON_PATH)
    config = preprocess_launch_json(json_data, launch_args)

    actions.extend(get_util_actions(config, launch_args))
    actions.extend(get_node_actions(config, launch_args))

    return actions


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch)
    ])
