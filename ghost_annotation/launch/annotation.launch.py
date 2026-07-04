import os
import sys

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory

# Get launch_utils share directory and add its src/ to sys.path
sys.path.append(os.path.join(get_package_share_directory('launch_utils'), 'src'))

from launch_utils.preprocess import preprocess_launch_json
from launch_utils.actions import get_util_actions, NodeAction
from launch_utils.common import try_load_json_from_args, parse_launch_args

PKG_PATH = get_package_share_directory('ghost_annotation')
DEFAULT_JSON_PATH = os.path.join(PKG_PATH, 'config', 'launch.json')


def get_annotator_actions(config, launch_args):
    actions = []

    bag_arg = None
    if 'bag' in launch_args:
        bag_arg = launch_args['bag']

    for prefix in ['pc', 'img']:
        key = f"{prefix}_annotator"
        node = f"{prefix}_annotator_node.py"
        if key in config:
            if bag_arg:
                config[key]['bag_path'] = bag_arg
            actions.append(
                NodeAction(config[key]).format_node(
                    package='ghost_annotation',
                    executable=node,
                    output='screen',
                    emulate_tty=True
                )
            )

    return actions


def launch(context, *args, **kwargs):
    actions = []

    launch_args = parse_launch_args(context.argv)
    json_data = try_load_json_from_args(launch_args, DEFAULT_JSON_PATH)
    config = preprocess_launch_json(json_data, launch_args)

    actions.extend(get_util_actions(config, launch_args))
    actions.extend(get_annotator_actions(config, launch_args))

    return actions


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch)
    ])
