import os
import sys

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory

# Get launch_utils share directory and add its src/ to sys.path
LAUNCH_UTILS_SHARE = get_package_share_directory('launch_utils')
sys.path.append(os.path.join(LAUNCH_UTILS_SHARE, 'src'))

from launch_utils.preprocess import preprocess_launch_json
from launch_utils.actions import get_util_actions, NodeAction, extract_util_configs
from launch_utils.common import try_load_json_from_args, parse_launch_args


def launch(context, *args, **kwargs):
    # Parse CLI launch arguments (e.g. key:=value)
    launch_args = parse_launch_args(context.argv)
    
    # Get ghost_annotation share directory
    pkg_share = get_package_share_directory('ghost_annotation')
    
    # Load JSON config, resolving imports
    json_data = try_load_json_from_args(
        launch_args, os.path.join(pkg_share, 'config', 'launch.json'))
        
    # Preprocess configuration using overrides (e.g. setting presets for actions)
    pp_config = preprocess_launch_json(json_data, launch_args)
    
    # If the common 'bag' CLI arg is provided, map it to the 'bag_path' param of annotator
    if 'annotator' in pp_config and 'bag' in launch_args:
        pp_config['annotator']['bag_path'] = launch_args['bag']
        
    # Get the standard utility actions (foxglove_bridge, joy_node, etc.)
    actions = get_util_actions(pp_config, launch_args)
    
    # Extract utility configs so only custom configs remain in pp_config
    extract_util_configs(pp_config)
    
    # If annotator is enabled, format it as a ROS2 Node and append to actions
    if 'annotator' in pp_config:
        annotator_action = NodeAction(pp_config['annotator'])
        actions.append(annotator_action.format_node(
            package='ghost_annotation',
            executable='annotator_node.py',
            output='screen',
            emulate_tty=True
        ))
        
    return actions


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch)
    ])
