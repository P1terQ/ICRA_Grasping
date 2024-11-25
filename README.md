## Installation
- [MoveIt](https://github.com/ros-planning/panda_moveit_config)
- [TRAC-IK](http://wiki.ros.org/trac_ik)
- [realsense_ros](https://github.com/IntelRealSense/realsense-ros)

- Ros Packages Compilation
```
catkin build franka_ros
catkin build franka_control_wrappers panda_control robot_helpers
catkin build vgn
catkin build active_grasp
```

- Python dependencies can be installed with
```
pip install -r requirements.txt
```

If you meet any problems during the installation, please refer to [VGN](https://github.com/ethz-asl/vgn) and [active_grasp](https://github.com/ethz-asl/active_grasp/tree/devel).


## Hardware Dependencies
franka_ros version: _0.10.1_

libfranka_version: _0.8.0_

## Run Simulation
```
roscore
roslaunch active_grasp env.launch sim:=true
python3 scripts/run.py nbv
```

## Run Hardware
```
roscore
roslaunch active_grasp hw_peter.launch
roslaunch active_grasp env.launch sim:=false
python3 scripts/run.py nbv
```

## Weights
https://drive.google.com/file/d/1Ue4u3yJLHssPE4JUX3eVu5XEOb8WzmJC/view?usp=drive_link

## Competition Vedio  
https://www.youtube.com/watch?v=1wqAk8SbL2Q
