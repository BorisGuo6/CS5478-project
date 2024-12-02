# CS5478-project

## STEP1 install ROS1 Noetic

```
wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh && chmod +x ./ros_install_noetic.sh && ./ros_install_noetic.sh
```

choose 1 to install noetic desktop edition.

## STEP2 Install dependences

```
bash catkin_ws/src/waterplus_map_tools/scripts/install_for_noetic.sh
bash catkin_ws/src/wpb_home/wpb_home_bringup/scripts/install_for_noetic.sh
bash catkin_ws/src/wpr_simulation/scripts/install_for_noetic.sh
pip install empy catkin_pkg
```
## Usage

```
python3 dce_nn_navigation.py --train_dir=$(pwd)/selected_network --experiment=selected_network --env=test --obs_key="observations" --load_checkpoint_kind=best
```
