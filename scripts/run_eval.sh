#!/bin/bash
with_ll="true"
bag_folder_path="$HOME/Messdaten/kitti_odom_dataset/data/"

#declare -a bags=("kitti_odometry_sequence_00.bag.out" "kitti_odometry_sequence_02.bag.out" "kitti_odometry_sequence_05.bag.out" "kitti_odometry_sequence_06.bag.out" "kitti_odometry_sequence_07.bag.out" "kitti_odometry_sequence_08.bag.out" "kitti_odometry_sequence_09.bag.out")
declare -a bags=("kitti_odometry_sequence_06.bag.out" "kitti_odometry_sequence_09.bag.out")


  echo "Following bags w1ill be played from folder $bag_folder_path"
for bag in "${bags[@]}"; do
  printf "\t -$bag \n"
done
tmux new -d -s rtabmap
tmux has-session -t rtabmap
if [ $? == 0 ]
then
 echo "RTAB Map Session started"

else
 echo "RTAB Map Session not started, quitting..."
 exit 1
fi

for bag in "${bags[@]}"; do
  db_path="$HOME/.ros/bag_$bag.db"
  bag_path="$bag_folder_path$bag"
  echo "Processing $bag_path"
  rosbag_cmd="rosbag play --clock -r 0.1 $bag_path"

  tmux send -t rtabmap "roslaunch cmr_lidarloop kitti_odom.launch database_path:=$db_path with_lidarloop:=$with_ll" ENTER
  sleep 5
 # ping_resp=$(rosnode ping /rtabmap/rtabmap -c 1)
 # if [[ "$ping_resp" == *"time"* ]]; then
 #    echo "Started RTABMap"
 # else
 #    echo "RTAB Map not started, quitting..."
 #    exit 1
 # fi

  tmux new -d -s rosbag "$rosbag_cmd; tmux wait -S rosbag"
  echo "DB path: $db_path"
  echo "Started rosbag"
  echo "Waiting for rosbag to finish...."

  tmux wait rosbag
  sleep 5
  echo "Saving map $db_path"
  rosservice call /rtabmap/backup "{}"
  sleep 60
  echo "Quitting RTAB Map"
  tmux send-keys -t rtabmap C-C
  if [ "$with_ll" = "true" ] ; then
      db_path_mv="${bag_path}_ll.db"
  else
      db_path_mv="${bag_path}.db"
  fi

  echo "Moving db to $db_path_mv"
  mv "$db_path.back" "$db_path_mv"
  sleep 10
done
  tmux kill-session -t rtabmap
  echo done
