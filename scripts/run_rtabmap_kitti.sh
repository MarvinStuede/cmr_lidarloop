#!/bin/bash
# Modified
rtabmap-kitti_dataset \
      --Optimizer/GravitySigma 0.3 \
      --Rtabmap/DetectionRate 2.0 \
      --RGBD/OptimizeMaxError 1.0 \
      --Odom/ScanKeyFrameThr 0.8 \
      --OdomF2M/ScanMaxSize 10000 \
      --OdomF2M/ScanSubtractRadius 0.5 \
      --GFTT/MinDistance 7 \
      --GFTT/QualityLevel 0.01 \
      --Vis/MaxFeatures 1500 \
      --Kp/MaxFeatures 750 \
      --Mem/STMSize 30 \
      --Mem/RehearsalSimilarity 0.2 \
      --OdomF2M/MaxSize 3000 \
      --Rtabmap/PublishRAMUsage true \
      --Rtabmap/CreateIntermediateNodes true \ 
      --gt "/home/stuede/Messdaten/kitti_odom_dataset/data/poses/00.txt" \     
      "/home/stuede/Messdaten/kitti_odom_dataset/data/sequences/00"

# Original
rtabmap-kitti_dataset \
       --Rtabmap/PublishRAMUsage true\
       --Rtabmap/DetectionRate 2\
       --Rtabmap/CreateIntermediateNodes true\
       --RGBD/LinearUpdate 0\
       --GFTT/QualityLevel 0.01\
       --GFTT/MinDistance 7\
       --OdomF2M/MaxSize 3000\
       --Mem/STMSize 30\
       --Kp/MaxFeatures 750\
       --Vis/MaxFeatures 1500\
       --gt "~/Messdaten/kitti_odom_dataset/data/poses/00.txt"\
       ~/Messdaten/kitti_odom_dataset/data/sequences/00
