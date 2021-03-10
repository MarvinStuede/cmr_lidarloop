cmake_minimum_required(VERSION 2.8.3)
project(cmr_lidarloop)

add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy std_msgs sensor_msgs geometry_msgs cv_bridge image_transport rtabmap_ros
  message_generation actionlib_msgs actionlib
)
find_package(Boost REQUIRED COMPONENTS system)
find_package(RTABMap REQUIRED)
find_package(PCL 1.2 REQUIRED)
find_package(Eigen3 REQUIRED)

 add_service_files(
   FILES
   LiDAR_Loopdetector.srv
 )

 add_action_files(
   FILES
   LiDAR_Registration.action
 )

 generate_messages(
   DEPENDENCIES
   roscpp std_msgs sensor_msgs geometry_msgs actionlib_msgs
 )

catkin_package(
CATKIN_DEPENDS actionlib_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${RTABMap_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(lidar_loopdetection src/lidar_loopdetection.cpp include/cmr_lidarloop/lidar_loopdetection.h include/cmr_lidarloop/lidar_registration.h)
target_link_libraries(lidar_loopdetection ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_dependencies(lidar_loopdetection cmr_lidarloop_gencpp)
add_dependencies(lidar_loopdetection ${cmr_lidarloop_EXPORTED_TARGETS})

add_executable(save_data_for_detector src/save_data_for_detector.cpp)
target_link_libraries(save_data_for_detector ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(save_data_for_detector cmr_lidarloop_gencpp)
add_dependencies(save_data_for_detector ${cmr_lidarloop_EXPORTED_TARGETS})

add_executable(lidar_registration_server
    src/lidar_registration_server.cpp
    include/cmr_lidarloop/lidar_registration.h)

target_link_libraries(lidar_registration_server
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_dependencies(
  lidar_registration_server
  ${cmr_lidarloop_EXPORTED_TARGETS}
)

