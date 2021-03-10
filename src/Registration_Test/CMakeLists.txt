cmake_minimum_required(VERSION 3.3)
project(Registration_Test)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.5 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(Registration_Test registration_test.cpp ../../include/cmr_lidarloop/lidar_registration.h)
target_link_libraries (Registration_Test ${PCL_LIBRARIES})
