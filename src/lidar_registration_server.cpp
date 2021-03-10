/**
 * @file   lidar_registration_server.cpp
 * @author Tim-Lukas Habich
 * @date   04/2020
 *
 * @brief  Action server for point cloud registration
 */

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <cmr_lidarloop/LiDAR_RegistrationAction.h>
#include "cmr_lidarloop/lidar_registration.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

class LiDAR_RegistrationAction
{
protected:

  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<cmr_lidarloop::LiDAR_RegistrationAction> as_;
  std::string action_name_;
  cmr_lidarloop::LiDAR_RegistrationFeedback feedback_;
  cmr_lidarloop::LiDAR_RegistrationResult result_;

public:

  LiDAR_RegistrationAction(std::string name) :
    as_(nh_, name, boost::bind(&LiDAR_RegistrationAction::executeCB, this, _1), false),
    action_name_(name)
  {
    as_.start();
    ROS_INFO("LiDAR Registration ready.");
  }

  ~LiDAR_RegistrationAction(void)
  {
  }

  void executeCB(const cmr_lidarloop::LiDAR_RegistrationGoalConstPtr &goal)
  {
    std::clock_t start_reg;
    start_reg = std::clock();
    ROS_DEBUG("Computing Registration.");
    //Transform to PCL
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(goal->current_cloud, *current_cloud.get());
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(goal->target_cloud, *target_cloud.get());

    //Debugging: Save clouds as .pcd
    if (goal->path_clouds!="false"){
      std::ifstream path_test(goal->path_clouds);
      if (path_test)
      {
        pcl::io::savePCDFileASCII (goal->path_clouds+"/"+std::to_string(goal->id)+".pcd", *current_cloud);
        pcl::io::savePCDFileASCII (goal->path_clouds+"/"+std::to_string(goal->loop_id)+".pcd", *target_cloud);
      }
      else{
        ROS_WARN("Path for saving the point clouds is not a folder. Scans are not saved.");
      }
    }
    double duration_reg;

    //Initial alignment with global registration approach
    Eigen::Matrix4f transformation_ia;
    double fitness_score_ia;
    int n_inliers;
    int sky_direction = goal->sky_direction;
    initial_alignment(current_cloud,target_cloud,fitness_score_ia,transformation_ia,n_inliers,sky_direction);
    //std::cout << "Transformation IA:\n" << transformation_ia<<std::endl;

    //ICP
    Eigen::Matrix4f transformation_icp;
    double fitness_score_icp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    icp_registration(current_cloud,target_cloud,output_cloud,transformation_ia,sky_direction,transformation_icp,fitness_score_icp);
    //double duration_reg = (( std::clock() - start_reg ) / (double) CLOCKS_PER_SEC);
    //ROS_INFO("Full Registration took %.1lf seconds!",duration_reg);


    //Compute variance for information matrix (with RTAB-Maps approach)
    double variance;
    int n_correspondences;
    computeVariance(target_cloud,output_cloud,1,variance,n_correspondences);

    std::vector<float> transformation_vctr;
    for (int row = 0; row < 4; row++) {
      for (int column = 0; column < 4; column++) {
        transformation_vctr.push_back(transformation_icp(row,column));
      }
    }
    result_.transformation_vctr=transformation_vctr;
    result_.id=goal->id;
    result_.loop_id=goal->loop_id;
    result_.variance=variance;
    result_.n_inliers=n_inliers;
    // set the action state to succeeded
    as_.setSucceeded(result_);
    }

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "LiDAR_Registration");

  LiDAR_RegistrationAction LiDAR_Registration("LiDAR_Registration");
  ros::spin();

  return 0;
}
