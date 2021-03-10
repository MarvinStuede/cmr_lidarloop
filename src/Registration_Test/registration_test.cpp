/**
 * @file   registration_test.cpp
 * @author Tim-Lukas Habich
 * @date   06/2020
 *
 * @brief  Standalone to test the scan registration of cmr_lidarloop
 *         Point clouds of loop pairs are saved with lidar_registration_server.cpp if enabled
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include "../../include/cmr_lidarloop/lidar_registration.h"
#include <cstdlib>

using namespace std::literals::chrono_literals;
//#######INPUT#######
//Id of first cloud
int id=793;
//Id of second cloud
int loop_id=91;
//Path to clouds
std::string path_clouds="~/clouds";
//Axis pointing in sky direction
int sky_direction=-2;
//###################

int main (int argc, char** argv)
{

    if (argc < 5) {
        // Tell the user how to run the program
        std::cerr << "Usage: "<<argv[0]<<" <PATH_FOLDER> <ID_CLOUD_1> <ID_CLOUD_2> <SKY_DIRECTION>"<< std::endl;
        std::cerr << "Example: "<<argv[0]<<" ~/clouds 793 91 -2"<< std::endl;

        return 1;
    }
  path_clouds = argv[1];
  id = atoi(argv[2]);
  loop_id = atoi(argv[3]);
  sky_direction = atoi(argv[4]);

  // Loading first scan
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI> (path_clouds+"/"+std::to_string(id)+".pcd", *current_cloud) == -1)
  {
    PCL_ERROR ("Couldn't find first cloud.");
    return (-1);
  }

  std::cout << "Loaded " << current_cloud->size () << " data points from first cloud." << std::endl;

  /*//DEBUG
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_trimmed (new pcl::PointCloud<pcl::PointXYZI>);
  *current_cloud_trimmed=*current_cloud;
  trim_cloud_z(current_cloud_trimmed,1.0,-2);
  twoCloudsVis(current_cloud_trimmed,current_cloud_trimmed);
  //DEBUG*/

  // Loading second scan
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI> (path_clouds+"/"+std::to_string(loop_id)+".pcd", *target_cloud) == -1)
  {
    PCL_ERROR ("Couldn't find second cloud.");
    return (-1);
  }

  std::cout << "Loaded " << target_cloud->size () << " data points from second cloud." << std::endl;

  //move current_cloud in x-y-direction
  /*Eigen::Matrix4f move_x_y;
  move_x_y << 1,0,0,2,
              0,1,0,2,
              0,0,1,0,
              0,0,0,1;
  pcl::transformPointCloud (*current_cloud, *current_cloud, move_x_y);*/

  //Initial alignment
  std::clock_t start_reg;
  start_reg = std::clock();
  Eigen::Matrix4f transformation_ia;
  double fitness_score_ia;
  int n_inliers;
  //std::vector<float> radii_features={1.,1.25};
  initial_alignment(current_cloud,target_cloud,fitness_score_ia,transformation_ia,n_inliers,sky_direction,false,false);
  std::cout << "Transformation IA:\n" << transformation_ia
            << "\nFitness Score IA: " << fitness_score_ia
            << "\nNumber of inliers: " <<n_inliers<<std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud_ia (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud (*current_cloud, *output_cloud_ia, transformation_ia);
  double duration_ia = (( std::clock() - start_reg ) / (double) CLOCKS_PER_SEC);
  printf("Initial alignment took %.1lf seconds!\n",duration_ia);

  //ICP
  Eigen::Matrix4f transformation_icp;
  double fitness_score_icp;
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  icp_registration(current_cloud,target_cloud,output_cloud,transformation_ia,sky_direction,
                   transformation_icp,fitness_score_icp);
  std::cout <<"\nFinal Transformation ICP:\n" << transformation_icp
            <<"\nFitness score ICP: "<<fitness_score_icp<< std::endl;
  double duration_reg = (( std::clock() - start_reg ) / (double) CLOCKS_PER_SEC);
  printf("Full registration took %.1lf seconds!\n",duration_reg);

  //Visualizing cloud with intensity
  //visualize_cloud_with_intensity(current_cloud, "current cloud");
  //visualize_cloud_with_intensity(target_cloud, "target cloud");

  //Visualizing cloud pairs
  twoCloudsVis(current_cloud,target_cloud);
  twoCloudsVis(output_cloud_ia,target_cloud);
  twoCloudsVis(output_cloud,target_cloud);

  return (0);
}
