/**
 * @file   lidar_registration.h
 * @author Tim-Lukas Habich
 * @date   04/2020
 *
 * @brief  Header file for registration of point clouds
 *         using PCL (pcl::SampleConsensusInitialAlignment)
 *         With: http://pointclouds.org/documentation/tutorials/template_alignment.php
 */

#ifndef LIDAR_REGISTRATION_H
#define LIDAR_REGISTRATION_H
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <thread>
#include <pcl/features/shot.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/features/multiscale_feature_persistence.h>

using namespace std::literals::chrono_literals;
using namespace pcl;

void computeNormals(PointCloud<PointXYZI>::Ptr cloud,
                    PointCloud<Normal>::Ptr normals,
                    int k)
{
  NormalEstimation<PointXYZI,Normal> norm_est;
  norm_est.setInputCloud (cloud);
  search::KdTree<PointXYZI>::Ptr SearchMethod (new search::KdTree<PointXYZI>);
  norm_est.setSearchMethod (SearchMethod);
  //norm_est.setRadiusSearch (radius);
  norm_est.setKSearch(k);
  norm_est.compute (*normals);
}

void computeKeypointsAndDescriptors(PointCloud<PointXYZI>::Ptr cloud,
                                    PointCloud<Normal>::Ptr normals,
                                    PointCloud<FPFHSignature33>::Ptr features,
                                    PointCloud<PointXYZI>::Ptr keypoints,
                                    std::vector<float> scale_values,
                                    float alpha)
{
  //Keypoint indices and features
  FPFHEstimation<PointXYZI, Normal, FPFHSignature33>::Ptr fpfh_est (new FPFHEstimation<PointXYZI, Normal, FPFHSignature33>);
  fpfh_est->setSearchSurface (cloud);
  fpfh_est->setInputCloud (cloud);
  fpfh_est->setInputNormals (normals);
  search::KdTree<PointXYZI>::Ptr tree (new search::KdTree<PointXYZI> ());
  fpfh_est->setSearchMethod (tree);

  MultiscaleFeaturePersistence<PointXYZI, FPFHSignature33> fpfh_per;
  fpfh_per.setScalesVector (scale_values);
  fpfh_per.setAlpha (alpha);
  fpfh_per.setFeatureEstimator (fpfh_est);
  fpfh_per.setDistanceMetric (pcl::CS);
  boost::shared_ptr<std::vector<int> > keypoint_indices (new std::vector<int> ());
  fpfh_per.determinePersistentFeatures(*features, keypoint_indices);

  //Extract keypoints from cloud
  ExtractIndices<PointXYZI> extract_indices_filter;
  extract_indices_filter.setInputCloud (cloud);
  extract_indices_filter.setIndices (keypoint_indices);
  extract_indices_filter.filter (*keypoints);
}

void normalsVis (
    PointCloud<PointXYZI>::Ptr cloud, PointCloud<Normal>::Ptr normals)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        color (cloud, 0, 0, 255);
  viewer->addPointCloud<pcl::PointXYZI> (cloud, color, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZI, pcl::Normal> (cloud, normals, 2, 0.1, "normals");
  //Viewer
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  // Wait until visualizer window is closed.
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }

}

void correspondencesVis(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints1,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints2,
                        pcl::CorrespondencesPtr corr){
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    //Clouds
    viewer->addPointCloud<pcl::PointXYZI> (cloud1, "Cloud 1");
    viewer->addPointCloud<pcl::PointXYZI> (cloud2, "Cloud 2");
    //Keypoints of clouds
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>blue1 (keypoints1, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZI> (keypoints1,blue1, "Keypoints 1");
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "Keypoints 1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>green2 (keypoints2, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZI> (keypoints2,green2, "Keypoints 2");
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "Keypoints 2");
    //Correspondences between keypoints
    viewer->addCorrespondences<pcl::PointXYZI> (keypoints1,keypoints2,*corr,"Correspondences");
    //Viewer
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    // Wait until visualizer window is closed.
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      std::this_thread::sleep_for(100ms);
    }
}

void visualize_cloud(pcl::visualization::PCLVisualizer::Ptr viewer,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                     int R, int G, int B, std::string name)
{
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color (cloud, R, G, B);
    viewer->addPointCloud<pcl::PointXYZI> (cloud, color, name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2, name);
}

void twoCloudsVis(PointCloud<PointXYZI>::Ptr cloud1,
                  PointCloud<PointXYZI>::Ptr cloud2){
  // Initializing point cloud visualizer
  visualization::PCLVisualizer::Ptr viewer (new visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (255,255,255);
  visualize_cloud(viewer,cloud1,0,80,155,"cloud1");
  visualize_cloud(viewer,cloud2,238,110,33,"cloud2");
  // Starting visualizer
  viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  //viewer->setCameraPosition(-1.14212, 5.56635, 7.29499, 0.258794, 0.965176, -0.0382295);
  // Wait until visualizer window is closed.
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
}

void transformation_2D(Eigen::Matrix4f &transformation,
                       int& sky_direction){
    // Use knowledge regarding sky_direction
    // translation=0 & pure rotation around axis
    if(abs(sky_direction)==1){
        //x
        transformation(0,3)=0;
        transformation(0,2)=0;
        transformation(0,1)=0;
        transformation(1,0)=0;
        transformation(2,0)=0;
        transformation(0,0)=1;

    }
    else if (abs(sky_direction)==2) {
        //y
        transformation(1,3)=0;
        transformation(0,1)=0;
        transformation(2,1)=0;
        transformation(1,0)=0;
        transformation(1,2)=0;
        transformation(1,1)=1;
    }
    else if (abs(sky_direction)==3){
        //z
        transformation(2,3)=0;
        transformation(2,0)=0;
        transformation(2,1)=0;
        transformation(0,2)=0;
        transformation(1,2)=0;
        transformation(2,2)=1;
    }

}

void visualize_cloud_with_intensity(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                    std::string name)
{
  // Initializing point cloud visualizer
  visualization::PCLVisualizer::Ptr viewer (new visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0,0,0);
  //Visualize intensity
  visualization::PointCloudColorHandlerGenericField<PointXYZI> intensity_distribution (cloud, "intensity");
  viewer->addPointCloud<PointXYZI> (cloud, intensity_distribution, name);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1, name);
  // Starting visualizer
  viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  //viewer->setCameraPosition(-2.8524, 3.75627, 17.1469,  0.26031, -0.939001, 0.224757);
  // Wait until visualizer window is closed.
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
}

void initial_alignment(PointCloud<PointXYZI>::Ptr input_cloud,
                       PointCloud<PointXYZI>::Ptr target_cloud,
                       double &fitness_score,
                       Eigen::Matrix4f &transformation_ia,
                       int &n_inliers,
                       int sky_direction,
                       bool visualize_normals=false,
                       bool visualize_corr=false,
                       int k_normals=100,
                       std::vector<float> radii_features={1.,1.25},
                       float alpha=.6,
                       float inlier_threshold=2.,
                       int nr_iterations =300)
{
    std::clock_t start_ia;
    start_ia = std::clock();

    //Input cloud: normals, keypoints & descriptors
    PointCloud<Normal>::Ptr input_cloud_normals (new PointCloud<Normal>);
    computeNormals(input_cloud,input_cloud_normals,k_normals);
    PointCloud<FPFHSignature33>::Ptr input_cloud_features (new PointCloud<FPFHSignature33>);
    PointCloud<PointXYZI>::Ptr input_cloud_keypoints (new PointCloud<PointXYZI>);
    computeKeypointsAndDescriptors(input_cloud,input_cloud_normals,input_cloud_features,
                                   input_cloud_keypoints,radii_features,alpha);

    //Target cloud: normals, keypoints & descriptors
    PointCloud<Normal>::Ptr target_cloud_normals (new PointCloud<Normal>);
    computeNormals(target_cloud,target_cloud_normals,k_normals);
    PointCloud<FPFHSignature33>::Ptr target_cloud_features (new PointCloud<FPFHSignature33>);
    PointCloud<PointXYZI>::Ptr target_cloud_keypoints (new PointCloud<PointXYZI>);
    computeKeypointsAndDescriptors(target_cloud,target_cloud_normals,target_cloud_features,
                                   target_cloud_keypoints,radii_features,alpha);

    double duration_descriptors = ((std::clock()-start_ia)/(double)CLOCKS_PER_SEC)*pow(10,3);
    //std::cout<<"Computation of normals, keypoints and descriptors took "<<duration_descriptors<<"ms."<<std::endl;

    //Visualize normals for debugging
    if (visualize_normals){
      normalsVis(input_cloud, input_cloud_normals);
      normalsVis(target_cloud, target_cloud_normals);
    }

    //Compute correspondences
    CorrespondencesPtr correspondences (new Correspondences);
    registration::CorrespondenceEstimation<FPFHSignature33, FPFHSignature33> cest;
    cest.setInputSource (input_cloud_features);
    cest.setInputTarget (target_cloud_features);
    cest.determineCorrespondences (*correspondences);

    //Reject false correspondences
    CorrespondencesPtr correspondences_filtered (new Correspondences);
    registration::CorrespondenceRejectorSampleConsensus<PointXYZI> rejector;
    rejector.setInputSource(input_cloud_keypoints);
    rejector.setInputTarget (target_cloud_keypoints);
    rejector.setInlierThreshold (inlier_threshold);
    rejector.setMaximumIterations (nr_iterations);
    rejector.setRefineModel (false);
    rejector.setInputCorrespondences (correspondences);
    rejector.getCorrespondences (*correspondences_filtered);
    //Save number of inliers after outlier rejection
    n_inliers=correspondences_filtered->size();

    //Visualize correspondences before and after rejection
    if (visualize_corr){
      correspondencesVis(input_cloud,input_cloud_keypoints,target_cloud,target_cloud_keypoints,correspondences);
      correspondencesVis(input_cloud,input_cloud_keypoints,target_cloud,target_cloud_keypoints,correspondences_filtered);
    }
    start_ia = std::clock();
    //3D Alignment
    registration::TransformationEstimationSVD<PointXYZI, PointXYZI> trans_est;
    trans_est.estimateRigidTransformation (*input_cloud_keypoints,*target_cloud_keypoints,
                                           *correspondences_filtered, transformation_ia);
    //2D SLAM
    transformation_2D(transformation_ia,sky_direction);

    //2D Alignment
    /*registration::TransformationEstimation2D <PointXYZI, PointXYZI> trans_est_2D;
    trans_est_2D.estimateRigidTransformation(*input_cloud_keypoints,*target_cloud_keypoints,
                                             *correspondences_filtered, transformation_ia);*/

    //Transformation validation
    registration::TransformationValidationEuclidean<PointXYZI,PointXYZI,float> trans_val;
    fitness_score=trans_val.validateTransformation(input_cloud,target_cloud,transformation_ia);

    double duration_ia = ((std::clock()-start_ia)/(double)CLOCKS_PER_SEC)*pow(10,3);
    //std::cout<<"3D Alignment took "<<duration_ia<<"ms."<<std::endl;
}

void icp_registration(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_ptr,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud_ptr,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud_ptr,
                      Eigen::Matrix4f initial_guess,
                      int sky_direction,
                      Eigen::Matrix4f &transformation_icp,
                      double &fitness_score,
                      float t_epsilon=1e-6,
                      float max_correspondence=1,
                      int max_iterations=50,
                      float outlier_threshold=0.05)
{
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setTransformationEpsilon(t_epsilon);
  //icp.setEuclideanFitnessEpsilon(e_epsilon);
  icp.setMaximumIterations(max_iterations);
  icp.setRANSACOutlierRejectionThreshold(outlier_threshold);
  icp.setMaxCorrespondenceDistance(max_correspondence);
  icp.setInputSource (input_cloud_ptr);
  icp.setInputTarget (target_cloud_ptr);
  icp.align(*output_cloud_ptr,initial_guess);
  transformation_icp=icp.getFinalTransformation();
  // Use knowledge regarding z axis
  transformation_2D(transformation_icp,sky_direction);
  fitness_score=icp.getFitnessScore();
  //std::cout << "Final Transformation ICP:\n" << transformation_icp<< std::endl;
}

void voxel_grid_filter_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
                             float leafsize)
{
  pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (leafsize, leafsize, leafsize);
  approximate_voxel_filter.setInputCloud (cloud_ptr);
  approximate_voxel_filter.filter (*cloud_ptr);
  //ROS_DEBUG("Filtered cloud contains %d points.",cloud_ptr->size());
}

void intensity_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
                      float i_limit)
{
  /*//Get min and max intensity in cloud
  double i_min,i_max;
  for(int i = 0; i<cloud_ptr->size(); i++){
    double temp_i=cloud_ptr->points[i].intensity;
    if(i==0){
      i_min=temp_i;
      i_max=temp_i;
    }
    else{
      if(temp_i<i_min){i_min=temp_i;}
      if(temp_i>i_max){i_max=temp_i;}
    }
  }
  double i_limit=i_min+(i_max-i_min)*i_percentage;*/

  pcl::PointIndices::Ptr inliers_cloud(new pcl::PointIndices());
  pcl::ExtractIndices<pcl::PointXYZI> extract_cloud;
  for(int i = 0; i<cloud_ptr->size(); i++){
    if(cloud_ptr->points[i].intensity <i_limit){
         inliers_cloud->indices.push_back(i);}}
  extract_cloud.setInputCloud(cloud_ptr);
  extract_cloud.setIndices(inliers_cloud);
  extract_cloud.setNegative(true);
  extract_cloud.filter(*cloud_ptr);
}

void trim_cloud_z(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
                  double z_limit, int sky_direction)
{
    if(sky_direction==0){
        //z-trimming disabled
        return;
    }

    pcl::PointIndices::Ptr inliers_cloud(new pcl::PointIndices());
    pcl::ExtractIndices<pcl::PointXYZI> extract_cloud;

    if(abs(sky_direction)==1){
        //x-axis
        for(int i = 0; i<cloud_ptr->size(); i++){
            if(cloud_ptr->points[i].x<z_limit){inliers_cloud->indices.push_back(i);}
        }
    }
    else if(abs(sky_direction)==2){
        //y-axis
        for(int i = 0; i<cloud_ptr->size(); i++){
            if(cloud_ptr->points[i].y<z_limit){inliers_cloud->indices.push_back(i);}
        }
    }
    else if(abs(sky_direction)==3){
        //z-axis
        for(int i = 0; i<cloud_ptr->size(); i++){
            if(cloud_ptr->points[i].z<z_limit){inliers_cloud->indices.push_back(i);}
        }
    }

    extract_cloud.setInputCloud(cloud_ptr);
    extract_cloud.setIndices(inliers_cloud);
    if(sky_direction>0){
        extract_cloud.setNegative(true);
    }
    else {
        extract_cloud.setNegative(false);
    }
    extract_cloud.filter(*cloud_ptr);
}

void random_downsampling(PointCloud<PointXYZI>::Ptr cloud,
                         int max_points)
{
  if (cloud->size()>max_points){
    //Downsample cloud randomly until max_points
    std::vector<int> indices_cloud;
    for (int i = 0; i < cloud->size(); ++i) {
      indices_cloud.push_back(i);
    }
    int n_del=cloud->size()-max_points;
    //Shuffle vector randomly
    std::random_shuffle(indices_cloud.begin(), indices_cloud.end());
    boost::shared_ptr<std::vector<int> > indices_delete (new std::vector<int> (indices_cloud.begin(),indices_cloud.begin()+n_del));
    //Delete points
    ExtractIndices<PointXYZI> indices_filter;
    indices_filter.setInputCloud (cloud);
    indices_filter.setIndices (indices_delete);
    indices_filter.setNegative(true);
    indices_filter.filter (*cloud);
  }
}

void trim_cloud_r(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
                  double r_limit)
{
  pcl::PointIndices::Ptr inliers_cloud(new pcl::PointIndices());
  pcl::ExtractIndices<pcl::PointXYZI> extract_cloud;

  for(int i = 0; i<cloud_ptr->size(); i++){
    double temp_r=sqrt(cloud_ptr->points[i].x*cloud_ptr->points[i].x+
                       cloud_ptr->points[i].y*cloud_ptr->points[i].y+
                       cloud_ptr->points[i].z*cloud_ptr->points[i].z);
    if(temp_r>r_limit){
         inliers_cloud->indices.push_back(i);}}
  extract_cloud.setInputCloud(cloud_ptr);
  extract_cloud.setIndices(inliers_cloud);
  extract_cloud.setNegative(true);
  extract_cloud.filter(*cloud_ptr);
}

//from rtabmap (util3d_registration.cpp)
void computeVariance(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr & cloudA,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr & cloudB,
    double maxCorrespondenceDistance,
    double & variance,
    int & correspondencesOut)
{
  variance = 1;
  correspondencesOut = 0;
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZI, pcl::PointXYZI>::Ptr est;
  est.reset(new pcl::registration::CorrespondenceEstimation<pcl::PointXYZI, pcl::PointXYZI>);
  est->setInputTarget(cloudA->size()>cloudB->size()?cloudA:cloudB);
  est->setInputSource(cloudA->size()>cloudB->size()?cloudB:cloudA);
  pcl::Correspondences correspondences;
  est->determineCorrespondences(correspondences, maxCorrespondenceDistance);

  if(correspondences.size()>=3)
  {
    std::vector<double> distances(correspondences.size());
    for(unsigned int i=0; i<correspondences.size(); ++i)
    {
      distances[i] = correspondences[i].distance;
    }

    //variance
    std::sort(distances.begin (), distances.end ());
    double median_error_sqr = distances[distances.size () >> 1];
    variance = (2.1981 * median_error_sqr);
  }

  correspondencesOut = (int)correspondences.size();
}
#endif // LIDAR_REGISTRATION_H
