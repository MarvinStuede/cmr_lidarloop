/**
 * @file   lidar_loopdetection.h
 * @author Tim-Lukas Habich
 * @date   02/2020
 *
 * @brief  Header file for loop detection
 */
#ifndef LIDAR_LOOPDETECTION_H
#define LIDAR_LOOPDETECTION_H
#include "ros/ros.h"
#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <ros/package.h>

#include "std_msgs/String.h"
#include "std_srvs/Empty.h"
#include <math.h>

#include <rtabmap_ros/MapData.h>
#include <rtabmap_ros/Info.h>
#include <rtabmap_ros/MsgConversion.h>
#include <rtabmap_ros/AddLink.h>

#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/RegistrationVis.h>
#include <rtabmap/utilite/UStl.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>

class lidar_loopdetector
{
public:
  lidar_loopdetector(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int detector_strategy, double rmax);
  void get_feature_vector(std::vector<double>&feature_vector, std::vector<int>&lengths);
  void get_feature_vector(std::vector<double>&feature_vector);
private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  int strategy_;
  void features_classification(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  std::vector<double> feature_vector;
  //parameters (see Granström)
  double r_max; //maximum range in m
  double g_dist=2.5;
  double g_r1;
  double g_r2;
  double g_r3;
  double bin_sizes[9] = {0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3}; //Bin sizes in metres for the range histograms
};

lidar_loopdetector::lidar_loopdetector(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int detector_strategy, double rmax):
cloud_(cloud),
strategy_(detector_strategy),
r_max(rmax)
{
  if(strategy_==1) //Detector according to the method of Granström - Learning to Close the Loop from 3D Point Clouds
  {
    g_r1 = r_max;
    g_r2 = 0.75*r_max;
    g_r3 = 0.5*r_max;
    features_classification(cloud_);
  }
}

void lidar_loopdetector::features_classification(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  auto start_hr = std::chrono::high_resolution_clock::now();
  //incremental calculation of mean and standard deviation with: http://datagenetics.com/blog/november22017/index.html

  //initialization
  double duration;
  double xyz_i[3],xyz_i_old[3],xyz_last[3],xyz_next[3]; //xyz coordinates
  double r_i, r_i_n, r_i_old, r_last, r_next, r_mean=0, r_mean_tilde=0, r_i_r, r_i_d; //ranges
  double features1[32]={};//features of type 1 (pointcloud ->real number)
  int n_tilde = 0;//number of points with r_i<r_max
  int n=cloud->size();//number of points
  int delta_n=0,curv_n=0,relative_n=0;
  int gr1_n=0, gr2_n=0, gr3_n=0;
  int bin;
  double Sn_tilde=0,mu_old_tilde,Sn=0,mu_old,Sn_centroid=0,mu_centroid_old,Sn_delta=0,delta_mu_old,Sn_curv=0,curv_mu_old;
  double Sn_relative=0,mu_old_relative,Sn_relative_tilde=0,mu_old_relative_tilde;
  double Sn_gr1=0,mu_old_gr1,Sn_gr2=0,mu_old_gr2,Sn_gr3=0,mu_old_gr3;
  double mean[3]={}, mean_tilde[3]={};
  Eigen::Matrix3d A_sphere,ATA;
  Eigen::Vector3d B_sphere,c_sphere;
  double A[3][3]={},B[3]={};
  double sum_R_sphere=0,R_sphere;
  double delta_sphere[3],delta_centroid[3],abs_delta_sphere,distance_centroid;
  double delta_i_old, delta_mu=0,delta_i_last,delta_i_next,delta_last_next;
  double s_tri,area_tri,curv_i;
  double kurtosis_2=0,kurtosis_4=0,kurtosis_2_tilde=0,kurtosis_4_tilde=0;

  //range histogram initialization with zeros
  int histogram_1[int(ceil(r_max/bin_sizes[0]))]={};
  int histogram_2[int(ceil(r_max/bin_sizes[1]))]={};
  int histogram_3[int(ceil(r_max/bin_sizes[2]))]={};
  int histogram_4[int(ceil(r_max/bin_sizes[3]))]={};
  int histogram_5[int(ceil(r_max/bin_sizes[4]))]={};
  int histogram_6[int(ceil(r_max/bin_sizes[5]))]={};
  int histogram_7[int(ceil(r_max/bin_sizes[6]))]={};
  int histogram_8[int(ceil(r_max/bin_sizes[7]))]={};
  int histogram_9[int(ceil(r_max/bin_sizes[8]))]={};

  //length of feature vector
  int n_feature_vector=sizeof(features1)/sizeof(features1[0]);
  for (int i=0;i<sizeof(bin_sizes)/sizeof(bin_sizes[0]);i++)
  {
    n_feature_vector+=int(ceil(r_max/bin_sizes[i]));
  }

  // ######## Loop 1 through point cloud ########
  for(int i = 0; i<n; i++)
  {
    xyz_i[0]=cloud->points[i].x;
    xyz_i[1]=cloud->points[i].y;
    xyz_i[2]=cloud->points[i].z;

    r_i=sqrt(xyz_i[0]*xyz_i[0]+xyz_i[1]*xyz_i[1]+xyz_i[2]*xyz_i[2]);
    if (r_i > r_max) //move point --> r_i = r_max
    {
      cloud->points[i].x=(xyz_i[0]/r_i)*r_max;
      cloud->points[i].y=(xyz_i[1]/r_i)*r_max;
      cloud->points[i].z=(xyz_i[2]/r_i)*r_max;

      r_i=r_max;
      xyz_i[0]=cloud->points[i].x;
      xyz_i[1]=cloud->points[i].y;
      xyz_i[2]=cloud->points[i].z;
    }
    // features with all points (r_i<r_max & r_i=r_max)

    //Range Histograms for all bin sizes
    histogram_1[int(ceil(r_i/bin_sizes[0]))-1]++;
    histogram_2[int(ceil(r_i/bin_sizes[1]))-1]++;
    histogram_3[int(ceil(r_i/bin_sizes[2]))-1]++;
    histogram_4[int(ceil(r_i/bin_sizes[3]))-1]++;
    histogram_5[int(ceil(r_i/bin_sizes[4]))-1]++;
    histogram_6[int(ceil(r_i/bin_sizes[5]))-1]++;
    histogram_7[int(ceil(r_i/bin_sizes[6]))-1]++;
    histogram_8[int(ceil(r_i/bin_sizes[7]))-1]++;
    histogram_9[int(ceil(r_i/bin_sizes[8]))-1]++;

    //Relative Range & Range Difference
    if(i!=n-1)
    {
      //calculation for next (i+1) point
      xyz_next[0]=cloud->points[i+1].x;
      xyz_next[1]=cloud->points[i+1].y;
      xyz_next[2]=cloud->points[i+1].z;
      r_next=sqrt(xyz_next[0]*xyz_next[0]+xyz_next[1]*xyz_next[1]+xyz_next[2]*xyz_next[2]);

      //mean of relative range
      r_i_r=r_i/r_next;
      mu_old_relative=features1[22];
      features1[22]=features1[22]+(r_i_r-features1[22])/(i+1);
      //for standard deviation of relative range
      Sn_relative=Sn_relative+(r_i_r-mu_old_relative)*(r_i_r-features1[22]);

      if(r_i<r_max && r_next<r_max)
      {
        //mean of relative range
        mu_old_relative_tilde=features1[24];
        features1[24]=features1[24]+(r_i_r-features1[24])/(relative_n+1);
        //for standard deviation of relative range
        Sn_relative_tilde=Sn_relative_tilde+(r_i_r-mu_old_relative_tilde)*(r_i_r-features1[24]);

        relative_n++;
      }

      //range difference
      r_i_d=fabs(r_i-r_next);

      if(r_i<=g_r1 && r_next<=g_r1)//gate g_r1
      {
        //mean of range difference
        mu_old_gr1=features1[26];
        features1[26]=features1[26]+(r_i_d-features1[26])/(gr1_n+1);
        //for standard deviation of range difference
        Sn_gr1=Sn_gr1+(r_i_d-mu_old_gr1)*(r_i_d-features1[26]);

        gr1_n++;
      }

      if(r_i<=g_r2 && r_next<=g_r2)//gate g_r2
      {
        //mean of range difference
        mu_old_gr2=features1[28];
        features1[28]=features1[28]+(r_i_d-features1[28])/(gr2_n+1);
        //for standard deviation of range difference
        Sn_gr2=Sn_gr2+(r_i_d-mu_old_gr2)*(r_i_d-features1[28]);

        gr2_n++;
      }

      if(r_i<=g_r3 && r_next<=g_r3)//gate g_r3
      {
        //mean of range difference
        mu_old_gr3=features1[30];
        features1[30]=features1[30]+(r_i_d-features1[30])/(gr3_n+1);
        //for standard deviation of range difference
        Sn_gr3=Sn_gr3+(r_i_d-mu_old_gr3)*(r_i_d-features1[30]);

        gr3_n++;
      }
    }

    //distance between consecutive points
    if(i!=0)
    {
      delta_i_old=sqrt((xyz_i[0]-xyz_i_old[0])*(xyz_i[0]-xyz_i_old[0])+(xyz_i[1]-xyz_i_old[1])*(xyz_i[1]-xyz_i_old[1])+(xyz_i[2]-xyz_i_old[2])*(xyz_i[2]-xyz_i_old[2]));
      features1[14]+=delta_i_old;

      if(r_i<r_max && r_i_old<r_max)
      {
        delta_i_old=sqrt((xyz_i[0]-xyz_i_old[0])*(xyz_i[0]-xyz_i_old[0])+(xyz_i[1]-xyz_i_old[1])*(xyz_i[1]-xyz_i_old[1])+(xyz_i[2]-xyz_i_old[2])*(xyz_i[2]-xyz_i_old[2]));
        features1[15]+=delta_i_old;
        // Regularity (standard deviation of delta_i_old)
        delta_mu_old=delta_mu;
        delta_mu=delta_mu+(delta_i_old-delta_mu)/(delta_n+1);
        Sn_delta=Sn_delta+(delta_i_old-delta_mu_old)*(delta_i_old-delta_mu);
        delta_n++;

        if(delta_i_old<g_dist)
        {
          features1[16]+=delta_i_old;
        }
      }
    }
    xyz_i_old[0]=xyz_i[0];
    xyz_i_old[1]=xyz_i[1];
    xyz_i_old[2]=xyz_i[2];
    r_i_old=r_i;

    //Volume 1
    features1[0]=features1[0]+(pow(r_i/r_max,3)-features1[0])/(i+1);
    //Average Range 2
    r_i_n=r_i/r_max;
    mu_old=features1[3];
    features1[3]=features1[3]+(r_i_n-features1[3])/(i+1);
    //for Standard Deviation of Range 2
    Sn=Sn+(r_i_n-mu_old)*(r_i_n-features1[3]);
    //mean position for Sphere
    mean[0]=mean[0]+(xyz_i[0]-mean[0])/(i+1);
    mean[1]=mean[1]+(xyz_i[1]-mean[1])/(i+1);
    mean[2]=mean[2]+(xyz_i[2]-mean[2])/(i+1);
    //mean range for Range Kurtosis
    r_mean=r_mean+(r_i-r_mean)/(i+1);

    //features with points r_i<r_max
    if (r_i<r_max)
    {
      //Volume 2
      features1[1]=features1[1]+(pow(r_i/r_max,3)-features1[1])/(n_tilde+1);
      //Average Range 1
      r_i_n=r_i/r_max;
      mu_old_tilde=features1[2];
      features1[2]=features1[2]+(r_i_n-features1[2])/(n_tilde+1);
      //for Standard Deviation of Range 1
      Sn_tilde=Sn_tilde+(r_i_n-mu_old_tilde)*(r_i_n-features1[2]);

      //mean position for Centroid
      mean_tilde[0]=mean_tilde[0]+(xyz_i[0]-mean_tilde[0])/(n_tilde+1);
      mean_tilde[1]=mean_tilde[1]+(xyz_i[1]-mean_tilde[1])/(n_tilde+1);
      mean_tilde[2]=mean_tilde[2]+(xyz_i[2]-mean_tilde[2])/(n_tilde+1);

      //mean range for Range Kurtosis
      r_mean_tilde=r_mean_tilde+(r_i-r_mean_tilde)/(n_tilde+1);

      //Curvature
      if(i!=0 && i!=n-1)//not for first and last index
      {
        //calculation for last (i-1) point
        xyz_last[0]=cloud->points[i-1].x;
        xyz_last[1]=cloud->points[i-1].y;
        xyz_last[2]=cloud->points[i-1].z;
        r_last=sqrt(xyz_last[0]*xyz_last[0]+xyz_last[1]*xyz_last[1]+xyz_last[2]*xyz_last[2]);

        if(r_last<r_max && r_next<r_max)
        {
          //calculation of distances between three points
          delta_i_last=sqrt((xyz_i[0]-xyz_last[0])*(xyz_i[0]-xyz_last[0])+(xyz_i[1]-xyz_last[1])*(xyz_i[1]-xyz_last[1])+(xyz_i[2]-xyz_last[2])*(xyz_i[2]-xyz_last[2]));
          delta_i_next=sqrt((xyz_i[0]-xyz_next[0])*(xyz_i[0]-xyz_next[0])+(xyz_i[1]-xyz_next[1])*(xyz_i[1]-xyz_next[1])+(xyz_i[2]-xyz_next[2])*(xyz_i[2]-xyz_next[2]));
          delta_last_next=sqrt((xyz_last[0]-xyz_next[0])*(xyz_last[0]-xyz_next[0])+(xyz_last[1]-xyz_next[1])*(xyz_last[1]-xyz_next[1])+(xyz_last[2]-xyz_next[2])*(xyz_last[2]-xyz_next[2]));

          if(delta_i_last<g_dist && delta_i_next<g_dist && delta_last_next<g_dist)
          {
            //area of triangle
            s_tri=(delta_i_last+delta_i_next+delta_last_next)/2;
            area_tri=sqrt(s_tri*(s_tri-delta_i_last)*(s_tri-delta_i_next)*(s_tri-delta_last_next));
            //curvature
            curv_i=4*area_tri/(delta_i_last*delta_i_next*delta_last_next);
            //mean of curvature
            curv_mu_old=features1[18];
            features1[18]=features1[18]+(curv_i-features1[18])/(curv_n+1);
            //for standard deviation of curvature
            Sn_curv=Sn_curv+(curv_i-curv_mu_old)*(curv_i-features1[18]);
            curv_n++;
          }
        }
      }
      n_tilde++;
    }
  }

  features1[9]=sqrt(mean_tilde[0]*mean_tilde[0]+mean_tilde[1]*mean_tilde[1]+mean_tilde[2]*mean_tilde[2]);

  // ######## Loop 2 through point cloud ########
  n_tilde=0;
  for(int i = 0; i<n; i++)
  {
    xyz_i[0]=cloud->points[i].x;
    xyz_i[1]=cloud->points[i].y;
    xyz_i[2]=cloud->points[i].z;

    A[0][0]+=xyz_i[0]*(xyz_i[0]-mean[0]);
    A[1][0]+=xyz_i[1]*(xyz_i[0]-mean[0]);
    A[2][0]+=xyz_i[2]*(xyz_i[0]-mean[0]);
    A[0][1]+=xyz_i[0]*(xyz_i[1]-mean[1]);
    A[1][1]+=xyz_i[1]*(xyz_i[1]-mean[1]);
    A[2][1]+=xyz_i[2]*(xyz_i[1]-mean[1]);
    A[0][2]+=xyz_i[0]*(xyz_i[2]-mean[2]);
    A[1][2]+=xyz_i[1]*(xyz_i[2]-mean[2]);
    A[2][2]+=xyz_i[2]*(xyz_i[2]-mean[2]);

    B[0]+=(xyz_i[0]*xyz_i[0]+xyz_i[1]*xyz_i[1]+xyz_i[2]*xyz_i[2])*(xyz_i[0]-mean[0]);
    B[1]+=(xyz_i[0]*xyz_i[0]+xyz_i[1]*xyz_i[1]+xyz_i[2]*xyz_i[2])*(xyz_i[1]-mean[1]);
    B[2]+=(xyz_i[0]*xyz_i[0]+xyz_i[1]*xyz_i[1]+xyz_i[2]*xyz_i[2])*(xyz_i[2]-mean[2]);

    r_i=sqrt(xyz_i[0]*xyz_i[0]+xyz_i[1]*xyz_i[1]+xyz_i[2]*xyz_i[2]);

    //Range Kurtosis
    kurtosis_2=kurtosis_2+(pow(r_i-r_mean,2)-kurtosis_2)/(i+1);
    kurtosis_4=kurtosis_4+(pow(r_i-r_mean,4)-kurtosis_4)/(i+1);

    if (r_i<r_max)
    {
      //Centroid mean distance
      delta_centroid[0]=mean_tilde[0]-xyz_i[0];
      delta_centroid[1]=mean_tilde[1]-xyz_i[1];
      delta_centroid[2]=mean_tilde[2]-xyz_i[2];
      distance_centroid=sqrt(delta_centroid[0]*delta_centroid[0]+delta_centroid[1]*delta_centroid[1]+delta_centroid[2]*delta_centroid[2]);
      mu_centroid_old=features1[10];
      features1[10]=features1[10]+(distance_centroid-features1[10])/(n_tilde+1);
      //for Centroid standard deviation
      Sn_centroid=Sn_centroid+(distance_centroid-mu_centroid_old)*(distance_centroid-features1[10]);

      //Range Kurtosis
      kurtosis_2_tilde=kurtosis_2_tilde+(pow(r_i-r_mean_tilde,2)-kurtosis_2_tilde)/(n_tilde+1);
      kurtosis_4_tilde=kurtosis_4_tilde+(pow(r_i-r_mean_tilde,4)-kurtosis_4_tilde)/(n_tilde+1);

      n_tilde++;
    }
  }

  //converting to Eigen
  for(int i=0;i<3;i++){for (int j=0;j<3;j++){A_sphere(i,j)=A[i][j];}}
  for (int i=0;i<3;i++){B_sphere(i)=B[i];}

  A_sphere=A_sphere*2/n;
  B_sphere/=n;
  //Calculation of center location of fitted sphere
  ATA=A_sphere.transpose()*A_sphere;
  c_sphere=ATA.inverse()*A_sphere.transpose()*B_sphere;
  double xyz_sphere[3]={c_sphere(0),c_sphere(1),c_sphere(2)};

  // ######## Loop 3 through point cloud ########
  for(int i = 0; i<n; i++)
  {
    xyz_i[0]=cloud->points[i].x;
    xyz_i[1]=cloud->points[i].y;
    xyz_i[2]=cloud->points[i].z;
    sum_R_sphere+=(xyz_i[0]-xyz_sphere[0])*(xyz_i[0]-xyz_sphere[0])+(xyz_i[1]-xyz_sphere[1])*(xyz_i[1]-xyz_sphere[1])+(xyz_i[2]-xyz_sphere[2])*(xyz_i[2]-xyz_sphere[2]);
  }

  //Calculation of radius of fitted sphere
  R_sphere=sqrt(sum_R_sphere/n);

  // ######## Loop 4 through point cloud ########
  for(int i = 0; i<n; i++)
  {
    delta_sphere[0]=xyz_sphere[0]-cloud->points[i].x;
    delta_sphere[1]=xyz_sphere[1]-cloud->points[i].y;
    delta_sphere[2]=xyz_sphere[2]-cloud->points[i].z;
    abs_delta_sphere=sqrt(delta_sphere[0]*delta_sphere[0]+delta_sphere[1]*delta_sphere[1]+delta_sphere[2]*delta_sphere[2]);
    features1[7]=features1[7]+((R_sphere-abs_delta_sphere)*(R_sphere-abs_delta_sphere)-features1[7])/(i+1);
  }

  //remaining calculation of features1
  features1[4]=sqrt(Sn_tilde/n_tilde);
  features1[5]=sqrt(Sn/n);
  features1[6]=R_sphere/r_max;
  features1[7]=features1[7]/R_sphere;
  features1[8]=sqrt(xyz_sphere[0]*xyz_sphere[0]+xyz_sphere[1]*xyz_sphere[1]+xyz_sphere[2]*xyz_sphere[2])/r_max;
  features1[11]=sqrt(Sn_centroid/n_tilde);
  features1[12]=n-n_tilde;
  features1[13]=n_tilde;
  features1[17]=sqrt(Sn_delta/delta_n);
  features1[19]=sqrt(Sn_curv/curv_n);
  features1[20]=kurtosis_4_tilde/(kurtosis_2_tilde*kurtosis_2_tilde)-3;
  features1[21]=kurtosis_4/(kurtosis_2*kurtosis_2)-3;
  features1[23]=sqrt(Sn_relative/(n-1));
  features1[25]=sqrt(Sn_relative_tilde/relative_n);
  features1[26]=features1[26]/g_r1;
  features1[27]=sqrt(Sn_gr1/gr1_n)/g_r1;
  features1[28]=features1[28]/g_r2;
  features1[29]=sqrt(Sn_gr2/gr2_n)/g_r2;
  features1[30]=features1[30]/g_r3;
  features1[31]=sqrt(Sn_gr3/gr3_n)/g_r3;

  //creation of feature vector
  //features of type 1
  for (int i=0;i<sizeof(features1)/sizeof(features1[0]);i++) {
    if (std::isnan(features1[i])){
      feature_vector.push_back(0.0);
    }
    else {
      feature_vector.push_back(features1[i]);
    }
  }
  //range histograms
  for (int i=0;i<sizeof(histogram_1)/sizeof(histogram_1[0]);i++) {feature_vector.push_back(histogram_1[i]);}
  for (int i=0;i<sizeof(histogram_2)/sizeof(histogram_2[0]);i++) {feature_vector.push_back(histogram_2[i]);}
  for (int i=0;i<sizeof(histogram_3)/sizeof(histogram_3[0]);i++) {feature_vector.push_back(histogram_3[i]);}
  for (int i=0;i<sizeof(histogram_4)/sizeof(histogram_4[0]);i++) {feature_vector.push_back(histogram_4[i]);}
  for (int i=0;i<sizeof(histogram_5)/sizeof(histogram_5[0]);i++) {feature_vector.push_back(histogram_5[i]);}
  for (int i=0;i<sizeof(histogram_6)/sizeof(histogram_6[0]);i++) {feature_vector.push_back(histogram_6[i]);}
  for (int i=0;i<sizeof(histogram_7)/sizeof(histogram_7[0]);i++) {feature_vector.push_back(histogram_7[i]);}
  for (int i=0;i<sizeof(histogram_8)/sizeof(histogram_8[0]);i++) {feature_vector.push_back(histogram_8[i]);}
  for (int i=0;i<sizeof(histogram_9)/sizeof(histogram_9[0]);i++) {feature_vector.push_back(histogram_9[i]);}

  auto finish_hr = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_hr = finish_hr - start_hr;
  ROS_DEBUG("Scan descriptor of size %i in %.1lf ms successfully calculated with point cloud of size %i!",
           n_feature_vector,duration_hr.count()*1000,n);
}

void lidar_loopdetector::get_feature_vector(std::vector<double>&features, std::vector<int>&lengths){
  features=feature_vector;
  lengths.push_back(32);//length of features1
  for (int i=0;i<sizeof(bin_sizes)/sizeof (bin_sizes[0]);i++)
  {
    //length of every range histogram
    lengths.push_back(int(ceil(r_max/bin_sizes[i])));
  }

}

void lidar_loopdetector::get_feature_vector(std::vector<double>&features){
  features=feature_vector;
}

void read_params(ros::NodeHandle nh,
                 double &loop_probability_min,
                 double &R_min,
                 double &r_max,
                 std::string &scan_topic_name,
                 double &z_limit,
                 int &sky_direction,
                 float &leafsize,
                 int &n_verify,
                 std::string &odom_topic_name,
                 int &n_max_nodes,
                 double &alpha_thres,
                 int &n_ms_verify,
                 double &R_ms_verify,
                 int &n_ms_start,
                 float &t_max,
                 float &i_limit,
                 std::string &path_clouds,
                 double &r_limit,
                 int &n_max_points,
                 int &n_min_points,
                 int &min_inliers,
                 double &beta,
                 double &throttle_dur)
{
  nh.getParam("/cmr_lidarloop/loop_probability_min", loop_probability_min);
  nh.getParam("/cmr_lidarloop/R_min", R_min);
  nh.getParam("/cmr_lidarloop/r_max", r_max);
  nh.getParam("/cmr_lidarloop/scan_topic_name", scan_topic_name);
  nh.getParam("/cmr_lidarloop/z_limit", z_limit);
  nh.getParam("/cmr_lidarloop/sky_direction", sky_direction);
  nh.getParam("/cmr_lidarloop/leafsize", leafsize);
  nh.getParam("/cmr_lidarloop/n_verify", n_verify);
  nh.getParam("/cmr_lidarloop/odom_topic_name", odom_topic_name);
  nh.getParam("/cmr_lidarloop/n_max_nodes", n_max_nodes);
  nh.getParam("/cmr_lidarloop/alpha_thres", alpha_thres);
  nh.getParam("/cmr_lidarloop/n_ms_verify", n_ms_verify);
  nh.getParam("/cmr_lidarloop/R_ms_verify", R_ms_verify);
  nh.getParam("/cmr_lidarloop/n_ms_start", n_ms_start);
  nh.getParam("/cmr_lidarloop/t_max", t_max);
  nh.getParam("/cmr_lidarloop/i_limit", i_limit);
  nh.getParam("/cmr_lidarloop/path_clouds", path_clouds);
  nh.getParam("/cmr_lidarloop/r_limit", r_limit);
  nh.getParam("/cmr_lidarloop/n_max_points", n_max_points);
  nh.getParam("/cmr_lidarloop/n_min_points", n_min_points);
  nh.getParam("/cmr_lidarloop/min_inliers", min_inliers);
  nh.getParam("/cmr_lidarloop/beta", beta);
  nh.getParam("/cmr_lidarloop/throttle_dur", throttle_dur);

  ROS_INFO("The cmr_lidarloop parameters were read successfully.");
}
#endif // LIDAR_LOOPDETECTION_H
