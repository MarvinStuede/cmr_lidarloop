/**
 * @file   lidar_loopdetection.cpp
 * @author Tim-Lukas Habich
 * @date   02/2020
 *
 * @brief  LiDAR-based loop detection in use with RTAB-Map
 *         (with the help of ExternalLoopDetectionExample.cpp in rtabmap_ros)
 *         detector_strategy==1 -> Detector according to the method of
 *	   Granström - Learning to Close the Loop from 3D Point Clouds
 */

#include "cmr_lidarloop/lidar_loopdetection.h"
#include "cmr_lidarloop/LiDAR_Loopdetector.h"
#include "cmr_lidarloop/lidar_registration.h"
#include <cmr_lidarloop/LiDAR_RegistrationAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <pcl/registration/icp.h>
#include <cstdlib>
#include <pcl/io/pcd_io.h>
#include <boost/serialization/access.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <sstream>
#include <string>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <rtabmap_ros/ScanDescriptor.h>
#include <rtabmap/core/Compression.h>
#include "rtabmap_ros/GetMap.h"
#include "rtabmap_ros/GetNodeData.h"
#include <algorithm>
#include <roscpp/SetLoggerLevel.h>
#include <cmath>
typedef message_filters::sync_policies::ExactTime<rtabmap_ros::MapData, rtabmap_ros::Info> MyInfoMapSyncPolicy;

//##################Parameters##################
//#########cfg/cmr_lidarloop_params.yaml########
double loop_probability_min, R_min, r_max, z_limit, alpha_thres, R_ms_verify, r_limit, beta, throttle_dur;
std::string scan_topic_name, odom_topic_name, path_clouds;
float leafsize, t_max, i_limit;
int n_verify, n_max_nodes, n_ms_verify, n_ms_start, n_max_points, n_min_points, min_inliers, sky_direction;
//##############################################

ros::ServiceClient addLinkSrv;
ros::ServiceClient detector_client;
int STM_size;
int id_old=-1;
int fromId_old=-1;
int toId_old=-1;
std::vector<double> R_search_history;
std::vector<std::vector<int>> added_loops;

//Publisher of scan descriptor with corresponding scan
ros::Publisher scanDescriptorPub;

//Action Client for Registration
actionlib::SimpleActionClient<cmr_lidarloop::LiDAR_RegistrationAction>* registration_ac;

//Current uncertainty of odometry (length of largest major axis of 95% confidence ellipse)
//-> is set to 0 at detected loop with LiDAR loop detector
double R_odom=0;

//RTAB-Maps current mode (mapping vs. localization)
bool MappingMode;

bool result_processed = true;

//Multi session
bool MultiSession;
std::vector<int> loopcandidates_ms;

ros::Time time_last_run = ros::Time(0);

bool featureVectorValid(const std::vector<double> &feature_vector){
  try {
    if(feature_vector.empty()) throw "Feature vector empty";
    for(const auto &el : feature_vector)
      if(std::isnan(el)) throw "NaN in feature vector";

  } catch (std::string s) {
    ROS_ERROR(s.c_str());
    return false;
  }
  return true;
}

class lidar_data
{
public:
  lidar_data();
  ~lidar_data();

  std::vector<int> all_ids;
  std::map<int, std::vector<double>> xyz;
  std::map<int, Eigen::Matrix4f> transformations;
  std::map<int, std::vector<double>> features;

  void read_rtabmap_db();
};
lidar_data::lidar_data(){
  features.clear();
};
void lidar_data::read_rtabmap_db(){
  //Read LiDAR data from rtabmap.db
  rtabmap_ros::GetMap getMapSrv;
  getMapSrv.request.global = true;
  getMapSrv.request.optimized = true;
  getMapSrv.request.graphOnly = true; // If you want the scans, set this to false
  if(!ros::service::call("/rtabmap/get_map_data", getMapSrv))
  {
    ROS_ERROR("Can't call \"/rtabmap/get_map_data\" service");
    return;
  }

  if (getMapSrv.response.data.nodes.size()>0){
    //Multi Session
    MultiSession=true;

    ROS_INFO("Adding %d nodes to LiDAR memory from rtabmap.db. Multi session operation switched on.", (int)getMapSrv.response.data.nodes.size());

    for(size_t i=0; i<getMapSrv.response.data.nodes.size(); ++i)
    {
      rtabmap::Signature s = rtabmap_ros::nodeDataFromROS(getMapSrv.response.data.nodes[i]);

      all_ids.push_back(s.id());

      //Get scan descriptor
      //std::cout<<"Descriptor size "<<s.sensorData().globalDescriptors().size()<<std::endl;
      //ROS_ASSERT(s.sensorData().globalDescriptors().size() == 1); //Assuming we are using only one descriptor
      //ROS_ASSERT(s.sensorData().globalDescriptors()[0].type() == 0); // Assuming descriptor is the same type that we set
      ROS_ASSERT(s.sensorData().globalDescriptors()[0].data().type() == CV_64FC1); // Make sure we get the same type that we set
      ROS_ASSERT(s.sensorData().globalDescriptors()[0].info().type() == CV_32FC1); // Make sure we get the same type that we set
      std::vector<double> current_feature_vector(s.sensorData().globalDescriptors()[0].data().total());
      std::vector<int> lengths_feature_vector(s.sensorData().globalDescriptors()[0].info().total());
      memcpy(current_feature_vector.data(), s.sensorData().globalDescriptors()[0].data().data, current_feature_vector.size()*sizeof(double));
      memcpy(lengths_feature_vector.data(), s.sensorData().globalDescriptors()[0].info().data, lengths_feature_vector.size()*sizeof(int));

      //Save features


      if(featureVectorValid(current_feature_vector)){
        features[s.id()]=current_feature_vector;
      }
      else{
        ROS_ERROR("Read feature vector empty");
      }
    }

    //Optimized poses
    ROS_ASSERT(getMapSrv.response.data.graph.poses.size() == getMapSrv.response.data.graph.posesId.size());
    for(size_t i=0; i<getMapSrv.response.data.graph.poses.size(); ++i)
    {
      rtabmap::Transform t = rtabmap_ros::transformFromPoseMsg(getMapSrv.response.data.graph.poses[i]);
      std::vector<double> temp_xyz={t.x(), t.y(), t.z()};
      Eigen::Matrix4f current_transformation;
      current_transformation <<t.r11(),t.r12(),t.r13(),t.o14(),
          t.r21(),t.r22(),t.r23(),t.o24(),
          t.r31(),t.r32(),t.r33(),t.o34(),
          0,0,0,1;
      int id = getMapSrv.response.data.graph.posesId[i];
      xyz[id]=temp_xyz;
      transformations[id] = current_transformation;
    }
  }
  else{
    MultiSession=false;
    ROS_INFO("No database is used. Multi session operation switched off.");
  }
}
//Save history of R_search and of added_loops at destruction
lidar_data::~lidar_data(){
  std::ofstream file;
  file.open ("cmr_lidarloop_history.csv");
  file<<"History of R_search in m:\n";
  for (int i = 0; i < R_search_history.size()-1; ++i) {
    file << R_search_history[i]<<",";
  }
  file<<R_search_history[R_search_history.size()-1]<<"\n\n";

  file<<"History of added_loops:\n";
  for (int i = 0; i < added_loops.size(); ++i) {
    file << added_loops[i][0]<<","<<added_loops[i][1]<<"\n";
  }

  file.close();
}

//Compute scan descriptor for RTAB-Map
void scanCallback(const sensor_msgs::PointCloud2ConstPtr & pointCloud2Msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*pointCloud2Msg, *cloud);

  //Compute features with raw cloud
  lidar_loopdetector current_lidar(cloud,1,r_max);
  std::vector<double> current_feature_vector;
  std::vector<int> lengths_feature_vector;
  current_lidar.get_feature_vector(current_feature_vector,lengths_feature_vector);
  if(!featureVectorValid(current_feature_vector))
    return;

  //Publish scan descriptor for RTAB-Map
  rtabmap_ros::ScanDescriptor scanDescriptor;
  scanDescriptor.header = pointCloud2Msg->header;
  scanDescriptor.scan_cloud = *pointCloud2Msg;
  scanDescriptor.global_descriptor.type=0;
  scanDescriptor.global_descriptor.info=rtabmap::compressData(cv::Mat(1, lengths_feature_vector.size(), CV_32FC1, (void*)lengths_feature_vector.data()));
  scanDescriptor.global_descriptor.data=rtabmap::compressData(cv::Mat(1, current_feature_vector.size(), CV_64FC1, (void*)current_feature_vector.data()));
  scanDescriptorPub.publish(scanDescriptor);
}

double uncertainty_0;
bool set_uncertainty_to_zero=true;
//Compute current uncertainty of position -> is set to zero at the beginning and at loop detection
void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  //Current xy covariance -> maximum eigenvalue
  Eigen::Matrix2d current_xy_covariance;
  current_xy_covariance<<msg->pose.covariance[0],msg->pose.covariance[1],
      msg->pose.covariance[6],msg->pose.covariance[7];
  Eigen::EigenSolver<Eigen::Matrix2d> es(current_xy_covariance, false);
  std::vector<double> current_ev={abs(es.eigenvalues()[0]),abs(es.eigenvalues()[1])};
  double current_max_ev=*std::max_element(current_ev.begin(),current_ev.end());

  //Compute uncertainty -> length of largest major axis of 95% confidence ellipse
  double current_uncertainty=2*sqrt(5.991*current_max_ev);
  if (set_uncertainty_to_zero){
    uncertainty_0=current_uncertainty;
    set_uncertainty_to_zero=false;
  }
  R_odom=current_uncertainty-uncertainty_0;
}

//LiDAR memory inside cmr_lidarloop
lidar_data lidar_memory;

void addLinkToRTABMap(){
  cmr_lidarloop::LiDAR_RegistrationResult registration_result;
  registration_result=*registration_ac->getResult();
  //Transform to Eigen4f
  std::vector<float> t_vectr=registration_result.transformation_vctr;
  Eigen::Matrix4f loop_transformation;
  loop_transformation <<  t_vectr[0], t_vectr[1], t_vectr[2], t_vectr[3],
      t_vectr[4], t_vectr[5], t_vectr[6], t_vectr[7],
      t_vectr[8], t_vectr[9], t_vectr[10], t_vectr[11],
      t_vectr[12], t_vectr[13], t_vectr[14], t_vectr[15];
  //Translational offset
  float t_offset=sqrt(t_vectr[3]*t_vectr[3]+t_vectr[7]*t_vectr[7]+t_vectr[11]*t_vectr[11]);
  //Save Ids
  int fromId = registration_result.id;
  int toId = registration_result.loop_id;

  if (fromId==fromId_old && toId==toId_old){
    ROS_INFO("Link between %d and %d already sent to RTAB-Map.",fromId, toId);
  }
  else if (t_offset>t_max) {
    ROS_INFO("Translational offset (%.1f m) is larger than desired maximum (%.1f m). Link rejected.",t_offset, t_max);
  }
  else if(registration_result.n_inliers<min_inliers){
    ROS_INFO("Number of inliers (%d) is less than desired minimum (%d). Link rejected.",
             registration_result.n_inliers,min_inliers);
  }
  else{
    ROS_INFO("Link between %d and %d to add in RTAB-Map.",fromId, toId);

    fromId_old=fromId;
    toId_old=toId;

    //Create information matrix
    cv::Mat infMatrix;
    double variance=registration_result.variance;
    if(variance==0){
      //Take identity
      infMatrix = cv::Mat::eye(6,6,CV_64FC1);
    }
    else{
      //Information matrix = covariance matrix^-1
      double diag_infMatrix=1/variance;
      infMatrix = cv::Mat::eye(6,6,CV_64FC1);
      infMatrix=infMatrix*diag_infMatrix;
    }
    //std::cout<<"InfMatrix: "<<infMatrix<<std::endl;

    //Add link in RTAB-Map
    rtabmap::Transform t=rtabmap::Transform::fromEigen4f(loop_transformation.inverse());
    rtabmap::Link link(fromId, toId, rtabmap::Link::kUserClosure, t, infMatrix);
    rtabmap_ros::AddLinkRequest req;
    rtabmap_ros::linkToROS(link, req.link);
    rtabmap_ros::AddLinkResponse res;
    if(!addLinkSrv.call(req, res))
    {
      ROS_ERROR("Failed to call %s service", addLinkSrv.getService().c_str());
    }
    else{
      ROS_INFO("Link added in RTAB-Map! Pose uncertainty is reset.");
      set_uncertainty_to_zero=true;
      std::vector<int> temp_pair{fromId,toId};
      added_loops.push_back(temp_pair);
      /*//DEBUG global optimized map
      rtabmap_ros::GetMap getMapSrv_global;
      getMapSrv_global.request.global = true;
      getMapSrv_global.request.optimized = true;
      getMapSrv_global.request.graphOnly = true; // If you want the scans, set this to false
      if(!ros::service::call("/rtabmap/get_map_data", getMapSrv_global))
      {
        ROS_ERROR("Can't call \"/rtabmap/get_map_data\" service");
      }
      //DEBUG local optimized map
      rtabmap_ros::GetMap getMapSrv_local;
      getMapSrv_local.request.global = false;
      getMapSrv_local.request.optimized = true;
      getMapSrv_local.request.graphOnly = true; // If you want the scans, set this to false
      if(!ros::service::call("/rtabmap/get_map_data", getMapSrv_local))
      {
        ROS_ERROR("Can't call \"/rtabmap/get_map_data\" service");
      }
      //Debug Ids WM
      std::vector<int> Ids_WM_debug=infoMsg->wmState;
      std::sort(Ids_WM_debug.begin(), Ids_WM_debug.end());*/
    }
  }
}
void mapDataCallback(const rtabmap_ros::MapDataConstPtr & mapDataMsg, const rtabmap_ros::InfoConstPtr & infoMsg)
{
  //ROS_INFO("\n\t\t\t\tReceived map data!");
  auto start_hr = std::chrono::high_resolution_clock::now();

  rtabmap::Statistics stats;
  rtabmap_ros::infoFromROS(*infoMsg, stats);

  //Add link in RTAB-Map, if action server computed new transformation
  if(!result_processed && registration_ac->getState()==actionlib::SimpleClientGoalState::SUCCEEDED){
    addLinkToRTABMap();
    result_processed = true;
  }

  if(stats.data().at("Loop/Id/") > 0){ //If a loop was found already
    if((ros::Time::now() - time_last_run).toSec() < throttle_dur){
      return; //When localized, throttle detection while waiting a specific duration before next iteration
    }
    else
      time_last_run = ros::Time::now();
  }


  bool smallMovement = (bool)uValue(stats.data(), rtabmap::Statistics::kMemorySmall_movement(), 0.0f);
  bool fastMovement = (bool)uValue(stats.data(), rtabmap::Statistics::kMemoryFast_movement(), 0.0f);

  if(smallMovement || fastMovement)
  {
    // The signature has been ignored from rtabmap, don't process it
    return;
  }

  rtabmap::Transform mapToOdom;
  std::map<int, rtabmap::Transform> poses;
  std::multimap<int, rtabmap::Link> links;
  std::map<int, rtabmap::Signature> signatures;
  rtabmap_ros::mapDataFromROS(*mapDataMsg, poses, links, signatures, mapToOdom);

  if(!signatures.empty() && signatures.rbegin()->second.sensorData().isValid())
  {
    int id = signatures.rbegin()->first;

    if(id_old==id){
      //id has not changed -> ignore
      ROS_INFO("Id has not changed. Ignore data.");
      return;
    }

    const rtabmap::SensorData & s =  signatures.rbegin()->second.sensorData();
    cv::Mat rgb;
    rtabmap::LaserScan scan;
    s.uncompressDataConst(&rgb, 0, &scan);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = rtabmap::util3d::laserScanToPointCloudI(scan, scan.localTransform());

    //Get scan descriptor

    if(s.globalDescriptors().size() != 1){
      ROS_WARN("Map descriptor wrong size");
      return;
    }
    ROS_ASSERT(s.globalDescriptors().size() == 1); //Assuming we are using only one descriptor
    ROS_ASSERT(s.globalDescriptors()[0].type() == 0); // Assuming descriptor is the same type that we set
    ROS_ASSERT(s.globalDescriptors()[0].data().type() == CV_64FC1); // Make sure we get the same type that we set
    ROS_ASSERT(s.globalDescriptors()[0].info().type() == CV_32FC1); // Make sure we get the same type that we set
    std::vector<double> current_feature_vector(s.globalDescriptors()[0].data().total());
    std::vector<int> lengths_feature_vector(s.globalDescriptors()[0].info().total());
    memcpy(current_feature_vector.data(), s.globalDescriptors()[0].data().data, current_feature_vector.size()*sizeof(double));
    memcpy(lengths_feature_vector.data(), s.globalDescriptors()[0].info().data, lengths_feature_vector.size()*sizeof(int));

    if(!featureVectorValid(current_feature_vector))
      return;


    //Save current pose
    std::vector<double> current_xyz={poses[id].o14(),poses[id].o24(),poses[id].o34()};
    Eigen::Matrix4f current_transformation;
    current_transformation << poses[id].r11(),poses[id].r12(),poses[id].r13(),poses[id].o14(),
        poses[id].r21(),poses[id].r22(),poses[id].r23(),poses[id].o24(),
        poses[id].r31(),poses[id].r32(),poses[id].r33(),poses[id].o34(),
        0,0,0,1;


    //LiDAR_Loopdetector Service

    //Determine current search radius for loops
    double R_search=R_min+beta*R_odom;
    R_search_history.push_back(R_search);
    //All Ids in RTAB-Maps WM
    std::vector<int> Ids_WM=infoMsg->wmState;
    std::sort(Ids_WM.begin(), Ids_WM.end());
    //Ids in WM AND in local map
    std::vector<int> Ids_WM_local=mapDataMsg->graph.posesId;
    //STM must be full and at least one node for calculation in WM
    if(Ids_WM.size()>STM_size){
      int n_nodes_calc=Ids_WM.size()-STM_size;
      ROS_DEBUG("Number of nodes for calculation in WM: %d",n_nodes_calc);
      std::vector<int> Ids_R;
      //Ratio between WM Ids in local map and all WM Ids
      //-> small ratio means a bad localization of the robot in the map
      double alpha=double(Ids_WM_local.size())/double(Ids_WM.size());
      ROS_DEBUG("Alpha=%lf",alpha);

      if(MultiSession && (added_loops.size()<n_ms_start || alpha<alpha_thres)){
        //Multi session operation -> Position in map of last session is not known
        //Search within radius of current position makes no sense -> search for loops throughout WM
        std::vector<int> Ids_ms(Ids_WM.begin(),Ids_WM.begin()+n_nodes_calc);
        Ids_R=Ids_ms;
        ROS_INFO("Multi session operation: Searching for loops throughout WM.");
      }

      else{
        //Search in radius R_search of current position for Ids in WM
        double c_xyz[3] = {current_xyz[0],current_xyz[1],current_xyz[2]};
        double temp_xyz[3],temp_R;
        int temp_id;
        for (int i = 0; i < n_nodes_calc; ++i) {
          temp_id=Ids_WM[i];
          if(lidar_memory.xyz.find(temp_id) != lidar_memory.xyz.end()){
            temp_xyz[0]=lidar_memory.xyz[temp_id][0];
            temp_xyz[1]=lidar_memory.xyz[temp_id][1];
            temp_xyz[2]=lidar_memory.xyz[temp_id][2];

            temp_R=sqrt((c_xyz[0]-temp_xyz[0])*(c_xyz[0]-temp_xyz[0])
                +(c_xyz[1]-temp_xyz[1])*(c_xyz[1]-temp_xyz[1])
                +(c_xyz[2]-temp_xyz[2])*(c_xyz[2]-temp_xyz[2]));
            bool features_exist = lidar_memory.features.find(Ids_WM[i]) != lidar_memory.features.end();
            if(temp_R<R_search && features_exist){
              Ids_R.push_back(Ids_WM[i]);
            }
          }
        }
        ROS_DEBUG("Number of nodes within R (%.1lf m): %d",R_search,Ids_R.size());
      }

      if (Ids_R.size()>n_max_nodes){
        ROS_DEBUG("Too many nodes for calculation (maximum number: %d) -> Create random subset.",n_max_nodes);
        //Shuffle vector randomly
        std::random_shuffle(Ids_R.begin(), Ids_R.end());
        //Delete nodes
        std::vector<int> temp_Ids_R(Ids_R.begin(),Ids_R.begin()+n_max_nodes);
        Ids_R=temp_Ids_R;
      }

      //Save all feature vectors in one 1D-Vector
      std::vector<double> features_1D_R;
      for (int i = 0; i < Ids_R.size(); ++i) {
        features_1D_R.insert(features_1D_R.end(),lidar_memory.features[Ids_R[i]].begin(),lidar_memory.features[Ids_R[i]].end());
      }

      //Create request for LiDAR_Loopdetector Service
      cmr_lidarloop::LiDAR_Loopdetector detector_srv;
      detector_srv.request.current_features = current_feature_vector;
      detector_srv.request.step=current_feature_vector.size();
      detector_srv.request.memory_features=features_1D_R;
      detector_srv.request.memory_ids=Ids_R;
      detector_srv.request.lengths_features=lengths_feature_vector;
      detector_srv.request.loop_probability_min=loop_probability_min;

      if (detector_client.call(detector_srv)){
        if (detector_srv.response.loop_id != -1){
          ROS_INFO("Loop detected with LiDAR Data between %d and %d"
                   ,id, detector_srv.response.loop_id);
          //Verify detected loop
          bool loop_verified=false;

          if(MultiSession && (added_loops.size()<n_ms_start || alpha<alpha_thres)){
            //Multi session operation
            //Verfification: n_ms_verify consecutive loop candidates must be from the same place
            //-> all candidates must lie within a radius R_ms_verify
            loopcandidates_ms.push_back(detector_srv.response.loop_id);
            if (loopcandidates_ms.size()>=n_ms_verify){
              //Enough consecutive loop candidates -> start verification
              std::vector<int> temp_loopcandidates(loopcandidates_ms.end()-n_ms_verify,loopcandidates_ms.end());
              //Compute mean xy position
              double mean_xy[2]={0,0};
              for (int i = 0; i < n_ms_verify; ++i) {
                int temp_id=temp_loopcandidates[i];
                mean_xy[0]+=lidar_memory.xyz[temp_id][0];
                mean_xy[1]+=lidar_memory.xyz[temp_id][1];
              }
              mean_xy[0]=mean_xy[0]/n_ms_verify;
              mean_xy[1]=mean_xy[1]/n_ms_verify;
              //All candidates must lie within R_ms_verify from mean_xy
              int count_candidates=0;
              double temp_xy[2],temp_R;
              for (int i = 0; i < n_ms_verify; ++i) {
                int temp_id=temp_loopcandidates[i];
                temp_xy[0]=lidar_memory.xyz[temp_id][0];
                temp_xy[1]=lidar_memory.xyz[temp_id][1];

                temp_R=sqrt((mean_xy[0]-temp_xy[0])*(mean_xy[0]-temp_xy[0])
                    +(mean_xy[1]-temp_xy[1])*(mean_xy[1]-temp_xy[1]));

                if(temp_R<R_ms_verify){
                  count_candidates+=1;
                }
              }
              if(count_candidates==n_ms_verify){
                loop_verified=true;
                ROS_INFO("%d consecutive candidates lie within radius of %.1lf m. Loop verified.",count_candidates,R_ms_verify);
              }
              else{
                ROS_INFO("Only %d out of %d consecutive candidates lie within radius of %.1lf m. Loop rejected.",count_candidates,n_ms_verify,R_ms_verify);
              }
            }
          }
          else{
            if (n_verify>0){
              //Verify loop: detect at least another loop between current id and [loop_id-n_verify,loop_id+n_verify]

              //Get Ids for verification
              std::sort(lidar_memory.all_ids.begin(), lidar_memory.all_ids.end());
              std::vector<int>::iterator it_loop_id = std::find(lidar_memory.all_ids.begin(), lidar_memory.all_ids.end(), detector_srv.response.loop_id);
              int idx_loop_id = std::distance(lidar_memory.all_ids.begin(), it_loop_id);
              std::vector<int> Ids_verify;
              for (int i = 0; i < n_verify; ++i) {
                //i starts at 0
                int step_idx=i+1;
                if (idx_loop_id-step_idx<0){
                  //First id is reached -> take only idx_loop_id+step_idx
                  Ids_verify.push_back(lidar_memory.all_ids[idx_loop_id+step_idx]);
                }
                else{
                  //Take idx_loop_id+step_idx and idx_loop_id-step_idx
                  Ids_verify.push_back(lidar_memory.all_ids[idx_loop_id+step_idx]);
                  Ids_verify.push_back(lidar_memory.all_ids[idx_loop_id-step_idx]);
                }
              }
              //std::cout<<"Loop_id: "<<detector_srv.response.loop_id<<"\nIds_verify: ";
              //for (int i = 0; i < Ids_verify.size(); ++i) {std::cout<<Ids_verify[i]<<" ";}
              //std::cout<<std::endl;

              //Save all feature vectors in one 1D-Vector
              std::vector<double> features_1D_verify;
              for (int i = 0; i < Ids_verify.size(); ++i) {
                features_1D_verify.insert(features_1D_verify.end(),lidar_memory.features[Ids_verify[i]].begin(),lidar_memory.features[Ids_verify[i]].end());
              }

              //Create request for LiDAR_Loopdetector Service
              cmr_lidarloop::LiDAR_Loopdetector verify_srv;
              verify_srv.request.current_features = current_feature_vector;
              verify_srv.request.step=current_feature_vector.size();
              verify_srv.request.memory_features=features_1D_verify;
              verify_srv.request.memory_ids=Ids_verify;
              verify_srv.request.lengths_features=lengths_feature_vector;
              verify_srv.request.loop_probability_min=loop_probability_min;

              if (detector_client.call(verify_srv)){
                if (verify_srv.response.loop_id != -1){
                  ROS_INFO("Loop detected with neighborhood. Loop verification successful.");
                  loop_verified=true;
                }
                else{ROS_INFO("No loop detected with neighborhood. Loop verification failed.");}
              }
              else{ROS_ERROR("Failed to call service LiDAR_Loopdetector for loop verification.");}
            }
            else{
              //Loop verification disabled (n_verify<1)
              loop_verified=true;
            }
          }

          if(loop_verified){
            //Compute transformation between nodes -> Registration with action server
            //ROS_INFO("Cloud size before voxel filtering %d", cloud->size());
            //Current id -> cloud processing
            voxel_grid_filter_cloud(cloud,leafsize);
            //ROS_INFO("Cloud size before trimming %d", cloud->size());
            trim_cloud_z(cloud,z_limit,sky_direction);
            //ROS_INFO("Cloud size before intensity filtering %d", cloud->size());
            intensity_filter(cloud,i_limit);
            //ROS_INFO("Cloud size before range trimming %d", cloud->size());
            trim_cloud_r(cloud,r_limit);
            //ROS_INFO("Cloud size before random downsampling %d", cloud->size());
            random_downsampling(cloud,n_max_points);

            //Loop closure id
            rtabmap_ros::GetNodeData getNodeDataSrv;
            getNodeDataSrv.request.ids.push_back(detector_srv.response.loop_id);
            getNodeDataSrv.request.images = false;
            getNodeDataSrv.request.scan = true;
            getNodeDataSrv.request.user_data = false;
            getNodeDataSrv.request.grid = false;
            if(!ros::service::call("/rtabmap/get_node_data", getNodeDataSrv))
            {
              ROS_ERROR("Can't call \"/rtabmap/get_node_data\" service!");
            }
            else if(getNodeDataSrv.response.data.size() == 1)
            {
              rtabmap::Signature s = rtabmap_ros::nodeDataFromROS(getNodeDataSrv.response.data[0]);
              rtabmap::LaserScan scan;
              s.sensorData().uncompressDataConst(0, 0, &scan);
              pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud = rtabmap::util3d::laserScanToPointCloudI(scan, scan.localTransform());

              //Cloud processing
              voxel_grid_filter_cloud(target_cloud,leafsize);
              trim_cloud_z(target_cloud,z_limit,sky_direction);
              intensity_filter(target_cloud,i_limit);
              trim_cloud_r(target_cloud,r_limit);
              random_downsampling(target_cloud,n_max_points);

              //initial guess with robot odometry
              /*Eigen::Matrix4f init_guess, T_0_current, T_0_loopid;
              T_0_current = current_transformation;
              T_0_loopid = lidar_memory.transformations[srv.response.loop_id];
              init_guess=T_0_loopid.inverse()*T_0_current;*/

              //Check number of points in clouds
              if(cloud->size()>n_min_points && target_cloud->size()>n_min_points){

                //Transform to pointcloud2
                sensor_msgs::PointCloud2 target_cloud_msg;
                pcl::toROSMsg(*target_cloud.get(),target_cloud_msg);
                sensor_msgs::PointCloud2 current_cloud_msg;
                pcl::toROSMsg(*cloud.get(),current_cloud_msg);

                //create a goal for the action
                cmr_lidarloop::LiDAR_RegistrationGoal registration_goal;
                registration_goal.target_cloud=target_cloud_msg;
                registration_goal.current_cloud=current_cloud_msg;
                registration_goal.id=id;
                registration_goal.loop_id=detector_srv.response.loop_id;
                registration_goal.path_clouds=path_clouds;
                registration_goal.sky_direction=sky_direction;

                //send the goal, if the action server isn't busy with another goal
                if(registration_ac->getState()==actionlib::SimpleClientGoalState::LOST or
                   registration_ac->getState()==actionlib::SimpleClientGoalState::SUCCEEDED){
                  registration_ac->sendGoal(registration_goal);
                  result_processed = false;
                  ROS_INFO("Goal just sent to action server to compute transformation between "
                           "%d and %d",id,detector_srv.response.loop_id);
                }
              }
              else{
                ROS_INFO("Not enough points in clouds (%d and %d). Minimum number of points is %d. Pair rejected.",
                         cloud->size(),target_cloud->size(),n_min_points);
              }
            }
          }
        }
      }
      else{ROS_ERROR("Failed to call service LiDAR_Loopdetector");}
    }
    else{ROS_INFO("All Nodes in STM.");}

    if(MappingMode)
    {
      //Save current data in lidar memory
      lidar_memory.features[id]=current_feature_vector;
      lidar_memory.all_ids.push_back(id);
      lidar_memory.xyz[id]=current_xyz;
      lidar_memory.transformations[id]=current_transformation;

      //Replace deprecated poses with optimized ones from RTAB-Map
      std::map<int, rtabmap::Transform>::iterator poses_it=poses.begin();
      while (poses_it!=poses.end()) {
        int temp_id=poses_it->first;
        std::vector<double> temp_xyz={poses_it->second.o14(),poses_it->second.o24(),poses_it->second.o34()};
        Eigen::Matrix4f temp_transformation;
        temp_transformation <<poses_it->second.r11(),poses_it->second.r12(),poses_it->second.r13(),poses_it->second.o14(),
            poses_it->second.r21(),poses_it->second.r22(),poses_it->second.r23(),poses_it->second.o24(),
            poses_it->second.r31(),poses_it->second.r32(),poses_it->second.r33(),poses_it->second.o34(),
            0,0,0,1;
        //Update xyz and transformation
        lidar_memory.xyz[temp_id]=temp_xyz;
        lidar_memory.transformations[temp_id]=temp_transformation;
        poses_it++;
      }
    }

    id_old=id;

    auto finish_hr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_hr = finish_hr - start_hr;
    ROS_DEBUG("cmr_lidarloop took %.1lf ms.",duration_hr.count()*1000);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cmr_lidarloop");

  ULogger::setType(ULogger::kTypeConsole);
  ULogger::setLevel(ULogger::kWarning);

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  //Read params (cfg/cmr_lidarloop_params.yaml)
  read_params(nh,loop_probability_min,R_min,r_max,
              scan_topic_name,z_limit,sky_direction,leafsize,
              n_verify,odom_topic_name,n_max_nodes,
              alpha_thres, n_ms_verify, R_ms_verify,
              n_ms_start, t_max, i_limit, path_clouds,
              r_limit, n_max_points, n_min_points,
              min_inliers, beta, throttle_dur);

  detector_client = nh.serviceClient<cmr_lidarloop::LiDAR_Loopdetector>("LiDAR_Loopdetector");
  //Waiting for LiDAR_Loopdetector server to start
  detector_client.waitForExistence();

  //pnh.param("localization", !MappingMode, !MappingMode);

  //service to add link
  addLinkSrv = nh.serviceClient<rtabmap_ros::AddLink>("/rtabmap/add_link");

  //action client declaration (use_sim_time has to be false)
  if(nh.hasParam("/use_sim_time")){nh.setParam("/use_sim_time", false);}
  registration_ac = new actionlib::SimpleActionClient<cmr_lidarloop::LiDAR_RegistrationAction>("LiDAR_Registration", true);
  ROS_INFO("Waiting for LiDAR_Registration action server to start...");
  registration_ac->waitForServer();
  ROS_INFO("LiDAR_Registration action server started.");

  //Odometry subscription
  ros::Subscriber odomSub;
  odomSub = nh.subscribe(odom_topic_name, 1, odomCallback);

  //Service to check RTAB-Map availability
  ros::ServiceClient srvClient;
  srvClient = nh.serviceClient<roscpp::SetLoggerLevel>("/rtabmap/rtabmap/set_logger_level");

  //read STM size and RTAB-Maps mode
  while(true){
    if(nh.hasParam("/rtabmap/rtabmap/Mem/STMSize") && srvClient.exists()){

      nh.getParam("/rtabmap/rtabmap/Mem/STMSize", STM_size);
      std::string temp;
      nh.getParam("/rtabmap/rtabmap/Mem/IncrementalMemory", temp);
      if (temp=="true"){MappingMode=true;}
      else{MappingMode=false;}
      break;
    }
    else {
      ROS_INFO_THROTTLE(10.0, "Please start RTAB-Map.");
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }

  ROS_INFO("STM_size: %d, MappingMode: %s",STM_size, MappingMode?"true":"false");

  // Load LiDAR data from database (rtabmap.db)
  lidar_memory.read_rtabmap_db();
  //lidar_data test_memory;
  //test_memory=lidar_memory;

  //Scan descriptor subscription and publisher
  scanDescriptorPub = nh.advertise<rtabmap_ros::ScanDescriptor>("scan_descriptor", 1);
  ros::Subscriber scanSub;
  scanSub = nh.subscribe(scan_topic_name, 1, scanCallback);

  //RTAB-Map subscription
  message_filters::Subscriber<rtabmap_ros::Info> infoTopic;
  message_filters::Subscriber<rtabmap_ros::MapData> mapDataTopic;
  infoTopic.subscribe(nh, "/rtabmap/info", 1);
  mapDataTopic.subscribe(nh, "/rtabmap/mapData", 1);
  message_filters::Synchronizer<MyInfoMapSyncPolicy> infoMapSync(
        MyInfoMapSyncPolicy(10),
        mapDataTopic,
        infoTopic);
  infoMapSync.registerCallback(&mapDataCallback);
  ROS_INFO("Subscribed to %s and %s", mapDataTopic.getTopic().c_str(),
           infoTopic.getTopic().c_str());

  ros::spin();

  return 0;
}
