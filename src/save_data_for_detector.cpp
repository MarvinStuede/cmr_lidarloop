/**
 * @file   save_data_for_detector.cpp
 * @author Tim-Lukas Habich
 * @date   02/2020
 *
 * @brief  Node to save feature vectors of pointclouds + positions
 *         -> Training (detector/training) and test (detector/test)
 *         of detector in the next step
 */

#include "cmr_lidarloop/lidar_loopdetection.h"
#include <fstream>
#include <rtabmap_ros/ScanDescriptor.h>
#include <rtabmap/core/Compression.h>
#include <chrono>
#include <thread>

//##################Parameters##################
//#########cfg/cmr_lidarloop_params.yaml########
double loop_probability_min, R_min, r_max, z_limit, alpha_thres, R_ms_verify, r_limit, beta, throttle_dur;
std::string scan_topic_name, odom_topic_name, path_clouds;
float leafsize, t_max, i_limit;
int n_verify, n_max_nodes, n_ms_verify, n_ms_start, n_max_points, n_min_points, min_inliers, sky_direction;
//##############################################

typedef message_filters::sync_policies::ExactTime<rtabmap_ros::MapData, rtabmap_ros::Info> MyInfoMapSyncPolicy;

//Publisher of scan descriptor with corresponding scan
ros::Publisher scanDescriptorPub;

int id_old=-1;

class detector_data
{
public:
  detector_data();
  ~detector_data();
  // xyz + related feature vector
  std::map<int,std::vector<double>> xyz;
  std::map<int, std::vector<double>> features;
  std::vector<int> all_ids;
  std::vector<int> lengths;
};
detector_data::detector_data(){}
//save detector data in .csv if node is shutting down
detector_data::~detector_data()
{
  std::ofstream file;
  file.open ("raw_detector_data.csv");
  //print header
  //lengths of feature vector --> number of features of type 1 and length of all range histograms
  file << "Lengths of feature vector:,";
  for (int i=0;i<lengths.size()-1;i++) { file << lengths[i] << ",";}
  file << lengths[lengths.size()-1] << "\n";
  file << "x,y,z,";
  for (int i=0;i<features.begin()->second.size()-1;i++) {file << "f" << i+1 << ",";}
  file << "f" << features.begin()->second.size() << "\n";
  int temp_id;
  for (int i = 0; i < all_ids.size(); ++i)
  {
    temp_id=all_ids[i];
    //xyz
    file << xyz[temp_id][0] << "," << xyz[temp_id][1] << "," << xyz[temp_id][2] << ",";
    //feature vector
    for (int j=0;j<features.begin()->second.size()-1;j++) { file << features[temp_id][j] << ",";}
    file << features[temp_id][features[temp_id].size()-1] << "\n";
  }
  file.close();
  std::cout<<"Detector Data saved!"<<std::endl;
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

    //Publish scan descriptor for RTAB-Map
    rtabmap_ros::ScanDescriptor scanDescriptor;
    scanDescriptor.header = pointCloud2Msg->header;
    scanDescriptor.scan_cloud = *pointCloud2Msg;
    scanDescriptor.global_descriptor.type=0;
    scanDescriptor.global_descriptor.info=rtabmap::compressData(cv::Mat(1, lengths_feature_vector.size(), CV_32FC1, (void*)lengths_feature_vector.data()));
    scanDescriptor.global_descriptor.data=rtabmap::compressData(cv::Mat(1, current_feature_vector.size(), CV_64FC1, (void*)current_feature_vector.data()));
    scanDescriptorPub.publish(scanDescriptor);
}

detector_data data_for_detector;

void DataCallback(const rtabmap_ros::MapDataConstPtr & mapDataMsg, const rtabmap_ros::InfoConstPtr & infoMsg)
{
  ROS_INFO("Received map data!");

  rtabmap::Statistics stats;
  rtabmap_ros::infoFromROS(*infoMsg, stats);

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = rtabmap::util3d::laserScanToPointCloud(scan, scan.localTransform());

    //Get scan descriptor
    ROS_ASSERT(s.globalDescriptors().size() == 1); //Assuming we are using only one descriptor
    ROS_ASSERT(s.globalDescriptors()[0].type() == 0); // Assuming descriptor is the same type that we set
    ROS_ASSERT(s.globalDescriptors()[0].data().type() == CV_64FC1); // Make sure we get the same type that we set
    ROS_ASSERT(s.globalDescriptors()[0].info().type() == CV_32FC1); // Make sure we get the same type that we set
    std::vector<double> current_feature_vector(s.globalDescriptors()[0].data().total());
    std::vector<int> lengths_feature_vector(s.globalDescriptors()[0].info().total());
    memcpy(current_feature_vector.data(), s.globalDescriptors()[0].data().data, current_feature_vector.size()*sizeof(double));
    memcpy(lengths_feature_vector.data(), s.globalDescriptors()[0].info().data, lengths_feature_vector.size()*sizeof(int));

    std::vector<double> current_xyz={poses[id].o14(),poses[id].o24(),poses[id].o34()};

    //Save current data
    data_for_detector.xyz[id]=current_xyz;
    data_for_detector.features[id]=current_feature_vector;
    //Replace deprecated poses with optimized ones
    std::map<int, rtabmap::Transform>::iterator poses_it=poses.begin();
    while (poses_it!=poses.end()) {
      int temp_id=poses_it->first;
      std::vector<double> temp_xyz={poses_it->second.o14(),poses_it->second.o24(),poses_it->second.o34()};
      //Update xyz
      data_for_detector.xyz[temp_id]=temp_xyz;
      poses_it++;
    }

    data_for_detector.all_ids.push_back(id);
    data_for_detector.lengths=lengths_feature_vector;

    id_old=id;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "save_data_for_detector");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  //Read cfg/cmr_lidarloop_params.yaml
  read_params(nh,loop_probability_min,R_min,r_max,
              scan_topic_name,z_limit,sky_direction,leafsize,
              n_verify,odom_topic_name,n_max_nodes,
              alpha_thres, n_ms_verify, R_ms_verify,
              n_ms_start, t_max, i_limit, path_clouds,
              r_limit, n_max_points, n_min_points,
              min_inliers, beta, throttle_dur);

  //Wait for RTAB-Map
  while(true){
    if(nh.hasParam("/rtabmap/Mem/STMSize")){
      break;
    }
    else {
      ROS_INFO("Please start RTAB-Map.");
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }

  //Scan subscription and publisher
  scanDescriptorPub = nh.advertise<rtabmap_ros::ScanDescriptor>("/cmr_lidarloop/scan_descriptor", 1);
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
  infoMapSync.registerCallback(&DataCallback);
  ROS_INFO("Subscribed to %s and %s", mapDataTopic.getTopic().c_str(), infoTopic.getTopic().c_str());

  ros::spin();
  return 0;
}
