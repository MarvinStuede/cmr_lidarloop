#!/usr/bin/env python
"""

/**
 * @file   LiDAR_Loopdetector_server.py
 * @author Tim-Lukas Habich, Marvin Stuede
 * @date   03/2020
 *
 * @brief  ROS Service to detect a loop closure with trained classifier
 *         (detector/training/LiDAR_Loopdetector.pickle)
 */"""

from cmr_lidarloop.srv import LiDAR_Loopdetector,LiDAR_LoopdetectorResponse
import rospy
import pickle
import numpy as np
import rospkg
import sys

if sys.version_info.major == 3:
    assert sys.version_info.minor >= 3, "Must use at least Python 3.3"
    from time import process_time as get_time
else:
    from time import clock as get_time


def handle_LiDAR_Loopdetector(req):
    t1=get_time()
    #Read request
    current_features=np.array(req.current_features,ndmin=2)
    step=req.step
    memory_features=np.array(req.memory_features,ndmin=2)
    memory_ids=np.array(req.memory_ids)
    lengths=np.array(req.lengths_features)
    loop_probability=req.loop_probability_min
    #print("Ids within Radius R: "+str(memory_ids))
    max_probability=0
    #Convert memory_features vector to matrix
    memory_features.shape=(-1,step)

    #Save features of type 1 and features of type 2 (range histograms)
    #Features of type 1
    data={"f_type_1": memory_features[:,0:lengths[0]]}
    current_data={"f_type_1": current_features[:,0:lengths[0]]}
    #Features of type 2
    data["f_type_2"]=memory_features[:,lengths[0]:]
    current_data["f_type_2"]=current_features[:,lengths[0]:]

    #Split f_type_2 into each range histogram
    lengths_his=lengths[1:]
    indizes_his=[]
    for i in np.arange(len(lengths_his)):
        indizes_his.append(sum(lengths_his[:i+1]))
    for i in np.arange(len(indizes_his)):
        if i==0:
            data[i+1]=data["f_type_2"][:,0:indizes_his[i]]
            current_data[i+1]=current_data["f_type_2"][:,0:indizes_his[i]]
        else:
            data[i+1]=data["f_type_2"][:,indizes_his[i-1]:indizes_his[i]]
            current_data[i+1]=current_data["f_type_2"][:,indizes_his[i-1]:indizes_his[i]]

    #Iterate through ids, to find a loop closure with highest probability
    #Compare current node with each node in memory
    #Features of type 1: subtraction of the respective vectors + absolute value of each entry
    #Features of type 2: comparison of the respective range histograms -> correlation coefficient

    #Initilization of loop_id (-1 -> no loop found)
    loop_id=-1
    count=0
    temp_probability=0
    for i in np.arange(memory_ids.size-1):
        count+=1
        #tdebug=process_time()
        #print(str(round((tdebug-t1)*10**3,1))+"ms ")

        #Calculation of feature vector (input for classifier)
        #Features of type 1
        try:
            features1=np.array(current_data["f_type_1"]-data["f_type_1"][i],ndmin=2)
        except Exception:
            a=0

        features1=np.absolute(features1)

        #Features of type 2
        for j in np.arange(len(indizes_his)):
            current_histogram=np.array(current_data[j+1],ndmin=2)
            histogram_i=np.array(data[j+1][i],ndmin=2)
            corr_coef=np.corrcoef(current_histogram,histogram_i)[0,1]
            if j==0:
                #Initilization of features2
                features2=np.array(corr_coef,ndmin=2)
            else:
                features2=np.vstack([features2,np.array(corr_coef,ndmin=2)])
        features2=np.transpose(features2)

        #Merge features1 & features2 to features
        features=np.hstack([features1,features2])

        #print("Prediction: "+str(detector.predict(features))+"\nProbabilities: "+str(detector.predict_proba(features)))

        #detector.classes_ -> array([0, 1])
        temp_probability=detector.predict_proba(features)[0,1]
        if temp_probability>loop_probability:
            loop_id=memory_ids[i]
            loop_probability=temp_probability
    if temp_probability > max_probability:
        max_probability = temp_probability

    if loop_id!=-1:
        rospy.logdebug("Loop closure with id "+str(loop_id)+" detected (probability: "+str(round(loop_probability*100,2))+" %)")
    else:
        rospy.logdebug("No loop detected. The highest probability was "+str(round(max_probability*100,2))+" %")

    t2=get_time()
    rospy.logdebug("\tLoop search with "+str(count)+" pairs took "+str(round((t2-t1)*10**3,1))+"ms.")
    return LiDAR_LoopdetectorResponse(int(loop_id))

def LiDAR_Loopdetector_server():
    rospy.init_node('LiDAR_Loopdetector_server')
    s = rospy.Service('/cmr_lidarloop/LiDAR_Loopdetector', LiDAR_Loopdetector, handle_LiDAR_Loopdetector)
    rospy.loginfo("LiDAR Loop Detector ready.")
    rospy.spin()

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    path_cmr_lidarloop=rospack.get_path('cmr_lidarloop')
    detector_name = 'LiDAR_Loopdetector_python' + str(sys.version_info.major) + '.pickle'
    with open(path_cmr_lidarloop+'/detector/training/' + detector_name, 'rb') as f:
            detector = pickle.load(f, encoding='latin1')
    LiDAR_Loopdetector_server()
