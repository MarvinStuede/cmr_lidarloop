#* @file   test_the_detector.py
#* @author Tim-Lukas Habich
#* @date   05/2020
#*
#* @brief  Standalone program to test the loop detector with
#*         raw_test_data.csv (or similar .csv passed as argument when calling the program)
#*

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
from time import clock
import pdb
import sys
import rospkg
#Find functions_for_detector
rospack = rospkg.RosPack()
path_cmr_lidarloop=rospack.get_path('cmr_lidarloop')
sys.path.insert(1, path_cmr_lidarloop+"/detector/training")
from functions_for_detector import *

##### Inputs #####
#Distance in m, for which the classifier should detect a loop closure
loop_closure_distance=3
#Maximum number of points (number of rows in csv file)
#n_max_points<0 --> No points are deleted
n_max_points=-1
#Minimum value for the loop probability, to accept a loop
loop_probability_min=0.524
#Numbers of LiDAR loop detectors
#Attention: it's a numpy array -> more numbers possible
#For example: numbers_detector=np.array([1,2])
#-> the two classifiers LiDAR_Loopdetector1.pickle and LiDAR_Loopdetector2.pickle are tested
numbers_detector=np.array([0])
#Computation and plot of ROC curve -> starting at loop_prob_start with increment loop_prob_inc until FA<fa_goal
#if more detectors are tested, compute_ROC is automatically set to True for comparison at FA<fa_goal
compute_ROC=False
loop_prob_start=0.5
loop_prob_inc=0.0005
fa_goal=1.0/100.0

##### Main #####
if len(sys.argv)>1:
    name_csv=sys.argv[1]
else:
    name_csv="raw_test_data.csv"

#Comparison of detectors at FA<fa_goal
if np.size(numbers_detector)>1:
    compute_ROC=True

raw_data, lengths, names = get_raw_data_from_csv(name_csv)
raw_data = delete_randomly_raw_data(raw_data,n_max_points)
data, indizes_his = split_raw_data(raw_data,lengths)
plot_x_y_map(data, "test_xy.pdf")

#Test desired detectors
for i_detector in numbers_detector:
    print("\nTesting LiDAR_Loopdetector"+str(i_detector)+".pickle.")
    #Load LiDAR loop detector
    with open('../training/LiDAR_Loopdetector'+str(i_detector)+'.pickle', 'rb') as f:
            detector = pickle.load(f)

    #Test the detector: classification matrix & distance matrix & probability matrix
    validation=compute_matrices(raw_data,data,indizes_his,loop_closure_distance,detector,loop_probability_min)
    plot_validation(validation,i_detector)

    #ROC curve -> until FA<fa_goal
    if compute_ROC:
        print("Computation and plot of ROC curve.")
        ROC={"detector_loop_prob_min": loop_probability_min,
             "detector_D": validation["detection_rate"],
             "detector_FA": validation["false_alarm_rate"]}
        temp_fa=1
        loop_prob_temp=loop_prob_start
        while temp_fa>fa_goal:
            temp_validation=compute_matrices(raw_data,data,indizes_his,loop_closure_distance,detector,loop_prob_temp,print_output=False)
            if loop_prob_temp==loop_prob_start:
                #Inizialization
                ROC["loop_prob_min"]=np.array(loop_prob_temp)
                ROC["D"]=np.array(temp_validation["detection_rate"])
                ROC["FA"]=np.array(temp_validation["false_alarm_rate"])
            else:
                #Append
                ROC["loop_prob_min"]=np.hstack([ROC["loop_prob_min"],np.array(loop_prob_temp)])
                ROC["D"]=np.hstack([ROC["D"],np.array(temp_validation["detection_rate"])])
                ROC["FA"]=np.hstack([ROC["FA"],np.array(temp_validation["false_alarm_rate"])])
            print("-> ROC curve: D = "+str(round(temp_validation["detection_rate"]*100,1))+"% at FA = "+str(round(temp_validation["false_alarm_rate"]*100,1))+"% (loop_probability_min = "+str(loop_prob_temp)+")")
            loop_prob_temp+=loop_prob_inc
            temp_fa=temp_validation["false_alarm_rate"]
        plot_ROC_with_matrices(ROC,validation,i_detector,plot_titles=True)

    #Save data for comparison
    if np.size(numbers_detector)>1:
        #Save result
        if i_detector==numbers_detector[0]:
            #Inizialization
            detector_comparison={"numbers_detector":numbers_detector,
                                 "D_0":ROC["D"][-1],
                                 "FA_0":ROC["FA"][-1]}
        else:
            #Append
            detector_comparison["D_0"]=np.hstack([detector_comparison["D_0"],np.array(ROC["D"][-1])])
            detector_comparison["FA_0"]=np.hstack([detector_comparison["FA_0"],np.array(ROC["FA"][-1])])

#Plot performances of all detectors at FA<fa_goal!
if np.size(numbers_detector)>1:
    plot_detector_comparison(detector_comparison,fa_goal)


