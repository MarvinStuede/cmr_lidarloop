#* @file   train_the_detector.py
#* @author Tim-Lukas Habich
#* @date   03/2020
#*
#* @brief  Standalone program to train the loop detector with
#*         raw_training_data.csv (or similar .csv passed as argument when calling the program)
#*         -> Detector according to the method of Granström - Learning to Close
#*	   the Loop from 3D Point Clouds

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
from time import clock
import pdb
import sys
from functions_for_detector import *

##### Inputs #####
#Distance in m, for which the classifier should detect a loop closure
loop_closure_distance=3
#Maximum number of points (number of rows in csv file)
#n_max_points<0 -> No points are deleted
n_max_points=-1
#Ratio of positive (loop) and negative (no loop) pairs
#pos_neg_ratio = loop_count/no_loop_count
pos_neg_ratio=1.0
#Desired number of trained detectors
#The training set of every detector varies because of the random selection of pairs
n_detectors=1
#Create .csv with data for AdaBoost
create_AdaBoost_csv=True
#Execute repetitive k-fold cross validation
execute_cv=True
#Number of folds for k-fold cross validation
n_folds=10
#Number of repeated k-fold cross validations
n_cross_validations=5
#ROC-Curve: minimum loop probability at start
loop_probability_start=0.5
#ROC-Curve: Increment to increase minimum loop probability
loop_probability_inc=0.01
#Name of .csv file, which contains raw_detector_data (xyz + LiDAR features)
if len(sys.argv)>1:
    name_csv=sys.argv[1]
else:
    name_csv="raw_training_data.csv"

##### Main #####
#Create training data for every detector and train the classifier
#If desired, execute repetitive k-fold cross validation also
ROC={"n_max_points":n_max_points}
for i_detector in np.arange(n_detectors):
    raw_data, lengths, names = get_raw_data_from_csv(name_csv)
    raw_data = delete_randomly_raw_data(raw_data,n_max_points)
    data, indizes_his = split_raw_data(raw_data,lengths)
    plot_x_y_map(data, "xy_map_"+str(np.size(data["xyz"],0))+"_nodes.pdf")

    #Creation of training data: each node is compared with each other
    train_data, loop_count, no_loop_count=compare_all_nodes(raw_data,data, indizes_his, loop_closure_distance, pos_neg_ratio)
    print("\nTraining data consisting of "+str(np.size(train_data["y"]))+" pairs created.")
    print("Among them are "+str(loop_count)+" loop-pairs and "+str(no_loop_count)+" no-loop-pairs (user defined positive negative ratio: "+str(pos_neg_ratio)+").\nThe defined loop closure distance is "+str(loop_closure_distance)+"m.")

    #Create .csv with data for AdaBoost
    if create_AdaBoost_csv:
        print_data_for_AdaBoost_in_csv(train_data,i_detector)

    #Training of the classifier
    classifier=train_AdaBoost(train_data["x"],train_data["y"][:,0])

    #Prediction time
    t1=clock()
    classifier.predict_proba(np.array(train_data["x"][0],ndmin=2))[0,1]
    t2=clock()
    print("AdaBoost Classifier trained.")
    print("Time for Prediction: "+str(round((t2-t1)*10**3,1))+"ms.")

    #Save classifier
    if n_detectors==1:
        with open('LiDAR_Loopdetector.pickle', 'wb') as f:
            pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)
        print("LiDAR Loop Detector saved as LiDAR_Loopdetector.pickle.")
    else:
        with open('LiDAR_Loopdetector'+str(i_detector)+'.pickle', 'wb') as f:
            pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)
        print("LiDAR Loop Detector saved as LiDAR_Loopdetector"+str(i_detector)+".pickle.")

    #k-fold cross validations
    if execute_cv:
        print(str(n_folds)+"-fold cross validation is repeated "+str(n_cross_validations)+" times. During each one the folds are different.")
        #Minimum value for the loop probability, to accept a loop
        #Iterate until false alarm < 1 % -> ROC curve
        loop_probability_min=loop_probability_start
        current_mean_fa=1
        while current_mean_fa*100>1:
            for i_cv in np.arange(n_cross_validations):
                #Print progress
                sys.stdout.write("\rCross validation progress: "+str(i_cv+1)+"/"+str(n_cross_validations)+" started")
                sys.stdout.flush()
                temp_rates=do_k_fold_cv(train_data,n_folds,loop_probability_min)
                #Rates dictionary for i_detector
                if i_cv==0:
                    #Initialization
                    rates_i_detector={"d_rates":np.array(temp_rates["d_rates"]),
                                      "fa_rates":np.array(temp_rates["fa_rates"])}
                else:
                    #Append
                    rates_i_detector["d_rates"]=np.hstack([rates_i_detector["d_rates"],np.array(temp_rates["d_rates"])])
                    rates_i_detector["fa_rates"]=np.hstack([rates_i_detector["fa_rates"],np.array(temp_rates["fa_rates"])])

            #Calculate mean and standard deviation of d_rates and fa_rates for i_detector
            current_mean_d=np.mean(rates_i_detector["d_rates"])
            current_mean_fa=np.mean(rates_i_detector["fa_rates"])
            current_std_d=np.std(rates_i_detector["d_rates"])
            current_std_fa=np.std(rates_i_detector["fa_rates"])
            #Print current iteration step
            print(" -> ROC curve: D = "+str(round(current_mean_d*100,1))+"% at FA = "+str(round(current_mean_fa*100,1))+"% (loop_probability_min = "+str(loop_probability_min)+")")
            #Save data
            if loop_probability_min==loop_probability_start:
                #First iteration -> Initalization
                ROC["D_"+str(i_detector)]=np.array(current_mean_d)
                ROC["FA_"+str(i_detector)]=np.array(current_mean_fa)
                ROC["loop_prob_min_"+str(i_detector)]=np.array(loop_probability_min)
            else:
                #Append
                ROC["D_"+str(i_detector)]=np.hstack([ROC["D_"+str(i_detector)],np.array(current_mean_d)])
                ROC["FA_"+str(i_detector)]=np.hstack([ROC["FA_"+str(i_detector)],np.array(current_mean_fa)])
                ROC["loop_prob_min_"+str(i_detector)]=np.hstack([ROC["loop_prob_min_"+str(i_detector)],np.array(loop_probability_min)])
            loop_probability_min+=loop_probability_inc
        plot_ROC_curve(ROC,i_detector)
        #Save mean and standard deviation at FA < 1%!!!
        if i_detector==0:
            #Initalization
            cv_results={"n_detectors": n_detectors,
                        "mean_d": np.array([current_mean_d]),
                        "mean_fa": np.array([current_mean_fa]),
                        "std_d": np.array([current_std_d]),
                        "std_fa": np.array([current_std_fa])}
        else:
            cv_results["mean_d"]=np.hstack([cv_results["mean_d"],np.array([current_mean_d])])
            cv_results["mean_fa"]=np.hstack([cv_results["mean_fa"],np.array([current_mean_fa])])
            cv_results["std_d"]=np.hstack([cv_results["std_d"],np.array([current_std_d])])
            cv_results["std_fa"]=np.hstack([cv_results["std_fa"],np.array([current_std_fa])])
        #pdb.set_trace()
if execute_cv:
    plot_cv_results(cv_results)
print("\n")
