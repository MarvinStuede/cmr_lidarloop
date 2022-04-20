#!/usr/bin/env python
"""
@file   functions_for_detector.py
@author Tim-Lukas Habich
@date   05/2020

@brief  Functions for detector scripts
"""
import numpy as np
import csv
import pdb
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pickle

#Colors
imesblau   = [0./255.,80./255.,155./255.]
imesorange = [231./255.,123./255.,41./255.]
imesgruen  = [200./255.,211./255.,23./255.]

#Functions
def reset_plot_settings():
    #Plot options
    mpl.rcParams.update({'font.size': 25})
    mpl.rcParams.update({'lines.linewidth': 6})
    mpl.rcParams.update({'axes.linewidth': 1.0})
    mpl.rcParams.update({'font.style': 'normal'})
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams.update({'lines.markersize': 8.0})
    mpl.rcParams.update({'xtick.major.size': 6.0})
    mpl.rcParams.update({'xtick.minor.size': 6.0})
    mpl.rcParams.update({'xtick.major.width': 4.0})
    mpl.rcParams.update({'xtick.minor.width': 4.0})
    mpl.rcParams.update({'ytick.major.size': 6.0})
    mpl.rcParams.update({'ytick.minor.size': 6.0})
    mpl.rcParams.update({'ytick.major.width': 4.0})
    mpl.rcParams.update({'ytick.minor.width': 4.0})
    mpl.rcParams.update({'xtick.major.pad': 12})
    mpl.rcParams.update({'ytick.major.pad': 12})
    mpl.rcParams.update({'grid.linestyle': '-'})
    mpl.rcParams.update({'grid.linewidth': 1.0})
    mpl.rcParams.update({'ytick.direction':'in'})
    mpl.rcParams.update({'xtick.direction':'in'})
    mpl.rcParams.update({'legend.fontsize':25})
    rc('font',**{'family':'serif','serif':['Times']})
    return

# reset_plot_settings()

def get_raw_data_from_csv(name_csv):
    with open(name_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #Lengths of feature vector --> number of features of type 1 and length of all range histograms
                lengths=row[1:len(row)]
                #Convert to int
                for i in np.arange(len(lengths)):
                    lengths[i]=int(lengths[i])
                line_count += 1
            elif line_count==1:
                #Name of each column
                names=row
                line_count += 1
            elif line_count==2:
                #Convert to float
                for i in np.arange(len(row)):
                    row[i]=float(row[i])
                #Initialize raw data
                raw_data=np.array(row,ndmin=2)
                line_count+=1
            else:
                #Convert to float
                for i in np.arange(len(row)):
                    row[i]=float(row[i])
                raw_data=np.vstack([raw_data,np.array(row,ndmin=2)])
                line_count += 1
    print("\n")
    return raw_data, lengths, names

def get_train_data_from_csv(name_csv):
    with open(name_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #Header
                line_count += 1
            elif line_count==1:
                #Convert to float
                for i in np.arange(len(row)):
                    row[i]=float(row[i])
                #Initialize train data
                train_data={"x": np.array(row[0:-1],ndmin=2),
                            "y": np.array(row[-1],ndmin=2)}
                line_count+=1
            else:
                #Convert to float
                for i in np.arange(len(row)):
                    row[i]=float(row[i])
                train_data["x"]=np.vstack([train_data["x"],np.array(row[0:-1],ndmin=2)])
                train_data["y"]=np.vstack([train_data["y"],np.array(row[-1],ndmin=2)])
                line_count += 1
    print("\n")
    return train_data

def delete_randomly_raw_data(raw_data,n_max_points):
    if n_max_points<0:
        return raw_data
    #Delete randomly points if n_points>n_max_points
    n_points=np.size(raw_data,0)
    if n_points > n_max_points:
        n_del=n_points-n_max_points
        print("File contains "+str(n_points)+" data rows. Deleting "+str(n_del)+" points randomly, to achieve goal ("+str(n_max_points)+" points).")
        for i in np.arange(n_del):
            i_del=random.randrange(np.size(raw_data,0))
            raw_data=np.delete(raw_data, i_del, 0)
    return raw_data

def split_raw_data(raw_data,lengths):
    #Split raw_data into xyz-poses and features
    data={"xyz":    raw_data[:,0:3]}
    features=raw_data[:,3:]

    #Save features of type 1 and features of type 2 (range histograms) in data-dict
    #Features of type 1
    data["f_type_1"]=features[:,0:lengths[0]]
    #Features of type 2
    data["f_type_2"]=features[:,lengths[0]:]
    #Split f_type_2 into each range histogram
    lengths_his=lengths[1:]
    indizes_his=[]
    for i in np.arange(len(lengths_his)):
        indizes_his.append(sum(lengths_his[:i+1]))
    for i in np.arange(len(indizes_his)):
        if i==0:
            data[i+1]=data["f_type_2"][:,0:indizes_his[i]]
        else:
            data[i+1]=data["f_type_2"][:,indizes_his[i-1]:indizes_his[i]]
    return data, indizes_his

def plot_x_y_map(data, img_name):
    #Plot x-y map of data
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_aspect('equal')
    plt.plot(data["xyz"][:,0], data["xyz"][:,1],mec='black',mew=0.3, color=imesorange,marker='o',linestyle='',alpha=0.5, markersize=6)
    box = ax.get_position()
    plt.grid()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height*0.45])
    plt.savefig(os.getcwd()+"/plots/"+img_name, dpi=200,bbox_inches='tight')
    plt.close()
    print("Data loaded. Map (x-y) with "+str(np.size(data["xyz"],0))+" nodes plotted in plots/"+img_name+".")
    return

def calc_input_classifier(data,first,sec, indizes_his):
    #Calculation of input for classifier

    #Features of type 1: subtraction of the respective vectors + absolute value of each entry
    features1=np.array(data["f_type_1"][first]-data["f_type_1"][sec],ndmin=2)
    features1=np.absolute(features1)

    #Features of type 2: comparison of the respective range histograms -> correlation coefficient
    for j in np.arange(len(indizes_his)):
        histogram_first=np.array(data[j+1][first],ndmin=2)
        histogram_sec=np.array(data[j+1][sec],ndmin=2)
        corr_coef=np.corrcoef(histogram_first,histogram_sec)[0,1]
        if j==0:
            #Initilization of features2
            features2=np.array(corr_coef,ndmin=2)
        else:
            features2=np.vstack([features2,np.array(corr_coef,ndmin=2)])
    features2=np.transpose(features2)

    #Merge features1 & features2 to features
    features=np.hstack([features1,features2])
    return features

def compare_all_nodes(raw_data,data, indizes_his, loop_closure_distance,pos_neg_ratio):
    #Each node is compared with each other (also with himself!)
    combinations=[]
    loop_count=0
    no_loop_count=0
    for i in np.arange(np.size(raw_data,0)):
        count=i
        while count!=np.size(raw_data,0):
            combinations.append([i,count])
            count+=1
    print(str(len(combinations))+" possible pairs.")

    #Iterate through all possible pairs in a random order
    random_order=np.arange(len(combinations))
    np.random.shuffle(random_order)
    count_combinations=0
    for i in random_order:
        #Print progress
        sys.stdout.write("\rNode comparison progress: "+str(round(float(count_combinations)/float(len(combinations))*100,0))+"%")
        sys.stdout.flush()

        #Indices of combination
        first=combinations[i][0]
        sec=combinations[i][1]
        #Distance between nodes
        distance=np.linalg.norm(data["xyz"][first]-data["xyz"][sec])

        features=calc_input_classifier(data,first,sec,indizes_his)

        #Save combination in detector_data dictionary
        if count_combinations==0:
            #Initialization

            #Input classifier: x
            detector_data={"x":    features}

            #Output classifier: y
            if distance < loop_closure_distance:
                #Loop closure --> y=1
                detector_data["y"]=np.array(1,ndmin=2)
                loop_count+=1
            else:
                #No loop closure --> y=0
                detector_data["y"]=np.array(0,ndmin=2)
                no_loop_count+=1
        else:
            #Add new data point
            #Consideration of ratio: pos_neg_ratio = loop_count/no_loop_count

            #Output classifier: y
            if distance < loop_closure_distance:
                #Loop closure --> y=1
                detector_data["y"]=np.vstack([detector_data["y"],np.array(1,ndmin=2)])
                loop_count+=1
                #Input classifier: x
                detector_data["x"]=np.vstack([detector_data["x"],features])
            else:
                #No loop closure --> y=0
                if pos_neg_ratio*no_loop_count < loop_count:
                    detector_data["y"]=np.vstack([detector_data["y"],np.array(0,ndmin=2)])
                    no_loop_count+=1
                    #Input classifier: x
                    detector_data["x"]=np.vstack([detector_data["x"],features])
        count_combinations+=1
    return detector_data, loop_count, no_loop_count

def compute_matrices(raw_data,data, indizes_his, loop_closure_distance, detector, loop_probability_min,print_output=True):
    #Each node is compared with each other to compute classification matrix, distance matrix & probability matrix
    #A loop is only accepted if: loop_probability>loop_probability_min

    #positive data pairs
    loop_count=0
    #negative data pairs
    no_loop_count=0
    #positive data pairs classified as positive
    true_positive=0
    #negative data pairs classified as positive
    false_positive=0

    #Possible combinations: number_nodes*number_nodes (Matrix)
    combinations=[]
    for col in np.arange(np.size(raw_data,0)):
        for row in np.arange(np.size(raw_data,0)):
            combinations.append([row,col])
    if print_output:
        print("\n"+str(len(combinations))+" pairs for matrix.")

    #Matrix initialization
    classification_mat=np.zeros([np.size(raw_data,0),np.size(raw_data,0)])
    distance_mat=np.zeros([np.size(raw_data,0),np.size(raw_data,0)])
    proba_mat=np.zeros([np.size(raw_data,0),np.size(raw_data,0)])

    #Fill matrices
    for i in np.arange(len(combinations)):
	#Print progress
        if print_output:
            sys.stdout.write("\rMatrix computation progress: "+str(round(float(i)/float(len(combinations))*100,0))+"%")
            sys.stdout.flush()

        first=combinations[i][0]
        sec=combinations[i][1]
        #Distance between nodes
        distance=np.linalg.norm(data["xyz"][first]-data["xyz"][sec])

        features=calc_input_classifier(data,first,sec,indizes_his)

        #Calculate entry inside matrices
        if distance < loop_closure_distance:
            #Loop closure --> y=1
            distance_mat[first,sec]=1
            loop_count+=1
            #Prediction of detector
            temp_probability=detector.predict_proba(features)[0,1]
            proba_mat[first,sec]=temp_probability
            if temp_probability>loop_probability_min:
                #Loop detected
                classification_mat[first,sec]=1
                true_positive+=1
            else:
                #No loop detected
                classification_mat[first,sec]=0
        else:
            #No loop closure --> y=0
            distance_mat[first,sec]=0
            no_loop_count+=1
            #Prediction of detector
            temp_probability=detector.predict_proba(features)[0,1]
            proba_mat[first,sec]=temp_probability
            if temp_probability>loop_probability_min:
                #Loop detected
                classification_mat[first,sec]=1
                false_positive+=1
            else:
                #No loop detected
                classification_mat[first,sec]=0

    #Create validation dictionary
    validation={"classification_mat":   classification_mat,
                "distance_mat":         distance_mat,
                "proba_mat":            proba_mat,
                "loop_count":           loop_count,
                "true_positive":        true_positive,
                "no_loop_count":        no_loop_count,
                "false_positive":       false_positive,
                "detection_rate":       float(true_positive)/float(loop_count),
                "false_alarm_rate":     float(false_positive)/float(no_loop_count)}
    return validation

def plot_validation(validation,i_detector,plot_titles=True):
    mpl.rcParams.update({'font.size': 12})
    mpl.rcParams.update({'xtick.major.pad': 4})
    mpl.rcParams.update({'ytick.major.pad': 4})
    fig, (proba, classification, distance) = plt.subplots(1, 3,figsize=(13, 4.5),sharey=True)
    proba.matshow(validation["proba_mat"],cmap="viridis")
    #proba.tick_params(labelsize=2)
    classification.matshow(validation["classification_mat"],cmap="viridis")
    distance.matshow(validation["distance_mat"],cmap="viridis")
    if plot_titles:
        proba.set_title(r'probability matrix',pad=11)
        classification.set_title(r'classification matrix',pad=11)
        distance.set_title(r'distance matrix',pad=11)
        plt.suptitle(r'detection-rate $D='+str(round(validation["detection_rate"]*100,1))+'$\% and false alarm-rate $FA='+str(round(validation["false_alarm_rate"]*100,1))+'\%$')
    plt.savefig(os.getcwd()+"/plots/matrices"+str(i_detector)+".pdf", dpi=200,bbox_inches='tight')
    plt.close()
    reset_plot_settings()
    print("\nMatrices visualized in plots/matrices"+str(i_detector)+".pdf\n")

def print_data_for_AdaBoost_in_csv(data,i_detector):
    with open('data_for_AdaBoost_detector'+str(i_detector)+'.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #Header
        header=[]
        for i in np.arange(np.size(data["x"],1)):
            temp_str='x'+str(i+1)
            header.append(temp_str)
        header.append('y')
        writer.writerow(header)
        #AdaBoost data
        for i in np.arange(np.size(data["y"])):
            row=[]
            temp_x=data["x"][i,:]
            temp_y=data["y"][i]
            for j in np.arange(np.size(temp_x)):
                row.append(str(temp_x[j]))
            row.append(str(temp_y[0]))
            writer.writerow(row)
            #pdb.set_trace()

def train_AdaBoost(x_train,y_train):
    classifier = AdaBoostClassifier(algorithm='SAMME')
    classifier.fit(x_train,y_train)
    #Tune hyperparameters begin
    """pdb.set_trace()
    from sklearn.model_selection import GridSearchCV
    grid_params = {'n_estimators': [25,50,75,100],'learning_rate': [0.25,0.5,0.75,1]}
    grid_object = GridSearchCV(estimator = AdaBoostClassifier(algorithm='SAMME'), param_grid = grid_params, scoring = 'accuracy', cv = 3, n_jobs = -1)
    grid_object.fit(x_train, y_train)
    print(grid_object.best_params_)"""
    #Tune hyperparameters end
    return classifier

def test_AdaBoost(x_test,y_test,classifier,loop_probability_min):
    #positive data pairs
    loop_count=0
    #negative data pairs
    no_loop_count=0
    #positive data pairs classified as positive
    true_positive=0
    #negative data pairs classified as positive
    false_positive=0

    #Predict loop probabilities with AdaBoost
    loop_probabilities=classifier.predict_proba(x_test)[:,1]

    #Compare with y_test (ground truth)
    for i in np.arange(np.size(y_test)):
        if y_test[i,0]==1:
            #Loop -> positive data pair
            loop_count+=1
            if loop_probabilities[i]>loop_probability_min:
                #Loop predicted -> Prediction correct -> true positive
                true_positive+=1
        else:
            #No loop -> negative data pair
            no_loop_count+=1
            if loop_probabilities[i]>loop_probability_min:
                #Loop predicted -> Prediction wrong -> false positive
                false_positive+=1

    #Compute rates
    detection_rate=float(true_positive)/float(loop_count)
    false_alarm_rate=float(false_positive)/float(no_loop_count)
    return detection_rate, false_alarm_rate


def do_k_fold_cv(data, n_folds, loop_probability_min):
    rates={"d_rates":[],
           "fa_rates":[]}
    kf = KFold(n_splits=n_folds,shuffle=True)
    kf.get_n_splits(data["x"])
    for train_index, test_index in kf.split(data["x"]):
        x_train=data["x"][train_index,:]
        x_test=data["x"][test_index,:]
        y_train=data["y"][train_index]
        y_test=data["y"][test_index]

        classifier=train_AdaBoost(x_train,y_train[:,0])

        #Compute detection rate and false-alarm rate with given loop_probability_min
        d_rate, fa_rate=test_AdaBoost(x_test,y_test,classifier,loop_probability_min)

        #Save rates
        rates["d_rates"].append(d_rate)
        rates["fa_rates"].append(fa_rate)

    return rates

def plot_cv_results(cv_results):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlabel(r'Detector No.')
    ax.set_ylabel(r'$D$ at $FA<0.5\%$')
    ax.errorbar(np.arange(cv_results["n_detectors"]), cv_results["mean_d"], yerr=cv_results["std_d"],elinewidth=3, fmt='-o',color=imesorange,ecolor=imesblau,capsize=10, capthick=4)
    #ax.errorbar(cv_results["pos_neg_ratios"], cv_results["mean_fa"], yerr=cv_results["std_fa"], fmt='-o',color=imesblau,capsize=10, capthick=4,label=r'$FA$')
    #ax.legend()
    ax.set_ylim(0,1)
    plt.xticks(np.arange(cv_results["n_detectors"]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height*0.45])
    plt.savefig(os.getcwd()+"/plots/cv_results.pdf", dpi=200,bbox_inches='tight')
    plt.close()
    #Save cv_results
    with open(os.getcwd()+"/plots/cv_results.pickle", 'wb') as f:
        pickle.dump(cv_results, f, pickle.HIGHEST_PROTOCOL)
    return

def plot_ROC_curve(ROC,i_detector):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlabel(r'$FA$')
    ax.set_ylabel(r'$D$')
    ax.set_title(r'$D='+str(round(ROC["D_"+str(i_detector)][-1]*100,1))+'\%$ at $FA='+str(round(ROC["FA_"+str(i_detector)][-1]*100,1))+'\%$')
    ax.plot(ROC["FA_"+str(i_detector)], ROC["D_"+str(i_detector)], linestyle='',marker='o',color=imesorange)
    #ax.errorbar(cv_results["pos_neg_ratios"], cv_results["mean_fa"], yerr=cv_results["std_fa"], fmt='-o',color=imesblau,capsize=10, capthick=4,label=r'$FA$')
    #ax.legend()
    ax.set_ylim(0,1)
    plt.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height*0.45])
    plt.savefig(os.getcwd()+"/plots/ROC_"+str(i_detector)+".pdf", dpi=200,bbox_inches='tight')
    plt.close()

    #Save characteristics in .csv
    ROC_file = open(os.getcwd()+"/plots/ROC_"+str(i_detector)+".csv", 'w')
    with ROC_file:
        writer = csv.writer(ROC_file)
        #Header
        writer.writerow(["loop_probability_min","D","FA"])
        #ROC content
        for i in np.arange(np.size(ROC["D_"+str(i_detector)])):
            writer.writerow([round(ROC["loop_prob_min_"+str(i_detector)][i],3),round(ROC["D_"+str(i_detector)][i],3),round(ROC["FA_"+str(i_detector)][i],3)])
    return

def plot_ROC_with_matrices(ROC,validation,i_detector,plot_titles=True):
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams.update({'xtick.major.pad': 4})
    mpl.rcParams.update({'ytick.major.pad': 4})
    fig, (ROCax, classification, distance) = plt.subplots(1, 3,figsize=(13, 4.5))
    #ROC
    ROCax.set_xlabel(r'$FA$')
    ROCax.set_ylabel(r'$D$')
    ROCax.plot(ROC["FA"], ROC["D"], linestyle='-',linewidth=4,color=imesorange)
    ROCax.plot(ROC["detector_FA"],ROC["detector_D"],linestyle='',marker='*',markersize=12,color=imesblau)
    ROCax.set_ylim(0,1)
    ROCax.grid()
    ROCax.set_aspect(1./ROCax.get_data_ratio())
    #Matrices
    #Create colormap with imes colors
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    grey = np.array([178.0/255.0, 178.0/255.0, 178.0/255.0, 1])
    orange = np.array([imesorange[0], imesorange[1], imesorange[2], 1])
    newcolors[:20, :] = grey
    newcolors[236:, :] = orange
    imescmp = ListedColormap(newcolors)
    classification.matshow(validation["classification_mat"],cmap=imescmp)
    distance.matshow(validation["distance_mat"],cmap=imescmp)
    if plot_titles:
        ROCax.set_title(r'ROC curve',pad=11)
        classification.set_title(r'classification matrix',pad=11)
        distance.set_title(r'distance matrix',pad=11)
        plt.suptitle(r'detection-rate $D='+str(round(validation["detection_rate"]*100,1))+'$\% and false alarm-rate $FA='+str(round(validation["false_alarm_rate"]*100,1))+'\%$')
    plt.savefig(os.getcwd()+"/plots/ROC_with_matrices"+str(i_detector)+".pdf", dpi=200,bbox_inches='tight')
    plt.close()
    reset_plot_settings()
    print("\nMatrices with ROC curve visualized in plots/ROC_with_matrices"+str(i_detector)+".pdf\n")

    #Save characteristics in .csv
    ROC_file = open(os.getcwd()+"/plots/ROC_"+str(i_detector)+".csv", 'w')
    with ROC_file:
        writer = csv.writer(ROC_file)
        #Header
        writer.writerow(["loop_probability_min","D","FA"])
        #ROC content
        for i in np.arange(np.size(ROC["D"])):
            writer.writerow([round(ROC["loop_prob_min"][i],3),round(ROC["D"][i],3),round(ROC["FA"][i],3)])
    return

def plot_detector_comparison(detector_comparison,fa_goal):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlabel(r'Detector No.')
    ax.set_ylabel(r'$D$ at $FA<'+str(fa_goal*100)+'\%$')
    ax.plot(detector_comparison["numbers_detector"], detector_comparison["D_0"], linestyle='-',linewidth=4,color=imesorange)
    ax.plot(detector_comparison["numbers_detector"], detector_comparison["D_0"], linestyle='',marker='o',markersize=12,color=imesblau)
    ax.set_ylim(0,1)
    plt.grid()
    #plt.xticks(detector_comparison["numbers_detector"])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height*0.45])
    plt.savefig(os.getcwd()+"/plots/detector_comparison.pdf", dpi=200,bbox_inches='tight')
    plt.close()
    print("Detector comparison visualized in plots/detector_comparison.pdf\n")
    #Save detector_comparison
    with open(os.getcwd()+"/plots/detector_comparison.pickle", 'wb') as f:
        pickle.dump(detector_comparison, f, pickle.HIGHEST_PROTOCOL)
    return

def plot_data_template(x_data,y_data,x_label,y_label,name_plot,with_marker=True):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x_data, y_data, linestyle='-',linewidth=4,color=imesorange)
    plt.grid()
    if with_marker:
        ax.plot(x_data, y_data, linestyle='',marker='o',markersize=12,color=imesblau)
    #ax.set_ylim(0,1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height*0.45])
    plt.savefig(name_plot, dpi=200,bbox_inches='tight')
    plt.close()
    print("Data plotted in "+name_plot+".")
    return
