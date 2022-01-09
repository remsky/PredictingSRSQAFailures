import re   #regular expression string searching
import pandas as pd #dataframe, data management
import pydicom  #DICOM reader
import numpy as np #general purpose vectorized math library
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.metrics.scorer import make_scorer

from assorted_utilities import *
#from complexity metric_utilities import calc_beam_aperture_complexity

#example code:
 
score_names =  ['Aperture',"MLC02","MLC05","MLC1","MLC2","uniMLC","accMLC02","accMLC05"
                "accMLC1","accMLC2","uniaccMLC", "FieldX","FieldY","gantryvel",
                "gantryacc","areamet005""areamet05","areamet1","areamet30", "areamet50"]

# where scores are generated from a pydicom read of a RP.*.dcm file 
# e.g:

#   ds = pydicom.read_file("RP. * .dcm")
#   beam_num = 0
#   aperture_score = calc_beam_aperture_complexity(ds, beam_num)


# combinations to test
n_combos = 5
combos = list(itertools.combinations(score_names, n_combos))
    
scoring = make_scorer(custom_scorer,weight=1)
results = []

def is_percent_close(a, b, thresh=0.10):
    if a == b:
        return True
        
    if a == 0:
        # avoid zero division, outright bad models
        return False

    if (abs(b - a) / abs(a)) > 0.10:
        return False
    else:
        return True

for i in combos:
    
    #load and filter
    #features are calcualted metrics to explore 
    #labels are 0:fail 1:pass
    train_features, test_features, train_labels, test_labels = #read_pickle/excel("...") , # , # , # 
    train_features, test_features = train_features.filter(list(i)), test_features.filter(list(i))
    #scaling metrics
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    #initial hyperparam random search
    svc = SVC(class_weight="balanced",probability=True)
    param_dist = {"C" : np.linspace(.1,50,200),
                 "gamma" : np.linspace(.1,50,200)}

    n_folds = 3
    n_iter_search = 200
    random_search_svc = RandomizedSearchCV(svc,scoring=scoring,param_distributions=param_dist,
                                       n_iter=n_iter_search,n_jobs=-1, cv=n_folds, verbose=0)
    
    random_search_svc.fit(train_features, train_labels)
    best_params = random_search_svc.best_params_
    best_c , best_g = best_params['C'], best_params['gamma']

    
    # fine-tune hyperparam grid search
    svc = SVC(class_weight="balanced",probability=True) # fresh model
    param_dist_g = {"C" :      np.linspace(best_c*.9, best_c*1.1, 10),
                    "gamma" :  np.linspace(best_g*.9, best_g*1.1, 10),}
    
    grid_tune = GridSearchCV(svc, scoring=scoring, param_grid=param_dist_g, n_jobs=-1, cv=n_folds, verbose=0)
    grid_tune.fit(train_features, train_labels)
    best_model =  grid_tune.best_estimator_
 
    # generate prediction and ranking scores 
    y_test_true = test_labels
    y_test_pred = best_model.predict(test_features)
    y_train_true = train_labels
    y_train_pred = best_model.predict(train_features)
    # test set. binary labels: 0 is fail, 1 is pass
    te_recall_fail = recall_score(y_test_true,y_test_pred,average="binary",labels=[0,1],pos_label=0)
    te_recall_pass = recall_score(y_test_true,y_test_pred,average="binary",labels=[0,1],pos_label=1)
    te_precision_fail = precision_score(y_test_true,y_test_pred,average="binary",labels=[0,1],pos_label=0)
    te_precision_pass = recall_score(y_test_true,y_test_pred,average="binary",labels=[0,1],pos_label=1)   
    
    te_collect = [te_recall_fail,te_recall_pass,te_precision_fail,te_precision_pass]
    
    # train set
    tr_recall_fail = recall_score(y_train_true,y_train_pred,average="binary",labels=[0,1],pos_label=0)
    tr_recall_pass = recall_score(y_train_true,y_train_pred,average="binary",labels=[0,1],pos_label=1)
    tr_precision_fail = precision_score(y_train_true,y_train_pred,average="binary",labels=[0,1],pos_label=0)
    tr_precision_pass = recall_score(y_train_true,y_train_pred,average="binary",labels=[0,1],pos_label=1)
    
    tr_collect = [tr_recall_fail,tr_recall_pass,tr_precision_fail,tr_precision_pass]
    
    # reject overfit/unstable models
    for te_score, tr_score in zip(te_collect, tr_collect):
        if not is_percent_close(tr_score,te_score):
            results.append([i,0,0,0,0]) # rejected
            continue #skip further checks, move to next iteration of loop  
        else:
            results.append([i,te_recall_fail,te_recall_pass,te_precision_fail,te_precision_pass])
            


    
df_results = pd.DataFrame(results, columns =['Scores',"Recall Fail", "Recall Pass", "Precision Fail", "Precision Pass"] )
# sort as desired
df_results= df_results.sort_values(by=["Recall Fail", "Recall Pass", "Precision Fail", "Precision Pass", ],axis=0,ascending=False)
# display results 
df_results.head(10)

## for 2-d decision boundaries
## replace score_names with top 5 score via df_results
## repeat with n_combos = 2 o
## call assorted_utilities.make_2d_decision_boundary() on top results for n-dim 2 to visualize
