import pandas as pd 
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

from SelezioneParametri import svm_param_selection, decision_tree_param_selection,random_forest_param_selection, mlp_param_selection, k_neighbors_classifier_param_selection, XGboost_param_selection

SCALER_FILE = "chosen_scaler.sav"
FEATURES_SELECTION_FILE = 'chosen_features_selector.sav' 
BEST_MODEL_FILE = "chosen_model.sav"
SUPPORTED_SKLEARN_SCALER = {"min_max": MinMaxScaler, 
                            "standard": StandardScaler, 
                            "robust": RobustScaler, 
                            "normalizer": Normalizer}


def na_analysis_and_removal(df):
    #printiamo NA per variabile
    #andiamo ad osservare il numero di NAN per ciascuna variabile, e poi salviamo i nomi delle variabili con almeno un NAN
    print("Numero osservazioni: ", len(df))
    print("Numero di valori NAN per variabile:")
    print(df.isnull().sum()) 
    proportion = df.isnull().sum()/df.shape[0]   
    columns_with_name = df.columns[df.isnull().any()] # mi prende solo il nome delle  variabili con na 
    # df[df.isnull().sum(axis = 1)>1] # numero di righe (osservazioni) con almeno 2 NA
    df.fillna(df.median(), inplace = True) #riempiamo gli na
    
       
       
def balance_training(X_train, y_train):
    oversampler = SMOTE()
    X_train_os, y_train_os = oversampler.fit_resample(X_train, y_train)
    return(X_train_os, y_train_os)      





def scale_data(X_train, X_test, method = "min_max"):
    #andiamo 
    scaler = SUPPORTED_SKLEARN_SCALER[method]()
    scaler.fit(X_train)    
    X_train_min_max_sc = scaler.transform(X_train)
    X_test_min_max_sc = scaler.transform(X_test)
    joblib.dump(scaler, open(SCALER_FILE, 'wb')) #potremmo usare pickle.dump
    return(X_train, X_test)
    
  



def features_selection(X_train, X_test, y_train, method = "rfe"):
#da aggiungere metodo con fclassif del kbest 
    if method == "kBest":
        k_best_anova = SelectKBest(chi2, k=15)
        k_best_anova.fit(X_train, y_train)
        X_train_new = k_best_anova.transform(X_train)
        X_test_new = k_best_anova.transform(X_test)
        joblib.dump(k_best_anova, open('chosen_features_selector.pkl', 'wb'))
    elif method == "rfe":
        clf = RandomForestClassifier(n_estimators=200, random_state=300, criterion = 'gini', bootstrap = True, warm_start = True)
        selector = RFE(clf, n_features_to_select=15, step=1)
        selector.fit(X_train, y_train)
        X_train_new = selector.transform(X_train)
        X_test_new = selector.transform(X_test)
        joblib.dump(selector, open(FEATURES_SELECTION_FILE, 'wb'))
    return(X_train_new, X_test_new)
    

def pre_processing(df, show_plot = False, training = False, apply_scaling = True, scaling_method = "min_max", features_selection_method = "rfe"):
   
    if show_plot:
        #solo nella fase di training andiamo a mostrare il boxplot delle variabili, 
        # per analizzare l'eventuale presenza di outlier
        sns.boxplot(data=df)
    
    if training:
        y = df["CLASS"]
        X = df.drop("CLASS", axis = 1)
        #splitto in train e test, con una proporzione del 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
        na_analysis_and_removal(X_train) #rimuovo gli na
        na_analysis_and_removal(X_test)
 
        print("Frequenza per classi: ")
        print(y_train.value_counts())
        print("Dataset sbilanciato!!!")           
        
        X_train, y_train = balance_training(X_train, y_train)
        
        print("Nuova frequenza per classi: ")
        print(y_train.value_counts())
        print("Dataset bilanciato!!!")  
                  
        if apply_scaling:
            X_train, X_test = scale_data(X_train, X_test, scaling_method)
        X_train_selection, X_test_selection = features_selection(X_train, X_test, method = features_selection_method)
        print("training preprocess fatto...")
        return(X_train_selection, X_test_selection, y_train, y_test)
    else:
        y = df["CLASS"]
        X = df.drop("CLASS", axis = 1)
        na_analysis_and_removal(X) 
        if apply_scaling:
        	scaler_loaded = joblib.load(open(SCALER_FILE, 'rb'))
        X = scaler_loaded.transform(X) #read pikle
        features_selection_loaded = joblib.load(open(FEATURES_SELECTION_FILE, 'rb'))
        X = features_selection_loaded.transform(X)
        return(X, y)


def models_training(df):
    # metodo per effettuare training dei classifcatori considerati
    # leggo il dataset
    
    X_train, X_test, y_train, y_test = pre_processing(df, show_plot = True, training = True, apply_scaling = True, scaling_method = "min_max", features_selection_method = "rfe")
   
    classifier_mlp = mlp_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_mlp =  f1_score(y_test, classifier_mlp.predict(X_test), average="macro")
    print("F1 for MLP :",f1_mlp)
    classifier_svm = svm_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_svm =  f1_score(y_test, classifier_svm.predict(X_test), average="macro")
    print("F1 for SVM :", f1_svm)
    classifier_dt = decision_tree_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_dt = f1_score(y_test, classifier_dt.predict(X_test), average="macro")
    print("F1 for DecisionTree :", f1_dt)
    classifier_rf = random_forest_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_rf =  f1_score(y_test, classifier_rf.predict(X_test), average="macro")
    print("F1 for RandomForest :", f1_rf)
    classifier_xgb = XGboost_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_xgb =  f1_score(y_test, classifier_xgb.predict(X_test), average="macro")
    print("F1 for XGBoost :", f1_xgb)
    classifier_knn = k_neighbors_classifier_param_selection(X_train, y_train, n_folds=10, metric='f1_macro')
    f1_knn = f1_score(y_test, classifier_knn.predict(X_test), average="macro")
    print("F1 for KNeighbors :", f1_knn)
    f1_scores = [f1_knn, f1_xgb, f1_rf, f1_dt, f1_svm, f1_mlp]
    models = [classifier_knn, classifier_xgb, classifier_rf, classifier_dt, classifier_svm, classifier_mlp]
    idx_best_model = np.argmax(f1_score)
    joblib.dump(models[idx_best_model], open(BEST_MODEL_FILE, 'wb'))


def model_evaluation(df):
    # metodo per valutare i classificatori dopo aver effettuato training
   
    X, y = pre_processing(df)
   
    model = joblib.load(BEST_MODEL_FILE)
    print()
    print("Model Parameters: ", model.get_params())
    print()
    y_pred = model.predict(X)
    performance = f1_score(y, y_pred, average="macro")
    print('Risultati Modello (f1_score): ', performance)


if __name__ == '__main__':
	
#   training_set_path = './training_set.csv'
#   training_set = pd.read_csv(training_set_path)
#   models_training(training_set)

    test_set_path = './testset.csv' # 1)
    
    test_set = pd.read_csv(test_set_path)
    model_evaluation(test_set)
