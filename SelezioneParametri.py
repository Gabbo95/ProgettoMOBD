
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def svm_param_selection(x, y, n_folds, metric):
    # Iperparametri per svm
  parameters = [
                  {"kernel": ['rbf'], 'C': [20, 50, 100, 150 ,200],
                   "gamma": [10e-1, 10, 0.5, 10 ** 2],
                   "decision_function_shape": ["ovo", "ovr"]
                   },
                  {
                   "kernel": ['linear'],
                   "C": [1, 10, 25], 
                   "decision_function_shape": ["ovo", "ovr"]
                  }
    ]
  clf = RandomizedSearchCV(SVC(), param_distribution=parameters, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
  clf.fit(x, y)

  print("Best parameters:\n")
  print(clf.best_params_)
  return (clf.best_estimator_)

def decision_tree_param_selection(X, y, n_folds, metric):
    # Iperparametri per DecisionTree
  param_grid = {
               'criterion': ['entropy', 'gini'],
               'splitter': ['best', 'random'],
               'max_features': [None ,'auto', 'log2'],
               'min_samples_leaf': [1, 3, 5],
               'max_depth': [ 25, 50, 100, 125],
               'min_samples_split': [2, 5, 10]
    }

  clf = RandomizedSearchCV(DecisionTreeClassifier(), param_distribution=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                      n_jobs=-1)
  clf.fit(X, y)

  print("Best parameters:")
  print(clf.best_params_)

  return (clf.best_estimator_)

def random_forest_param_selection(X, y, n_folds, metric):
    # Iperparametri per RandomForest
  param = {
        'criterion': ['entropy', 'gini'],
        'max_features': [None, 'auto', 'log2'],
        'min_samples_leaf': [1, 3, 5],
        'max_depth': [25, 50, 100, 125],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [100, 175, 350, 400]
    }


  clf = RandomizedSearchCV(RandomForestClassifier(), param_grid=param, scoring=metric,
                                     cv=n_folds, refit=True,
                                     n_jobs=-1)
  clf.fit(X, y)

  print("Best parameters:")
  print(clf.best_params_)
  return (clf.best_estimator_)

def mlp_param_selection(X, y, n_folds, metric):
    # Iperparametri per MLP
  parameters = [{
      'max_iter': [2200, 3500, 5000, 7000, 12000],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'hidden_layer_sizes': [(150, 75), (250, 125, 60), (512, 250, 125)],
      'shuffle': [True, False],
      'learning_rate': ['invscaling', 'constant', 'adaptive'],
      'learning_rate_init': [0.01, 0.008, 0.006, 0.004],
      'solver': ['sgd', 'adam'],
      'warm_start': [True, False],
      'early_stopping': [True, False], 
      'momentum': [0.87, 0.86, 0.85, 0.84, 0.83],
      'n_iter_no_change': [20, 30, 33, 35, 40],
  }]

  clf = RandomizedSearchCV(MLPClassifier(), param_distribution=parameters,
                                     scoring=metric,
                                     cv=n_folds, refit=True,
                                     n_jobs=-1)
  clf.fit(X, y)

  print("Best parameters:")
  print(clf.best_params_)

  return (clf.best_estimator_)


def k_neighbors_classifier_param_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
  parameters = [{
        'n_neighbors': [5, 10, 13, 18],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 25, 30, 40],
        'p': [1, 2, 3, 4, 5],
    }]

  clf = RandomizedSearchCV(KNeighborsClassifier(), param_distribution=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
  clf.fit(X, y)

  print("Best parameters:")
  print(clf.best_params_)
  return (clf.best_estimator_)

def XGboost_param_selection(X, y, n_folds, metric):
    # Iperparametri per XGboost
  parameters = [{
        'learning_rate': [0.01, 0.008, 0.006, 0.004],
        'n_estimators': [100, 200, 350, 500],
        'max_depth': [ 50, 100, 150, 200],
        'booster': ['gbtree', 'gblinear', 'dart']
            
    }]

  clf = RandomizedSearchCV(XGBClassifier(), param_distribution=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
  clf.fit(X, y)

  print("Best parameters:")
  print(clf.best_params_)
  return (clf.best_estimator_)