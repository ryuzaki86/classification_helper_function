
import time
import pandas as pd
import numpy as np
from sklearn import metrics
from classifier import Classifier


class RandomForestClassifier(Classifier):
    """
    Subclass of classifier to run classification via random forest
    """



    def run_pipeline(
            self, 
            criteria_to_include, 
            max_features_to_include,
            estimators_range_to_include, 
            number_of_splits, 
            scoring_metric
            ):
        """
        Function to call train/test data split and run pipeline
        
        Takes in responses from user:
            
        criteria_to_include is a list of bools to indicate whether to include
        the gini and entropy split measurement criteria
        
        max_features_to_include is a list of bools to indicate whether to 
        include the log2 and sqrt options for number of features to consider
        for a split
        
        estimators_range_to_include is a range containing the number
        of trees in the forest to try
        
        """
        
        split_data = self.split_data()
        
        X_train = split_data["training_predictors"]
                
        y_train = split_data["training_outcome"]
        
        X_test = split_data["test_predictors"]
                
        y_test = split_data["test_outcome"]
        
        all_criteria = ['gini', 'entropy']
    
        all_max_features = ['log2', 'sqrt']
        
        pipeline_config = {            
            "criterion" : Classifier.bool_filter(
                    all_criteria, criteria_to_include
                    ),
            "n_estimators" : estimators_range_to_include,
            "max_features" : Classifier.bool_filter(
                    all_max_features, max_features_to_include
                    ),
        }
        
        pipeline_config = {k: v for k, v in pipeline_config.items() if v}
        
        if not pipeline_config:
            raise ValueError("Invalid pipeline")
                                
        n_splits = number_of_splits
                
        scoring_metric = scoring_metric
        
        method = "Random Forest"

        model = self.get_best_estimator(predictors = X_train, response = y_train, 
                                  pipeline_config = pipeline_config,
                               n_splits = n_splits, scoring_metric = scoring_metric, 
                               method = method
                              )
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        conf_matrix_lr = metrics.confusion_matrix(preds, y_test) 
        
        self.plot_confusion_matrix(conf_matrix_lr, classes = self.outcome.columns.tolist(), normalize = True)
        
        self.plot_feature_importance(features = X_train, estimator = model)

        return model



        
