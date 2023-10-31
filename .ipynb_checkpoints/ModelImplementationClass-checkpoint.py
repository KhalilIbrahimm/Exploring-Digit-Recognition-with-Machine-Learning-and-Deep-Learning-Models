
## loade biblioteker ##
import time
import pickle
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


## Define a class for implementing various machine learning models
class Models_Implementation:
    def __init__(self):
        # # Here, all models will be collected for final evaluation
        self.find_best_model = {}
        self.random_state = 2023

    def cross_validation(self, X, y, model, parameters):
        """
        Perform cross-validation to find the best model hyperparameters.

        Inputs:
        X: Feature data.
        y: Target labels.
        model: The machine learning model.
        parameters: Dictionary of hyperparameters for grid search.

        Outputs:
        best_model: Best model after cross-validation.
        fit_mean_time: Mean time taken for model fitting.
        best_score: Best accuracy score.
        pred_mean_time: Mean time taken for prediction.
        """
        gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, verbose=2, n_jobs = 6)
        gs.fit(X,y)
        best_model = gs.best_estimator_
        best_score = gs.best_score_ * 100    # 0.90 * 100 = 90%
        best_parameters = gs.best_params_

        best_parameter_index = gs.cv_results_["params"].index(best_parameters)
        fit_mean_time = gs.cv_results_["mean_fit_time"][best_parameter_index]
        pred_mean_time = gs.cv_results_["mean_score_time"][best_parameter_index]
        
        print(f"Mean score: {best_score}")
        print(f"With parameter: {best_parameters}")
        print(f"Fit mean time: {fit_mean_time}")
        print(f"Prediction mean time: {pred_mean_time}")
        return best_model, fit_mean_time, best_score, pred_mean_time
        
    def support_vector(self, X, y, save_model_checks = None):
        """
        Implement a Support Vector Machine (SVM) model with cross-validation.

        Inputs:
        X: Feature data.
        y: Target labels.
        save_model_checks: Flag to save the model as a pickle file.

        Outputs:
        If save_model_checks is True, the model is saved as a pickle file.
        Otherwise, the best model is returned.
        """
        start = time.time()
        print("\n    ** Support Vector Machine **")
        print("        - Start:")
        parameters = {"kernel":["linear", "poly", "sigmoid"],
        "C":[1, 10, 100]}
        
        support_vector_model = SVC(random_state = self.random_state)
        best_model, fit_mean_time, best_score, pred_mean_time = self.cross_validation(X, y, support_vector_model, parameters)
        slutt = time.time()
        tid = slutt - start
        
        ## Add the best model to the find_best_model dictionary
        self.find_best_model[best_score] = [best_model, "Support_Vector.pkl", pred_mean_time]

        #print(f"        - Tid: {round(tid)}s:{round(tid/60)}m")
        if save_model_checks:
            self.models_save(model_navn = "Validation_Support_Vector", model_result_dict = self.find_best_model, save_model = save_model_checks)
            print("Model saved!")
        else:
            best_model
        print("Done!\n\n")


    def MLPC(self, X, y, save_model_checks = None):
        """
        Implement a Multi-layer Perceptron (MLP) classifier model with cross-validation.

        Inputs:
        - X: Feature data.
        - y: Target labels.
        - save_model_checks: Flag to save the model as a pickle file.

        Outputs:
        - If save_model_checks is True, the model is saved as a pickle file.
        - Otherwise, the best model is returned.
        """
        
        start = time.time()
        print("\n    ** MLPClassifier ** ")
        print("        - Start:")
        parameters = {"hidden_layer_sizes" :[(64,2), (128,3), (256,4), (400,4)],
        "learning_rate_init" : [0.001, 0.01, 0.1]}
        mlpc_model = MLPClassifier(activation="relu", solver = "adam", max_iter = 1000)
        best_model, fit_mean_time, best_score, pred_mean_time = self.cross_validation(X, y, mlpc_model, parameters)
        slutt = time.time()
        tid = slutt - start
        # Add to find_best_model dict
        self.find_best_model[best_score] = [best_model, "MLPClassifier.pkl", pred_mean_time]
        
        #print(f"            - Tid: {round(tid)}s:{round(tid/60)}m")
        
        if save_model_checks:
            self.models_save(model_navn = "Validation_MLPClassifier", model_result_dict = self.find_best_model, save_model = save_model_checks)
            print("Model saved!")
        else:
            best_model
        print("Done!\n\n")

    def RandomForest(self, X, y, save_model_checks = None):
        """
        Implement a Random Forest classifier model with cross-validation.

        Inputs:
        - X: Feature data.
        - y: Target labels.
        - save_model_checks: Flag to save the model as a pickle file.

        Outputs:
        - If save_model_checks is True, the model is saved as a pickle file.
        - Otherwise, the best model is returned.
        """
        start = time.time()
        print("\n    ** RandomForest **")
        print("        - Start:")
        parameters = {"n_estimators": [20,30,40], 
                     "criterion": ["entropy", "gini", "log_loss"]}
        random_forest_model = RandomForestClassifier(self.random_state)
        best_model, fit_mean_time, best_score, pred_mean_time = self.cross_validation(X, y, random_forest_model, parameters)
        slutt = time.time()
        tid = slutt - start
        # Add the best model to the find_best_model dictionary
        self.find_best_model[best_score] = [best_model, "RandomForest.pkl", pred_mean_time]
        
        #print(f"            - Tid: {round(tid)}s:{round(tid/60)}m")
        if save_model_checks:
            self.models_save(model_navn = "Validation_RandomForest", model_result_dict = self.find_best_model, save_model = save_model_checks)
            print("Model saved!")            
        else:
            best_model
        print("Done!\n\n")


    def KNN(self, X, y, save_model_checks = None):
        """
        Implement a k-Nearest Neighbors (KNN) classifier model with cross-validation.

        Inputs:
        - X: Feature data.
        - y: Target labels.
        - save_model_checks: Flag to save the model as a pickle file.

        Outputs:
        - If save_model_checks is True, the model is saved as a pickle file.
        - Otherwise, the best model is returned.
        """
        
        start = time.time()
        print("\n    ** KNN **")
        print("        - Start:")
        parameters = {"n_neighbors": [1,3,5,7,9],
                     "weights": ["uniform", "distance"]}
        knn_model = KNeighborsClassifier()
        best_model, fit_mean_time, best_score, pred_mean_time = self.cross_validation(X, y, knn_model, parameters)
        slutt = time.time()
        tid = slutt - start
        # Add to find_best_model dict
        self.find_best_model[best_score] = [best_model, "KNN.pkl", pred_mean_time]
            
        #print(f"            - Tid: {round(tid)}s:{round(tid/60)}m")
        if save_model_checks:
            self.models_save(model_navn = "Validation_KNN", model_result_dict = self.find_best_model, save_model = save_model_checks)
            print("Model saved!")
        else:
            best_model
        print("Done!\n\n")


    def find_best_validation_model(self):
        """
        Find and print the best model based on accuracy and prediction time.
        """
        
        models_dict = self.find_best_model
         # Sorter modellene først etter prediksjonstid (laveste først), deretter etter nøyaktighet (høyeste først)
        sorted_models = sorted(models_dict.items(), key=lambda x: (x[1][2], -x[0]), reverse=False)
        
        # Get the best model from the sorted list
        best_accuracy, (best_model, file_name, best_model_pred_time) = sorted_models[0]
    
        # Print the best model along with its accuracy score and prediction time
        print(f"Best model: {best_model}.")
        print(f"Best accuracy score: {best_accuracy}.")
        return best_model, file_name

        
    def models_save(self, model_navn = None, model_result_dict = None, save_model = None):
        """
        Save the best model and its results as a pickle file.

        Inputs:
        - model_name: Name of the model.
        - model_result_dict: Dictionary of model results.
        - save_model: Flag to save the model as a pickle file.

        Output:
        - Saved model pickle file.
        """
        if save_model:
            with open (f"models/{model_navn}.pkl", "wb") as pickle_file:
                pickle.dump(model_result_dict, pickle_file)

    def load_models(self):
        """
        Load saved models from pickle files.

        Outputs:
        - A dictionary with the loaded models.
        """
        models_file = ["Validation_KNN.pkl", "Validation_MLPClassifier.pkl", "Validation_RandomForest.pkl", "Validation_Support_Vector.pkl"]
        models = {}
        for model in models_file:
            for model_file in os.listdir("models"):
                if model_file.startswith(model):
                    with open(os.path.join("models", model_file), "rb") as file:
                        #print(pickle.load(file))
                        model_ = [find_model[0] for find_model in pickle.load(file).values()][0]
                        


    def plot_model_comparsion(self):
        """
        Plot and compare model performance in terms of accuracy and prediction time.
        """
        model_data = self.find_best_model
        
        #Hent nøyaktighetsverdier, modellnavn, modeller og tid fra datasettet
        accuracy_values = list(model_data.keys())
        model_names = [model_data[model][1] for model in accuracy_values]
        models = [model_data[model][0] for model in accuracy_values]
        time_taken = [model_data[model][2] for model in accuracy_values]

        #plt.figure(figsize=(6, 4))
        # Slå sammen de to plottene
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Validation Accuracy", color = "blue")
        ax1.plot(model_names, accuracy_values, marker="o", color="blue", label = "Accuracy Score")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Prediction time (s)", color = "red")
        ax2.plot(model_names, time_taken, marker="o", color="red", label="Time (s)")
        
        plt.title("Validation Models Perfomance Results!")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
            
    def validate_all(self, X, y, save_model = None):
        """
        Perform cross-validation for all implemented models.

        Inputs:
        - X: Feature data.
        - y: Target labels.
        - save_model: Flag to save the model as a pickle file.
        """
        # Her skal det evalueres alle modeller på en funksjon. 
        self.support_vector(X, y, save_model)
        self.KNN(X, y, save_model)
        self.RandomForest(X, y, save_model)
        self.MLPC(X, y, save_model)


