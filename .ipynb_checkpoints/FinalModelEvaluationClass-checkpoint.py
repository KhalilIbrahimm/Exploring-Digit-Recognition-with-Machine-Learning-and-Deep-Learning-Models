import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class FinalModelEvaluation:
    """
    A class for learning, predicting, testing, and plotting predicted performance on X_test, y_test dataset for the final and best-selected validation model.

    """
    def __init__(self):
        self.random_state = 2023


    def learn(self, X_train, y_train, file_name, model, save_model = None):
        '''
        Train a model on the provided training data.

        Input:
        - X_train: The training data features.
        - y_train: The training data labels.
        - file_name: The name of the file to save the trained model.
        - model: The machine learning model to be trained.
        - save_model: A flag to save the trained model.

        Output:
        - None
        '''
        
        model_learn = model.fit(X_train, y_train)
        if save_model:
            self.models_save(file_name = file_name, model = model)
        
        
    def _final_evaluation(self, X_test, y_test):
        '''
        Evaluate the final model on the test data.

        Input:
        - X_test: The test data features.
        - y_test: The test data labels.

        Output:
        - y_pred: The predicted labels.
        '''
        
        start = time.time()
        model_name, model = self.load_models() 
        y_pred = model.predict(X_test)
        svc_acc = accuracy_score(y_test, y_pred)
        end = time.time()

        totale_time = end-start
        print(f"{model_name} accuracy score:", svc_acc)
        print(f"Time: {round(totale_time/60)}m:{round(totale_time)}s.")
        return y_pred
        
    def final_evaluation(self, X_test, y_test):
        y_pred = self._final_evaluation(X_test, y_test)
        print(f"\n\n")
        self.plot_prediction_performance(X_test, y_test, y_pred)
        print(f"\n\n")
        self.confusion_matric(y_test, y_pred)

    def models_save(self, file_name = None, model = None):
        """
        Save the final model in pickle file
        """
        with open (f"models/{file_name}.pkl", "wb") as json_file:
            pickle.dump(model, json_file)

    def plot_prediction_performance(self, X_test, y_test, y_pred):
        '''
        Plot the predicted performance of the model.

        Input:
        - X_test: The test data features.
        - y_test: The test data labels.
        - y_pred: The predicted labels.

        Output:
        - A plot showing the performance of the model on the test data.
        '''
        
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', "EMPTY"]
        
        fig = plt.figure(figsize=(6,4))  # Legg til denne linjen for å opprette en figur med større størrelse
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=1, wspace=0.5)
    
        for index in range(20):
            predicted_label = y_pred[index]
            # plt.subplot( radantaler du vil i ploten, antall kolonner i vil i ploten, index+1)
            ax = fig.add_subplot(8,7,index + 1, xticks = [], yticks = [])
            ax.grid(True)
            color_map = 'Greens' if predicted_label == y_test[index] else 'Reds'
            ax.imshow(X_test[index].reshape((20, 20)), cmap=color_map, interpolation='nearest')
            ax.set_title(f"Actual:{digits[y_test[index]]} | Pred:{digits[y_pred[index]]}", fontsize = 6)

    def confusion_matric(self, y_test, y_pred):
        '''
        Plot the confusion matrix for the model's performance.

        Input:
        - y_test: The test data labels.
        - y_pred: The predicted labels.

        Output:
        - A plot of the confusion matrix.
        '''
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', "EMPTY"]
        confu_metric = confusion_matrix(y_test, y_pred)
        fig_metric = ConfusionMatrixDisplay(confu_metric, display_labels = digits)
        fig, ax = plt.subplots(figsize = (10,10))
        fig_metric.plot(ax = ax)
        ax.set_title("Confusion Matrix")
        plt.show()

        
        
    def load_models(self):
        """
        Load models pickle file in models folder. 

        Output:
        A dictinary with models that exsists.
        """
        models_file = ["KNN.pkl", "MLPClassifier.pkl", "RandomForest.pkl", "Support_Vector.pkl"]
        models = {}
        for model_name in models_file:
            for model_file in os.listdir("models"):
                if model_file.startswith(model_name):
                    with open(os.path.join("models", model_file), "rb") as file:
                        return model_name.split(".")[0], pickle.load(file)  
                        
