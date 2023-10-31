import numpy as np
import matplotlib.pyplot as plt


## A class for data preparation, cleaning and visualization

class DataPreparation:
    def __init__(self):
        pass

    def data_normalize(self, data):
        '''
        Normalize data to speed up calculations.
        
        Input:
        - data: The input data to be normalized.
        
        Output:
        - normalized_data: The normalized data.
        '''
        min_value = np.min(data)
        max_value = np.max(data)

        normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data
    
    def clean_data(self, X):
        '''
        Clean and normalize data.

        Input:
        - X: The data to be cleaned and normalized.

        Output:
        - X: Cleaned and normalized data.
        '''

        # Clean pixel backgrounds
        X[X<=140] = 0
        X[X[:]>= 160] = 255

        # Normalize the dataset for faster processing
        X = self.data_normalize(X)
        return X

    


    def plot_digits_distribution(self, y):
        '''
        Plot the distribution of digits in the labels.

        Input:
        - y: The labels containing digit information.

        Output:
        - A bar plot showing the distribution of digits.
        '''
        
        # Find the digits and their occurrences in the y-label
        unique_labels, label_counts = np.unique(y, return_counts=True)
        # Change digit names from numbers 0-14 to 0-10 and A-F to use them on the x-axis
        unique_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', "EMPTY"])
        plt.figure(figsize=(8, 6))
        plt.bar(unique_labels, label_counts, )
        plt.xlabel('Digits')
        plt.ylabel('Sum')
        plt.title('Digits size')
        
        # Add numbers above the bars
        for i, count in enumerate(label_counts):
            plt.text(unique_labels[i], count, str(count), ha='center', va='bottom')
        plt.show()