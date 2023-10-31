# INF264 - Prosjekt 2 - Kib019

##Setup and Requirements
 
1. This project uses Python 3.11.
2. Create a virtual environment by running the following command in your terminal:

    ```bash
    python3 -m venv venv
    ```

    Then, activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

3. Install the required packages by running the following commands:

    ```
    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install scikit-learn
    ```

4. Make sure you have the dataset in the project folder, under the "data" directory. The files should be named "data/emnist_hex_images.npy" and "data/emnist_hex_labels.npy".


## Filstruktur og Forklaring

The project folder contains the following files and their purposes:

- ModelImplementationClass: This file implements all the models, including the process of selecting the best model.

- DataPreparationClass: This file handles the data preprocessing process.

- FinalModelEvaluationClass: This file tests the best-selected model, along with its visualization graphs and matrices.

- "models"-folder: All validation models and the final selected model are stored in pickle files.

- main.ipynb: This is the main program to be executed. Here you will find code for evaluation, testing, and generating final results based on the selected models. To generate final results, install all the necessary packages and run main.ipynb by selecting "Run All" and waiting for the output results. Note that the "Find the best validation model (Cross Validation)" cell will take time to execute, as it performs cross-validation on all the models.



