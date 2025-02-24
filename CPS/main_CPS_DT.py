# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import os  # For interacting with the operating system
from sklearn.model_selection import StratifiedKFold  # For stratified k-fold cross-validation
from sklearn import metrics  # For evaluating model performance
import numpy as np  # For numerical operations

# Import custom modules
from dataprocess import load_in_data  # Custom function to load and preprocess data
from classifier import DecisionTree  # Custom Decision Tree classifier
from CPS_CCA_test_k_lamda import CPS  # Custom Critical Pattern Selection (CPS) resampling technique

# Main script execution
if __name__ == '__main__':

    # Define the path to the dataset directory
    path = "./Dataset"
    files = os.listdir(path)  # List all files in the dataset directory

    k = 5  # Parameter for calculating Bayes posterior probability, default=5

    # Loop through each file in the dataset directory
    for s in range(len(files)):
        pathway = open(path + "\\" + files[s])  # Open the dataset file
        dataframe = pd.read_csv(pathway, header=None)  # Read the dataset into a DataFrame
        name = files[s]
        name = name.replace('.csv', '')  # Remove the file extension from the name
        print(name)  # Print the name of the current dataset

        # Create directories to store results if they don't exist
        file_path = r'./result/DT/result_auc'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_auc.xls'
        result_auc = open(file_name, 'a+', encoding='gbk')  # Open file to store AUC results

        file_path = r'./result/DT/result_Fmeasure'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_Fmeasure.xls'
        result_Fmeasure = open(file_name, 'a+', encoding='gbk')  # Open file to store F-measure results

        file_path = r'./result/DT/result_Gmean'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_Gmean.xls'
        result_Gmean = open(file_name, 'a+', encoding='gbk')  # Open file to store G-mean results

        file_path = r'./result/DT/result_recall'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_recall.xls'
        result_recall = open(file_name, 'a+', encoding='gbk')  # Open file to store Recall results

        file_path = r'./result/DT/result_spec'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = file_path + '/' + name + '_spec.xls'
        result_spec = open(file_name, 'a+', encoding='gbk')  # Open file to store Specificity results

        # Load and preprocess the data using the custom function
        data = load_in_data(dataframe)

        # Define parameters for CPS resampling
        lambda1_list = [0.5]  # Parameter for minority critical pattern selection, default=0.5
        lambda2_list = [0.4]  # Parameter for majority cleaning, default=0.4

        # Write the lambda2 values to the result files
        for i in range(len(lambda2_list)):
            result_auc.write('\t')
            result_auc.write(str(lambda2_list[i]))
            result_Fmeasure.write('\t')
            result_Fmeasure.write(str(lambda2_list[i]))
            result_Gmean.write('\t')
            result_recall.write(str(lambda2_list[i]))
            result_recall.write('\t')
            result_recall.write(str(lambda2_list[i]))
            result_spec.write('\t')
            result_spec.write(str(lambda2_list[i]))

        result_auc.write('\n')
        result_Fmeasure.write('\n')
        result_Gmean.write('\n')
        result_recall.write('\n')
        result_spec.write('\n')

        # Loop through each lambda1 value
        for m in range(len(lambda1_list)):
            lambda1 = lambda1_list[m]
            result_auc.write(str(lambda1))
            result_auc.write('\t')
            result_Fmeasure.write(str(lambda1))
            result_Fmeasure.write('\t')
            result_Gmean.write(str(lambda1))
            result_Gmean.write('\t')
            result_recall.write(str(lambda1))
            result_recall.write('\t')
            result_spec.write(str(lambda1))
            result_spec.write('\t')

            # Initialize lists to store average metrics
            average_auc = []
            average_Fmeasure = []
            average_Gmean = []
            average_recall = []
            average_spec = []
            auc_index = []
            fmeasure_index = []
            gmean_index = []
            recall_index = []
            spec_index = []

            # Initialize average metrics and indices
            for j in range(len(lambda2_list)):
                average_auc.append(0.0)
                average_Gmean.append(0.0)
                average_Fmeasure.append(0.0)
                average_recall.append(0.0)
                average_spec.append(0.0)
            for j in range(len(lambda2_list)):
                auc_index.append(0)
                fmeasure_index.append(0)
                gmean_index.append(0)
                recall_index.append(0)
                spec_index.append(0)

            # Perform 10 iterations of 5-fold cross-validation
            for l in range(10):
                print('------' + str(l+1) + 'th 5-fold cross validation------')
                kf = StratifiedKFold(n_splits=5, shuffle=True)  # Initialize stratified k-fold cross-validation
                X = []
                y = []
                size = len(data)
                num_of_minority = 0
                num_of_majority = 0

                # Separate features (X) and labels (y)
                for j in range(len(data)):
                    pattern = data[j].copy()
                    label = pattern.pop()
                    y.append(label)
                    X.append(pattern)
                    if label == 0:
                        num_of_minority += 1
                    else:
                        num_of_majority += 1

                kf.get_n_splits(X, y)

                # Iterate through each fold
                for train_index, test_index in kf.split(X, y):
                    # Divide the data into training and testing sets
                    train_data = []
                    test_data = []
                    for i in range(len(train_index)):
                        train_data.append(data[train_index[i]])

                    for i in range(len(test_index)):
                        test_data.append(data[test_index[i]])

                    # Separate features and labels for training and testing sets
                    train_label = []
                    train_pattern = []
                    for i in range(len(train_data)):
                        pattern = train_data[i].copy()
                        train_label.append(pattern.pop())
                        train_pattern.append(pattern)

                    test_label = []
                    test_pattern = []
                    for i in range(len(test_data)):
                        pattern = test_data[i].copy()
                        test_label.append(pattern.pop())
                        test_pattern.append(pattern)

                    # Apply CPS resampling and train the Decision Tree classifier
                    for n in range(len(lambda2_list)):
                        lambda2 = lambda2_list[n]

                        train_pattern_resampled, train_label_resampled = CPS(train_data, k, lambda1, lambda2)
                        result = DecisionTree(train_pattern_resampled, train_label_resampled, test_pattern)

                        # Calculate evaluation metrics
                        cm = metrics.confusion_matrix(test_label, result)
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        tp = cm[0, 0]
                        fn = cm[0, 1]
                        fp = cm[1, 0]
                        tn = cm[1, 1]
                        tp_rate = tp / (tp + fn)
                        fp_rate = fp / (fp + tn)
                        spec_temp = tn / (tn + fp)
                        auc_temp = (1 + tp_rate - fp_rate) / 2
                        precision_temp = tp / (tp + fp)
                        recall_temp = tp / (tp + fn)  # recall
                        f_measure_temp = (2 * tp) / (2 * tp + fp + fn)
                        g_mean_temp = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

                        # Print the metrics for the current fold
                        print('AUC:' + str(auc_temp) + ' Fmeasure:' + str(f_measure_temp) + ' Gmean:' + str(g_mean_temp) + ' Recall:' + str(recall_temp) + ' Spec:' + str(spec_temp))

                        # Accumulate metrics for averaging
                        average_auc[n] += auc_temp
                        auc_index[n] += 1
                        average_Gmean[n] += g_mean_temp
                        gmean_index[n] += 1
                        fmeasure_index[n] += 1
                        average_Fmeasure[n] += f_measure_temp
                        average_recall[n] += recall_temp
                        recall_index[n] += 1
                        average_spec[n] += spec_temp
                        spec_index[n] += 1

            # Calculate the average metrics across all folds
            for i in range(len(average_auc)):
                average_auc[i] /= auc_index[i]
                average_Fmeasure[i] /= fmeasure_index[i]
                average_Gmean[i] /= gmean_index[i]
                average_recall[i] /= recall_index[i]
                average_spec[i] /= spec_index[i]

            # Write the average metrics to the result files
            for i in range(len(average_auc)):
                result_auc.write(str(average_auc[i]))
                result_auc.write('\t')
                result_Fmeasure.write(str(average_Fmeasure[i]))
                result_Fmeasure.write('\t')
                result_Gmean.write(str(average_Gmean[i]))
                result_Gmean.write('\t')
                result_recall.write(str(average_recall[i]))
                result_recall.write('\t')
                result_spec.write(str(average_spec[i]))
                result_spec.write('\t')

            result_auc.write('\n')
            result_Fmeasure.write('\n')
            result_Gmean.write('\n')
            result_recall.write('\n')
            result_spec.write('\n')