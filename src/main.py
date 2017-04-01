from src.preprocess import DataPreprocessing
import numpy as np
from src.NaiveBayesClassifier import NaiveBayesClassifier
import pandas as pd
import time


def preprocess_data(filepath, rmv_missing_values=True):
    """
    This function loads the data from 'filepath' and handles the missing value in the data.
    """
    process = DataPreprocessing()
    df = process.loadfile(filepath)
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('education-num', axis=1, inplace=True)
    if rmv_missing_values:
        df_processed = process.append_y(process.handle_missing_values(df, process.remove_missing_values))
    else:
        df_processed = process.append_y(process.handle_missing_values(df, process.allocate_category_to_missing_values))
    return df_processed


def evaluate_model(result):
    # condition 1 and predicted 1
    true_pos = len(result[(result['Y'] == 1) & (result['Y_posteriori_1'] > result['Y_posteriori_0'])])
    # condition 0 and predicted 0
    true_neg = len(result[(result['Y'] == 0) & (result['Y_posteriori_1'] < result['Y_posteriori_0'])])
    # condition 1 and predicted 0
    false_neg = len(result[(result['Y'] == 1) & (result['Y_posteriori_1'] < result['Y_posteriori_0'])])
    # condition 0 and predicted 1
    false_pos = len(result[(result['Y'] == 0) & (result['Y_posteriori_1'] > result['Y_posteriori_0'])])

    print("\t\t\t\t\tResults")
    print("\n\t\t\t\t\tConfusion Matrix")
    print("===============================================================")
    print("\t\t\t  Predictions")
    print("\tsalary>50k\t\t\tsalary<=50k")
    print("\t\t" + str(true_pos) + "\t\t\t\t" + str(false_neg) + "\t\tsalary>50k")
    print("\t\t" + str(false_pos) + "\t\t\t\t" + str(true_neg) + "\t\tsalary<=50k")
    print("===============================================================")
    print("\ntotal rows processed = " + str(true_neg + true_pos + false_neg + false_pos))
    print("\ntotal rows processed accurately = " + str(true_neg + true_pos))

    # computing performance measures
    accuracy = (true_pos + true_neg) / (true_neg + true_pos + false_neg + false_pos)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_measure = (2 * precision * recall) / (precision + recall)

    precision_0 = true_neg / (true_neg + false_neg)
    recall_0 = true_neg / (true_neg + false_pos)
    f1_measure_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0)

    print('\nAccuracy = ' + str(accuracy))
    print("\nClass = 'salary>50K'")
    print('\nPrecision = ' + str(precision)
          + ',\tRecall = ' + str(recall) + ",\tF1 Measure = " + str(f1_measure))
    print("\nClass = 'salary<=50K'")
    print('\nPrecision = ' + str(precision_0)
          + ',\tRecall = ' + str(recall_0) + ",\tF1 Measure = " + str(f1_measure_0))


def main():
    """
    main driver function
    """
    binning = [True, False]
    rmv_missing_val = [True, False]

    # run for all possible combinations of {binning, gaussian} and
    # {remove missing values, assign category to missing values}
    for rmv_bool in rmv_missing_val:
        for bin_bool in binning:
            print("----------------------------------Start------------------------------------------")
            print("\n\t\tBinning = " + str(bin_bool) + " and Removing missing value = " + str(rmv_bool))
            # get start time
            st_time = time.time()

            # preprocess data
            df = preprocess_data("../data/adult_censusdata.txt", rmv_bool)

            # define categorical and numeric attributes
            col_categorical = ['Workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                               'native-country']
            col_numeric = {'Age': 5, 'capital-gains': 5075, 'capital-loss': 270, 'hours-per-week': 10}
            # define dependent variable
            dependent_var = 'Y'

            # split dataset for 10 fold cross validation

            KFold = np.array_split(df, 10)
            tempresult = []
            print("Running 10 fold cross validation...")

            # run 10 fold cross validation
            for i in range(10):
                train_data = pd.DataFrame()
                # test on all data sets except for ith one
                for k in range(10):
                    if k != i:
                        # accumulate train data
                        train_data = train_data.append(KFold[k])
                # get test data
                test_data = KFold[i]
                # define NaiveBayes classifier
                classifier = NaiveBayesClassifier(col_categorical, col_numeric, bin_bool)
                # generate Naive Bayes model
                model = classifier.train_model(train_data, dependent_var)
                # test model on test data set.
                tempresult.append(classifier.test_model(model, test_data))
            # accumulate all 10 fold crossvalidation results
            result = pd.concat(tempresult)
            print("Evaluating model...")
            # evaluate the model's results
            evaluate_model(result)
            print("-------------------------------Run time = {} secs-------------------".format(time.time() - st_time) + "\n")


if __name__ == '__main__':
    main()
