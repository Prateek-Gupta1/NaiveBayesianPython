import numpy as np
import pandas as pd
import math

def calc_probability(col_dict):
    """
    This function calculates probability of each item in the dictionary
    For example, {'item1':10,'item2':30} gives  prob = {'item1': 0.25, 'item2': 0.75}
    """
    s = sum(col_dict.values())
    for key, val in col_dict.items():
        col_dict[key] = val / s
    return col_dict


def calc_gaussian(val, mean, std):
    """
    This function determines the gaussian probability of a given value based on mean and standard deviation.
    :param val:
    :param mean:
    :param std:
    :return: numeric probability
    """
    variance = float(std) ** 2
    pi = 3.1415926
    denom = (2 * pi * variance) ** .5
    num = math.exp(-(float(val) - float(mean)) ** 2 / (2 * variance))
    num_prob = num / denom
    return num_prob


def binning(col, bins):
    """
    This function converts numeric to categorical values by allocating a bin  from bins to a given value
    :param col: dataframe column
    :param bins: array of bin values
    :return: dictionary of bins and there corresponding counts
    """
    return pd.value_counts(pd.cut(col, bins).values).to_dict()


class NaiveBayesClassifier:
    """
    This is an implementation of Naive Bayes Classifier.
    """
    def __init__(self, categorical_cols, numeric_cols, use_binning=False):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.use_binning = use_binning
        self.y = 'Y'

    def train_model(self, df, dependent_var):
        """
        This function generates a model from a given dataframe object.
        :param: df : dataframe
        :param: dependent_var : string name of dependent variable in the dataframe
        """
        self.y = dependent_var

        # define model as empty
        model = {}

        # dictionary containing dependent variable and count of each dependent variable
        y_values = df[dependent_var].value_counts().to_dict()

        # probability of each dependent variable(prior probability)
        model['p_prior'] = calc_probability(y_values)

        # if numeric data is to be converted to categorical using binning then model is loaded with bins for each column
        # for example, data = [3,6,9,14,6] will give bin = [3,8,13,18] for bin size 5
        if self.use_binning is True:
            for column, binsize in self.numeric_cols.items():
                # check if bin size greater than 0
                if binsize > 0:
                    model[column] = [i for i in range(df[column].min(), df[column].max(), binsize)]

        # for each dependent variable
        for y in y_values:
            # define model as dictionary.
            model[y] = {}

            # for each categorical type column
            for column in self.categorical_cols:
                # count the frequency of each category in a given categorical column and a given y value.
                # for example, in 'education' column find count of category '10th' given salary is '>50k'
                col_cat_counts = pd.value_counts(df[column][df[dependent_var] == y].values).to_dict()

                # find probability associated with each category and put all categories of
                # the column as dictionary in the model
                model[y][column] = calc_probability(col_cat_counts)

            # for each continuous numeric type columnc
            for column, binsize in self.numeric_cols.items():

                # if binning is enabled and binsize provided is > 0
                if self.use_binning is True and binsize > 0:
                    model[y][column] = calc_probability(binning(df[column][df[dependent_var] == y], model[column]))
                # To use gaussian distribution formula for calculating probability we store
                #mean and standard distribution of the values.
                else:
                    model[y][column] = {'mean': df[column][df[dependent_var] == y].values.mean(),
                                        'std': df[column][df[dependent_var] == y].values.std()}
        return model

    def test_model(self, model, test_data):
        """
        This function predicts posteriori probabilities for every class of dependent variable, and returns ground truth
        and predictions as dataframe.
        for example, for row {x1:4, x2:6, y:1} return {y:1, y_1: 0.007293, y_2: 0.0028323}
        :param model: Dictionary object
        :param test_data: Dataframe
        :return: test dataframe with dependent variable and corresponding predictions
        """
        # for each unique class of dependent variable, create a new column and assign 0.0 in every row
        for y in model['p_prior']:
            test_data.loc[:, 'Y_posteriori_' + str(y)] = 0.0

        # The following block of code predicts the posteriori probability of a row vector based on the
        # model passed to the function.
        for idx, row in test_data.iterrows():
            # given prior = y
            for y in model['p_prior']:
                likelihood = 1

                # the following block of code calculates the likelihood of a row vector for the given class label 'y'.

                # for all categorical attributes
                for col in self.categorical_cols:
                    try:
                        # if probability of an attribute is 0 then multiply a low probability of 0.0001
                        likelihood *= model[y][col][row[col]] if model[y][col][row[col]] else 0.0001
                    except KeyError:
                        pass
                # for all numeric attributes
                for col, val in self.numeric_cols.items():
                    # if binning is set to true
                    if self.use_binning:
                        # get bins for the attribute from model
                        bins = model[col]
                        # test the edge condition if the attribute value fall above the bin max value
                        if row[col] > max(bins):
                            prob = 1.0
                            for key, val in model[y][col].items():
                                prob = val
                            # multiply probability of max bin if it is not zero
                            likelihood *= prob if prob else 0.0001
                        # test the edge condition if the attribute value fall below the bin min value
                        elif row[col] <= min(bins):
                            for key, val in model[y][col].items():
                                # multiply the probability of min bin if it is not 0
                                likelihood *= val if val else 0.0001
                                break
                        # if the attribute value fall within the bin.min and bin.max
                        else:
                            # find the bin in which the value falls
                            bin = pd.cut([row[col] if row[col] else 1.0], model[col])
                            prob = float(model[y][col][bin[0]])
                            # multiply probability of the bin if it is not 0
                            likelihood *= prob if prob else 0.0001
                    # if binning is set to false, then we use gaussian probability function.
                    else:
                        likelihood *= calc_gaussian(row[col], model[y][col]['mean'], model[y][col]['std'])
                # for each prior, put the corresponding posteriori probability in a separate column
                test_data.loc[idx,'Y_posteriori_' + str(y)] = likelihood * model['p_prior'][y]
        #print(test_data)
        # return only ground truth and predicted values for evaluations
        return test_data.ix[:,self.y:]
