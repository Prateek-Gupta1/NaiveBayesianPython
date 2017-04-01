import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', None)
pd.set_option('display.width', 300)

class DataPreprocessing():
    """
    All data prepocessing tasks are delegated to this class
    """

    def handle_missing_values(self, dataset, func):
        """
        Handles missing value acording to the function object provided
        :param dataset: dataframe
        :param func: function object
        :return: dataframe
        """
        return func(dataset)

    def loadfile(self, filepath):
        """
        Loads data from csv file to a dataframe object
        :param filepath: csv file
        :return: dataframe
        """
        df = pd.read_csv(filepath)
        print("\nData loaded successfully from the file {}".format(filepath))
        return df

    def remove_missing_values(self, dataframe):
        """
        Removes the rows which are missing at attribute value
        :param dataframe: Pandas Dataframe
        :return: dataframe
        """
        df = dataframe.replace({'?': np.nan}).dropna()
        print("Removed rows that are missing a value.")
        return df

    def allocate_category_to_missing_values(self, df):
        """
        Assign a new category 'Unknown' to missing attribute values
        :param df: dataframe
        :return: dataframe
        """
        df = df.replace({'?' : 'Unknown'})
        print("Replaced missing values with a new category called 'Unknown'. ")
        return df

    @staticmethod
    def append_y(df):
        """
        Append auxiliary column for dependent variable
        :param df: dataframe
        :return: dataframe
        """
        df['Y'] = np.where(df['salary'] == "<=50K", 0, 1)
        print("Appended 'Y' for salary")
        return df
