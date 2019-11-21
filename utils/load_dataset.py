import pandas as pd

def load_data(path_to_csv):
    """

    :param path_to_csv: path to csv file
    :return: tuple which contains features and labels
    """
    print('Start Reading Data')
    data = pd.read_csv(path_to_csv)
    x = data.drop(['class'], axis=1)
    y = data['class']

    return x, y