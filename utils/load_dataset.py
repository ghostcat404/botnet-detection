import pandas as pd

def load_data(path_to_csv):
    """

    :param path_to_csv:
    :return:
    """
    print('Start Reading Data')
    data = pd.read_csv(path_to_csv)
    x = data.drop(['class'], axis=1)
    y = data['class']

    return x, y