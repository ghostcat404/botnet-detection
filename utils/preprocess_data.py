import pandas as pd
import sklearn.utils as util
from sklearn.model_selection import train_test_split


def load():
    """

    :return:
    """
    print('Start data extraction')
    benign = pd.read_csv('../../data/benign/benign_traffic.csv')
    benign['class'] = 0
    g_combo = pd.read_csv('../../data/gafgyt_attacks/combo.csv')
    g_combo['class'] = 1
    g_junk = pd.read_csv('../../data/gafgyt_attacks/junk.csv')
    g_junk['class'] = 2
    g_scan = pd.read_csv('../../data/gafgyt_attacks/scan.csv')
    g_scan['class'] = 3
    g_tcp = pd.read_csv('../../data/gafgyt_attacks/tcp.csv')
    g_tcp['class'] = 4
    g_udp = pd.read_csv('../../data/gafgyt_attacks/udp.csv')
    g_udp['class'] = 5
    print("Data extraction : Success\n")

    print('Start data preprocessing')
    malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp])
    benign = util.shuffle(benign)
    malicious = util.shuffle(malicious)
    traffic_ = pd.concat([benign, malicious])
    traffic_ = util.shuffle(traffic_)
    print('Data preprocessing : Success\n')

    split_data(traffic_)

def split_data(traffic_):
    """

    :param traffic_:
    :return:
    """
    target = traffic_['class']
    traffic_ = traffic_.drop(['class'], axis=1)
    traffic_train, traffic_test, target_train, target_test = train_test_split(traffic_,
                                                                              target,
                                                                              test_size=0.1,
                                                                              random_state=1011)
    X_train, X_val, y_train, y_val = train_test_split(traffic_train,
                                                      target_train,
                                                      test_size=0.2,
                                                      random_state=1011)
    write_to_csv(X_train, X_val, traffic_test, y_train, y_val, target_test)

def write_to_csv(X_train, X_val, X_test, y_train, y_val, y_test):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :param y_train:
    :param y_val:
    :param y_test:
    :return:
    """
    X_train = pd.concat([X_train, pd.DataFrame(y_train, columns=['class'])], axis=1)
    X_val = pd.concat([X_val, pd.DataFrame(y_val, columns=['class'])], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(y_test, columns=['class'])], axis=1)
    print('Write Data')
    X_train.to_csv("../../data_split/train.csv", index=False)
    X_val.to_csv("../../data_split/val.csv", index=False)
    X_test.to_csv("../../data_split/test.csv", index=False)
    print('Write Data : Success')

if __name__ == '__main__':
    load()
