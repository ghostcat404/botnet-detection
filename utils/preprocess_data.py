import os
import json
import argparse
import pandas as pd
import sklearn.utils as util
from termcolor import colored
from sklearn.model_selection import train_test_split

# Парсинг аргументов из командной строки
parser = argparse.ArgumentParser(description='Classificator arguments')
parser.add_argument('--dir-path', help='Path to dir with traffic', default='data/Danmini_Doorbell')


# Загрузка данных, присвоение меток
def load(dir_path):
    """

    :return: None
    """
    attacks = {}
    print('Start data extraction')
    benign = pd.read_csv(dir_path + 'benign_traffic.csv')
    benign['class'] = 0
    attacks[0] = 'benign'
    g_combo = pd.read_csv(dir_path + 'gafgyt_attacks/combo.csv')
    g_combo['class'] = 1
    attacks[1] = 'bashlite_combo'
    g_junk = pd.read_csv(dir_path + 'gafgyt_attacks/junk.csv')
    g_junk['class'] = 2
    attacks[2] = 'bashlite_junk'
    g_scan = pd.read_csv(dir_path + 'gafgyt_attacks/scan.csv')
    g_scan['class'] = 3
    attacks[3] = 'bashlite_scan'
    g_tcp = pd.read_csv(dir_path + 'gafgyt_attacks/tcp.csv')
    g_tcp['class'] = 4
    attacks[4] = 'bashlite_tcp'
    g_udp = pd.read_csv(dir_path + 'gafgyt_attacks/udp.csv')
    g_udp['class'] = 5
    attacks[5] = 'bashlite_udp'
    if os.path.exists(dir_path + 'mirai_attacks/'):
        m_ack = pd.read_csv(dir_path + 'mirai_attacks/ack.csv')
        m_ack['class'] = 6
        attacks[6] = 'mirai_ack'
        m_scan = pd.read_csv(dir_path + 'mirai_attacks/scan.csv')
        m_scan['class'] = 7
        attacks[7] = 'mirai_scan'
        m_syn = pd.read_csv(dir_path + 'mirai_attacks/syn.csv')
        m_syn['class'] = 8
        attacks[8] = 'mirai_syn'
        m_udp = pd.read_csv(dir_path + 'mirai_attacks/udp.csv')
        m_udp['class'] = 9
        attacks[9] = 'mirai_udp'
        m_udpplain = pd.read_csv(dir_path + 'mirai_attacks/udpplain.csv')
        m_udpplain['class'] = 10
        attacks[10] = 'mirai_udpplain'
        print(colored("Success\n", 'green'))
        malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp, m_ack, m_scan, m_syn, m_udp, m_udpplain])
    else:
        print(colored("Success\n", 'green'))
        malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp])
    with open(dir_path + 'data_split/attacks_lable.json', 'w') as outfile:
        json.dump(attacks, outfile)

    print('Start data preprocessing')
    benign = util.shuffle(benign)
    malicious = util.shuffle(malicious)
    traffic_ = pd.concat([benign, malicious])
    traffic_ = util.shuffle(traffic_)
    print(colored('Success\n', 'green'))

    split_data(traffic_, dir_path)

# Разбиаение данных дла обучающую, валидационную и тестовую выборки
def split_data(traffic_, dir_path):
    """

    :param traffic_: DataFrame which contains traffic features
    :return: None
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
    write_to_csv(X_train, X_val, traffic_test, y_train, y_val, target_test, dir_path)

# Сохранение выборок
def write_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, dir_path):
    """

    :param X_train: train dataset
    :param X_val: val dataset
    :param X_test: test dataset
    :param y_train: train labels
    :param y_val: val labels
    :param y_test: test labels
    :return: None
    """
    X_train = pd.concat([X_train, pd.DataFrame(y_train, columns=['class'])], axis=1)
    X_val = pd.concat([X_val, pd.DataFrame(y_val, columns=['class'])], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(y_test, columns=['class'])], axis=1)
    print('Write Data')
    X_train.to_csv(dir_path + "data_split/train.csv", index=False)
    X_val.to_csv(dir_path + "data_split/val.csv", index=False)
    X_test.to_csv(dir_path + "data_split/test.csv", index=False)
    print(colored('Success', 'green'))


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_dir = args.dir_path if args.dir_path[-1] == '/' else args.dir_path + '/'
    load(path_to_dir)
