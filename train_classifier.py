import warnings
import json
warnings.filterwarnings('ignore')
import lightgbm as lgb
import argparse
from utils.load_dataset import load_data

# парсинг аргументов из командной строки
parser = argparse.ArgumentParser(description='Classificator arguments')
parser.add_argument('--dir-path', help='Path to dir with traffic', default='data/Danmini_Doorbell')
parser.add_argument('--boosting-type', help='Type of model boosting', default='gbdt')
parser.add_argument('--num-leaves', help='number of leaves', type=int, default=31)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.05)
parser.add_argument('--num-boost-round', help='Number of boosting rounds', type=int, default=50)
parser.add_argument('--early-stopping', help='Number of eraly stopping rounds', type=int, default=5)
parser.add_argument('--feature-fraction', type=float, default=0.9)
parser.add_argument('--bagging-fraction', type=float, default=0.8)
parser.add_argument('--bagging-freq', type=int, default=5)
parser.add_argument('--verbose', help='logging', type=int, default=1)

# функция для обучения модели
def train_model(x_train, x_val, y_train, y_val, model_params_, train_params_):
    """

    :param x_train: train data
    :param x_val: val data
    :param y_train: train labels
    :param y_val: val labels
    :param model_params_: dict of parameters for lgbm
    :param train_params_: num_boost_rounds and early stopping rounds
    :return: trained Booster model
    """
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    print('Start training...')
    gbm = lgb.train(model_params_,
                    lgb_train,
                    num_boost_round=train_params_['num_boost_round'],
                    valid_sets=lgb_eval,
                    early_stopping_rounds=train_params_['early_stopping_rounds'])
    return gbm

# функция для записи модели в файл
def save_model(model_, path):
    """

    :param model_: trained model
    :param path: path for saving model
    :return: None
    """
    model_.save_model(path)


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_dir = args.dir_path if args.dir_path[-1] == '/' else args.dir_path + '/'
    # загрузка обучающей и валидационной выборки
    Xtrain, ytrain = load_data(path_to_dir + 'data_split/train.csv')
    Xval, yval = load_data(path_to_dir + 'data_split/val.csv')
    # загрузка файла с метками и соответствующими им классами
    with open(path_to_dir + 'data_split/attacks_lable.json') as file:
        attacks = json.load(file)
    # определение гиперпараметров модели
    model_params = dict(boosting_type=args.boosting_type, objective='multiclass', num_class=len(attacks.values()),
                        num_leaves=args.num_leaves, learning_rate=args.learning_rate,
                        feature_fraction=args.feature_fraction, bagging_fraction=args.bagging_fraction,
                        bagging_freq=args.bagging_freq, verbose=args.verbose)
    train_params = dict(num_boost_round=args.num_boost_round, early_stopping_rounds=args.early_stopping)
    # обучение модели
    model = train_model(Xtrain, Xval, ytrain, yval, model_params, train_params)
    print('Saving model...')
    # сохранение модели
    save_model(model, path_to_dir + 'model/booster.txt')
