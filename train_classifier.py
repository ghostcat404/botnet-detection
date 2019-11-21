import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import argparse
from utils.load_dataset import load_data
#TODO: дописать доку ко всем функциям


parser = argparse.ArgumentParser(description='Classificator arguments')
parser.add_argument('--train-manifest', help='Path to train manifest csv', default='data_split/train.csv')
parser.add_argument('--val-manifest', help='Path to validation manifest csv', default='data_split/val.csv')
parser.add_argument('--save-model-path', help='Path where model will save', default='save_models/best_model.txt')
parser.add_argument('--boosting-type', help='Type of model boosting', default='gbdt')
parser.add_argument('--num-leaves', help='number of leaves', type=int, default=31)
parser.add_argument('--num-class', help='Number of classes in dataset', type=int, default=6)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.05)
parser.add_argument('--num-boost-round', help='Number of boosting rounds', type=int, default=50)
parser.add_argument('--early-stopping', help='Number of eraly stopping rounds', type=int, default=5)
parser.add_argument('--feature-fraction', type=float, default=0.9)
parser.add_argument('--bagging-fraction', type=float, default=0.8)
parser.add_argument('--bagging-freq', type=int, default=5)
parser.add_argument('--verbose', help='logging', type=int, default=1)

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

def save_model(model_, path):
    """

    :param model_: trained model
    :param path: path for saving model
    :return: None
    """
    model_.save_model(path)

if __name__ == '__main__':
    args = parser.parse_args()
    Xtrain, ytrain = load_data(args.train_manifest)
    Xval, yval = load_data(args.val_manifest)

    model_params = {
        'boosting_type': args.boosting_type,
        'objective': 'multiclass',
        'num_class': args.num_class,
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'verbose': args.verbose
    }
    train_params = {
        'num_boost_round': args.num_boost_round,
        'early_stopping_rounds': args.early_stopping
    }
    model = train_model(Xtrain, Xval, ytrain, yval, model_params, train_params)

    save_model(model, args.save_model_path)
