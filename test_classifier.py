import json
import warnings
import argparse
from utils.load_dataset import load_data

warnings.filterwarnings('ignore')
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# парсинг аргументов из командной строки
parser = argparse.ArgumentParser(description='Classificator arguments')
parser.add_argument('--dir-path', help='Path to dir with traffic', default='data/Danmini_Doorbell')
parser.add_argument('--num-features-importance', help='Number of max features for plot', type=int, default=5)

# функция для формирования вектора предсказаний
def predict_(model_, x_test):
    """

    :param model_: trained lgbm model
    :param x_test: test data
    :return: list of predictions for sample of feature
    """
    predictions_ = []
    predict = model_.predict(x_test)
    for element in predict:
        predictions_.append(np.argmax(element))
    return predictions_

# функция для вычисления точности и записи метрик в файл
def test_results(y_test, predictions_, dir_path):
    """

    :param y_test: true labels
    :param predictions_: predicted labels
    :return: None
    """
    accuracy = round(accuracy_score(y_test, predictions), 5)
    report = classification_report(y_test, predictions)
    with open('booster_results.txt', 'a') as br:
        br.write(str(dir_path.split('/')[1]) + ',' + str(accuracy) + '\n')
    with open('classification_report.txt', 'a') as cr:
        cr.write(str(dir_path.split('/')[1] + '\n' + report + '\n'))
    print(f'\nAccuracy : {accuracy}\n')
    print(f'traffic_analysis report : \n{report}')
    plot_confusion_matrix(y_test, predictions_, dir_path)

# функция для построения матрицы несоответствий
def plot_confusion_matrix(y_test, predictions_, dir_path):
    """

    :param y_test: true labels
    :param predictions_: predicted labels
    :return: None
    """
    with open(dir_path + 'data_split/attacks_lable.json', 'r') as file:
        attacks = json.load(file)
    cm = confusion_matrix(y_test, predictions_)
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(attacks.values()), yticklabels=list(attacks.values()),
           ylabel='Действительные значения',
           xlabel='Предсказанные значения')
    plt.setp(ax.get_xticklabels(), rotation=90)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(dir_path + 'plots/confusion_matrix_booster.png')

# функция для построения графика важности признаков модели
def plot_feature_importance(model_, num_features, dir_path):
    """

    :param model_: trained lgbm model
    :param num_features: Max number of features to plot
    :return: None
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    lgb.plot_importance(model_, ax=ax, max_num_features=num_features)
    fig.savefig(dir_path + 'plots/feature_importance_booster.png')


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_dir = args.dir_path if args.dir_path[-1] == '/' else args.dir_path + '/'
    # загрузка сохраненой модели
    model = lgb.Booster(model_file=path_to_dir + 'model/booster.txt')
    # загрузка тестовой выборки
    Xtest, ytest = load_data(path_to_dir + 'data_split/test.csv')
    # формирование вектора предсказаний
    predictions = predict_(model, Xtest)
    # построение матрицы несоответствий и графика важности признаков
    test_results(ytest, predictions, path_to_dir)
    plot_feature_importance(model, args.num_features_importance, path_to_dir)
