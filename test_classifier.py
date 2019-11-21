import warnings
import argparse
from utils.load_dataset import load_data
warnings.filterwarnings('ignore')
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


parser = argparse.ArgumentParser(description='Classificator arguments')
parser.add_argument('--test-manifest', help='Path to train manifest csv', default='data_split/test.csv')
parser.add_argument('--model-path', help='Path to trained model', default='models/best_booster.txt')
parser.add_argument('--num-features-importance', help='Number of max features for plot', type=int, default=5)

def predict_(model_, x_test):
    """

    :param model_:
    :param x_test:
    :return:
    """
    predictions_ = []
    predict = model_.predict(x_test)
    for element in predict:
        predictions_.append(np.argmax(element))
    return predictions_

def test_results(y_test, predictions_):
    """

    :param y_test:
    :param predictions_:
    :return:
    """
    print(f'\nAccuracy : {accuracy_score(y_test, predictions) * 100.:2f}%\n')
    print(f'traffic_analysis report : {classification_report(y_test, predictions)}')
    plot_confusion_matrix(y_test, predictions_)

def plot_confusion_matrix(y_test, predictions_):
    """

    :param y_test:
    :param predictions_:
    :return:
    """
    cm = confusion_matrix(y_test, predictions_)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=(0, 6), yticklabels=(0, 6),
           title='Confusion matrix, without normalization',
           ylabel='True label',
           xlabel='Predicted label')
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('plots/confusion_matrix.png')

def plot_feature_importance(model_, num_features):
    """

    :param model_:
    :param num_features:
    :return:
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    lgb.plot_importance(model_, ax=ax, max_num_features=num_features)
    fig.savefig('plots/feature_importance.png')

if __name__ == '__main__':
    args = parser.parse_args()
    model = lgb.Booster(model_file=args.model_path)
    Xtest, ytest = load_data(args.test_manifest)
    predictions = predict_(model, Xtest)
    test_results(ytest, predictions)
    plot_feature_importance(model, args.num_features_importance)
