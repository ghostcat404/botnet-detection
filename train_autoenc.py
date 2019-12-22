from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import optimizers
import argparse

# парсинг аршументов из командной строки
parser = argparse.ArgumentParser(description='Autoencoder arguments')
parser.add_argument('--dir-path', help='Path to dir with traffic', default='data/Danmini_Doorbell')
parser.add_argument('--epochs', help='Number epochs', type=int,  default=40)
parser.add_argument('--lr', help='Learning Rate', type=float,  default=1.0)
parser.add_argument('--fisher', help='Top features', type=int,  default=0)

# функция рассчета границы
def calc_threshold(autoencoder, Xval, yval):
    print('Calc Threshold')
    ypredval = autoencoder.predict(Xval)
    n = len(Xval)
    l = []
    err_benign = 0
    err_mal = 0
    count_benign = 0
    count_mal = 0
    for i in range(n):
        temp = metrics.mean_squared_error(Xval[i], ypredval[i]) ** 0.5
        if yval[i] == 0:
            err_benign += temp
            count_benign += 1
        else:
            err_mal += temp
            count_mal += 1
        l.append((temp, yval[i]))
    print(err_benign / count_benign, err_mal / count_mal)

    rmsedf = pd.DataFrame.from_records(l, columns=['rmse', 'class'])
    rmsedf.head()

    rdf = rmsedf[rmsedf['class'] == 0]
    threshold = rdf['rmse'].mean() + rdf['rmse'].std()
    print("threshold", threshold)
    return threshold, rmsedf

# функция построения графика распределения 
def plot_traffic_distribution(rmsedf,
                              threshold,
                              dir_path,
                              class_='benign',
                              fisher=False):
    df2 = pd.DataFrame()
    if class_ == 'benign':
        df2 = rmsedf[rmsedf['class'] == 0]
    elif class_ == 'malicious':
        df2 = rmsedf[rmsedf['class'] == 1]
    k = range(1, len(df2) + 1)
    fig, ax = plt.subplots()
    plt.scatter(range(1, len(df2) + 1), df2['rmse'])
    plt.plot(k, [threshold for _ in k], "r")
    plt.xlabel("Номер пакета трафика")
    plt.ylabel("RMSE")
    plt.legend(["Граница", "Значение RMSE"])
    # plt.show()

    if not fisher:
        fig.savefig(dir_path + f'plots/{class_}_distribution.png')
    else:
        fig.savefig(dir_path + f'plots/{class_}_distribution_fisher{fisher}.png')

# функция построения матрицы несоответствий
def plot_conf_matrix(model, Xtest, ytest, threshold, dir_path, fisher=False):
    ypredtest = model.predict(Xtest)
    print(np.shape(ypredtest))
    l = []
    n = len(Xtest)
    for i in range(n):
        temp = metrics.mean_squared_error(Xtest[i], ypredtest[i]) ** 0.5
        l.append(temp)

    print(np.shape(l))
    n = len(Xtest)

    for i in range(n):
        if l[i] < threshold:
            l[i] = 0
        else:
            l[i] = 1
    cm = metrics.confusion_matrix(ytest, l)
    tp, fn, fp, tn = cm.ravel()
    FRR = fn / (tp + fn)  # ошибка 1 рода
    FAR = fp / (fp + tn)  # ошибка 2 рода
    # сохранение ошибок 1 и 2 рода для каждого устройства
    with open('errors.txt', 'a') as er:
        if not fisher:
            er.write(str(dir_path.split('/')[1])+','+str(115)+','+str(round(FRR * 100, 2))+','+str(round(FAR * 100, 2))+'\n')
        else:
            er.write(str(dir_path.split('/')[1])+','+str(fisher)+','+str(round(FRR * 100, 2))+','+str(round(FAR * 100, 2))+'\n')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=(0, 1), yticklabels=(0, 1),
           ylabel='Действительные значения',
           xlabel='Предсказанные значения')
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if not fisher:
        fig.savefig(dir_path + 'plots/confusion_matrix.png')
    else:
        fig.savefig(dir_path + f'plots/confusion_matrix_fisher{fisher}.png')
    print(metrics.classification_report(ytest, l))

# функция построения графика значения ошибки в зависимости от эпохи
def plot_loss(model, dir_path):
    x = range(model.history.params['epochs'])
    loss_train = model.history.history['loss']
    loss_val = model.history.history['val_loss']
    fig, ax = plt.subplots()
    plt.plot(x, loss_train, loss_val)
    plt.xlabel("Эпоха")
    plt.ylabel("Значение RMSE")
    plt.legend(["RMSE на обучающей выборке", "RMSE на валидационной выборке"])
    fig.savefig(dir_path + 'plots/loss.png')


if __name__ == '__main__':
    args = parser.parse_args()
    path_to_dir = args.dir_path if args.dir_path[-1] == '/' else args.dir_path + '/'

    # Загрузка данных
    fishers_cols = ['MI_dir_L0.01_weight', 'H_L0.01_weight', 'MI_dir_L0.01_mean', 'H_L0.01_mean', 'MI_dir_L0.1_mean']
    benign = pd.read_csv(path_to_dir + 'benign_traffic.csv')#[fishers_cols]
    benign['class'] = 0
    g_combo = pd.read_csv(path_to_dir + 'gafgyt_attacks/combo.csv')#[fishers_cols]
    g_combo['class'] = 1
    g_junk = pd.read_csv(path_to_dir + 'gafgyt_attacks/junk.csv')#[fishers_cols]
    g_junk['class'] = 1
    g_scan = pd.read_csv(path_to_dir + 'gafgyt_attacks/scan.csv')#[fishers_cols]
    g_scan['class'] = 1
    g_tcp = pd.read_csv(path_to_dir + 'gafgyt_attacks/tcp.csv')#[fishers_cols]
    g_tcp['class'] = 1
    g_udp = pd.read_csv(path_to_dir + 'gafgyt_attacks/udp.csv')#[fishers_cols]
    g_udp['class'] = 1
    if os.path.exists(path_to_dir + 'mirai_attacks/'):
        m_ack = pd.read_csv(path_to_dir + 'mirai_attacks/ack.csv')#[fishers_cols]
        m_ack['class'] = 1
        m_scan = pd.read_csv(path_to_dir + 'mirai_attacks/scan.csv')#[fishers_cols]
        m_scan['class'] = 1
        m_syn = pd.read_csv(path_to_dir + 'mirai_attacks/syn.csv')#[fishers_cols]
        m_syn['class'] = 1
        m_udp = pd.read_csv(path_to_dir + 'mirai_attacks/udp.csv')#[fishers_cols]
        m_udp['class'] = 1
        m_udpplain = pd.read_csv(path_to_dir + 'mirai_attacks/udpplain.csv')
        m_udpplain['class'] = 1
        print("Data extraction : Success")
        malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp, m_ack, m_scan, m_syn, m_udp, m_udpplain])
    else:
        print("Data extraction : Success")
        malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp])

    # разделение на обучающую, валидационную и тестовую выборки
    benign = shuffle(benign)
    malicious = shuffle(malicious)

    n = len(benign)
    benignVal = benign[:int(n / 6)]
    benignTest = benign[int(5 * n / 6):]
    benignTrain = benign[int(n / 6):int(5 * n / 6)]

    n = len(malicious)
    maliciousVal = malicious[:int(n / 2)]
    maliciousTest = malicious[int(n / 2):]

    dataTest = pd.concat([benignTest, maliciousTest])
    dataVal = pd.concat([benignVal, maliciousVal])

    print("Data Randomized")
    # выбор количества признаков
    if args.fisher:
        fisher = pd.read_csv('fisher.csv')
        l = fisher['Feature'].loc[:args.fisher-1]
        print(f'Use top {args.fisher} features')
    else:
        l = list(benignTrain)
        l.remove('class')

    X = benignTrain[l]
    y = benignTrain['class']
    X = normalize(X)

    Xval = dataVal[l]
    yval = dataVal['class'].values
    Xval = normalize(Xval)

    XTest = dataTest[l]
    yTest = dataTest['class'].values
    XTest = normalize(XTest)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    model_path = path_to_dir + f'model/anomaly_detector_fisher{args.fisher}.h5' if args.fisher else path_to_dir + 'model/anomaly_detector.h5'
    # обучение или загрузка сохраненной модели, если такая имеется
    if not os.path.exists(model_path):
        # выбор архитектуры сети в зависимости от количества признаков
        if not args.fisher:
            input_img = Input(shape=(115,))
            # "encoded" is the encoded representation of the input
            encoded = Dense(int(0.75 * 115), activation='tanh')(input_img)
            encoded = Dense(int(0.5 * 115), activation='tanh')(encoded)
            encoded = Dense(int(0.33 * 115), activation='tanh')(encoded)

            # # "decoded" is the lossy reconstruction of the input
            decoded = Dense(int(0.25 * 115), activation='tanh')(encoded)
            decoded = Dense(int(0.33 * 115), activation='tanh')(decoded)
            decoded = Dense(int(0.5 * 115), activation='tanh')(decoded)
            decoded = Dense(int(0.75 * 115), activation='tanh')(decoded)
            decoded = Dense(115)(decoded)
        else:
            input_img = Input(shape=(args.fisher,))
            # "encoded" is the encoded representation of the input
            encoded = Dense(int(0.7 * args.fisher), activation='tanh')(input_img)
            encoded = Dense(int(0.2 * args.fisher), activation='tanh')(encoded)

            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(int(0.7 * args.fisher), activation='tanh')(encoded)
            decoded = Dense(args.fisher, activation='sigmoid')(decoded)

            # this model maps an input to its reconstruction
        model = Model(input_img, decoded)
        adadelta = optimizers.Adadelta(learning_rate=args.lr, rho=0.95)
        model.compile(optimizer=adadelta, loss='mean_squared_error')
        print('Model was compile')
        print('Train model')
        # Обучение модели
        model.fit(X_train, X_train,
                  epochs=args.epochs,
                  batch_size=1024,
                  shuffle=True,
                  validation_data=(X_test, X_test))
        print('Saving model')
        # сохранение модели
        model.save(model_path)
    else:
        print('Load model')
        model = load_model(model_path)
    # Вычисление границы и построение графиков
    threshold, error_df = calc_threshold(model, Xval, yval)
    print('Saving plots')
    plot_traffic_distribution(error_df, threshold, path_to_dir, 'benign', args.fisher)
    plot_traffic_distribution(error_df, threshold, path_to_dir, 'malicious', args.fisher)
    plot_conf_matrix(model, XTest, yTest, threshold, path_to_dir, args.fisher)
    plot_loss(model, path_to_dir)
