import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize

class Data_csv:
    def __init__(self):
        self.preprocess_data()

    def preprocess_data(self):
        benign = pd.read_csv('data/benign/benign_traffic.csv')
        benign['class'] = 0
        g_combo = pd.read_csv('data/gafgyt_attacks/combo.csv')
        g_combo['class'] = 1
        g_junk = pd.read_csv('data/gafgyt_attacks/junk.csv')
        g_junk['class'] = 1
        g_scan = pd.read_csv('data/gafgyt_attacks/scan.csv')
        g_scan['class'] = 1
        g_tcp = pd.read_csv('data/gafgyt_attacks/tcp.csv')
        g_tcp['class'] = 1
        g_udp = pd.read_csv('data/gafgyt_attacks/udp.csv')
        g_udp['class'] = 1
        m_ack = pd.read_csv('data/mirai_attacks/ack.csv')
        m_ack['class'] = 1
        m_scan = pd.read_csv('data/mirai_attacks/scan.csv')
        m_scan['class'] = 1
        m_syn = pd.read_csv('data/mirai_attacks/syn.csv')
        m_syn['class'] = 1
        m_udp = pd.read_csv('data/mirai_attacks/udp.csv')
        m_udp['class'] = 1
        m_udpplain = pd.read_csv('data/mirai_attacks/udpplain.csv')
        m_udpplain['class'] = 1

        print("Data extraction : Success")

        malicious = pd.concat([g_combo, g_junk, g_scan, g_tcp, g_udp, m_ack, m_scan, m_syn, m_udp, m_udpplain])

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
        l = list(benignTrain)
        l.remove('class')

        X = benignTrain[l]
        y = benignTrain['class']
        X = normalize(X)

        Xval = dataVal[l]
        self.yval = dataVal['class']
        self.Xval = normalize(Xval)

        XTest = dataTest[l]
        self.yTest = dataTest['class']
        self.XTest = normalize(XTest)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)