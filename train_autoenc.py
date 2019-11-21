from model.autoencoder import Autoencoder
from utils.data_set import Data_csv


if __name__ == '__main__':
    data = Data_csv()
    model = Autoencoder()
    model.train_model(data.X_train, data.X_test)
    model.model.save('save_models/anomaly_detector.h5')
