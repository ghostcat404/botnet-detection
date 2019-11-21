# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input, Dense
from keras.models import Model


class Autoencoder:
    def __init__(self):
        # # this is the size of our encoded representations
        # encoding_dim = 128  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        # this is our input placeholder
        input_img = Input(shape=(115,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(64, activation='relu')(input_img)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(10, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(115, activation='sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.model = Model(input_img, decoded)
        self.model.compile(optimizer='adadelta', loss='mean_squared_error')
        print('Model was compile')

    def train_model(self, X_train, X_test):
        """

        :param X_train:
        :param X_test:
        :return:
        """
        self.model.fit(X_train, X_train,
                       epochs=100,
                       batch_size=1024,
                       shuffle=True,
                       validation_data=(X_test, X_test))

        return self.model