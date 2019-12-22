from keras.layers import Input, Dense
from keras.models import Model, load_model


class Autoencoder:
    def __init__(self):
        input_img = Input(shape=(115,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(64, activation='relu')(input_img)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(5, activation='relu')(encoded)
        decoded = Dense(10, activation='relu')(decoded)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(115, activation='sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.model = Model(input_img, decoded)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print('Model was compile')

    def train_model(self, X_train, X_test):
        """

        :param X_train: train data
        :param X_test: test data
        :return: trained model
        """
        self.model.fit(X_train, X_train,
                       epochs=100,
                       batch_size=1024,
                       shuffle=True,
                       validation_data=(X_test, X_test))

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self, path_to_model):
        self.model = load_model(path_to_model)
