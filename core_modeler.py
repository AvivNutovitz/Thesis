from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import random
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, Embedding, Dropout, SpatialDropout1D, LSTM, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.optimizers import  Adam
random.seed(42)


class Modeler():
    """
    this class is incharge of the deep learning model
    """

    def __init__(self, m_conf):
        self.m_conf = m_conf

    def build(self, data_type, *args):

        if data_type == 'base_text':
            model = self.create_text_model(*args)

        elif data_type == 'base_image':
            model = self.create_image_model_vgg(*args)

        elif data_type == 'base_tabular':
            model = self.create_tabular_model(*args)

        print("model fitting - simplified convolutional neural network")
        print(model.summary())
        return model

    def create_text_model(self, len_word_index, text_conf, embedding_matrix):
        embedding_layer = Embedding(len_word_index, text_conf['EMBEDDING_DIM'],
                                    weights=[embedding_matrix],
                                    input_length=text_conf['MAX_SEQUENCE_LENGTH'],
                                    trainable=True)

        sequence_input = Input(shape=(text_conf['MAX_SEQUENCE_LENGTH'],), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        l_cov1 = Conv1D(64, 5, activation='relu')(embedded_sequences)
        l_sd1 = SpatialDropout1D(0.5)(l_cov1)
        l_lstm_1 = LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(l_sd1)
        l_sd2 = SpatialDropout1D(0.5)(l_lstm_1)
        l_lstm_2 = LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(l_sd2)
        l_td1 = TimeDistributed(Dense(len_word_index))(l_lstm_2)
        l_flat = Flatten()(l_td1)
        preds = Dense(12, activation='softmax')(l_flat)

        model = Model(sequence_input, preds)
        adam = Adam(lr=0.001, clipnorm=5.)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy', self.f1])
        return model

    def create_image_model_vgg(self, input_shape, num_classes):

        input_tensor = Input(shape=input_shape)
        # this assumes K.image_data_format() == 'channels_last' (32, 32, 3)
        base_model = VGG16(input_tensor=input_tensor, weights=None, include_top=False, classes=num_classes)

        x = base_model.output
        x = Dropout(0.5)(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        adam = Adam(lr=0.0005, clipnorm=5.)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy', self.f1])
        return model

    def create_tabular_model(self, input_shape, num_classes):
        print(input_shape)
        model = Sequential()
        model.add(Dense(1000, activation="relu", input_shape=(input_shape,)))
        model.add(Dropout(0.5))
        model.add(Dense(2000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(num_classes))
        adam = Adam(lr=0.0005, clipnorm=5.)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy', self.f1])
        return model

    @staticmethod
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))