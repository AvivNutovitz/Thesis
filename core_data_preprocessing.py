import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.datasets import cifar10, cifar100
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups, fetch_kddcup99
import random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
random.seed(42)


class Preprocessing():
    """
    this class is incharge of the preprocess of the data before it goes into the model

    links from were the data have been loaded:

    main link: https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/

    image_1: http://www.cs.toronto.edu/~kriz/cifar.html

    test_1: https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/
    """

    def __init__(self, conf, on_server):
        # dp = data_preprocessing
        self.dp_conf = conf['dp_conf']
        if not on_server:
            self.data_folder = 'My Drive/Lambda - Aviv Irad/Data/'
        else:
            self.data_folder = '{}/data/'.format(os.getcwd())
        self.on_server = on_server

    def build(self, data_type):
        self.data_type = data_type
        if data_type == 'base_text':
            return self._text_data()
        elif data_type == 'base_image':
            return self._image_data()
        elif data_type == 'base_tabular':
            return self._tabular_data()

    # def _download_and_extract_form_drive(self, file_name, extract=True):
    #     import tarfile
    #     from google.colab import drive
    #
    #     drive.mount('/content/drive/')
    #     full_path = "/content/drive/{}{}".format(self.data_folder, file_name)
    #     if extract:
    #         tar = tarfile.open(full_path)
    #         tar.extractall()
    #         tar.close()
    #
    #     return full_path

    def _extract_form_sever(self, file_name, extract=True):
        import tarfile
        full_path = '{}{}'.format(self.data_folder, file_name)
        if extract:
            tar = tarfile.open(full_path)
            tar.extractall()
            tar.close()
        return full_path

    def _build_embeding_matrix(self, word_index):
        embeddings_index = {}
        # or glove.6B.100d.txt
        file_name = self._extract_form_sever('glove.6B.50d.txt', extract=False)

        f = open(os.path.join(file_name), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Total %s word vectors in Glove 6B 50d.' % len(embeddings_index))

        embedding_matrix = np.random.random((len(word_index) + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def _text_data(self):

        all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                          'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                          'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
                          'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                          'talk.politics.misc', 'talk.religion.misc']
        run_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                          'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles',
                          'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med']
        # download
        newsgroups_train = fetch_20newsgroups(subset='train', categories=run_categories, shuffle=True)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=run_categories, shuffle=True)

        # read config
        text_conf = self.dp_conf['text_conf']
        self.MAX_NB_WORDS = text_conf['MAX_NB_WORDS']
        self.MAX_SEQUENCE_LENGTH = text_conf['MAX_SEQUENCE_LENGTH']
        self.EMBEDDING_DIM = text_conf['EMBEDDING_DIM']
        self.HIDDEN_SIZE = text_conf['HIDDEN_SIZE']
        self.cold_start_size = 2000

        # build corpus
        self.train_texts = newsgroups_train.data
        self.train_labels = list(newsgroups_train.target)
        self.test_texts = newsgroups_test.data
        self.test_labels = list(newsgroups_test.target)

        tokenizer = Tokenizer(nb_words=self.MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.train_texts)
        train_sequences = tokenizer.texts_to_sequences(self.train_texts)
        test_sequences = tokenizer.texts_to_sequences(self.test_texts)
        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))

        train_data = pad_sequences(train_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        train_labels = to_categorical(np.asarray(self.train_labels))

        test_data = pad_sequences(test_sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        test_labels = to_categorical(np.asarray(self.test_labels))

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        x_train = train_data[:self.cold_start_size]
        y_train = train_labels[:self.cold_start_size]
        print('Shape of train data tensor:', x_train.shape)
        print('Shape of train label tensor:', y_train.shape)

        x_test = np.concatenate((test_data, train_data[self.cold_start_size:]))
        y_test = np.concatenate((test_labels, train_labels[self.cold_start_size:]))
        print('Shape of test data tensor:', x_test.shape)
        print('Shape of test label tensor:', y_test.shape)

        embedding_matrix = self._build_embeding_matrix(word_index)
        return x_train, y_train, x_test, y_test, embedding_matrix, word_index

    def _image_data(self):

        self.cold_start_size = 2000
        self.num_classes = 100

        (train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

        x_train = train_data[:self.cold_start_size]
        y_train = train_labels[:self.cold_start_size]
        print('Shape of train data tensor:', x_train.shape)
        print('Shape of train label tensor:', y_train.shape)

        x_test = np.concatenate((test_data, train_data[self.cold_start_size:]))
        y_test = np.concatenate((test_labels, train_labels[self.cold_start_size:]))
        print('Shape of test data tensor:', x_test.shape)
        print('Shape of test label tensor:', y_test.shape)

        # Convert class vectors to binary class matrices.
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # plot first few images
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # plot raw pixel data
            pyplot.imshow(x_train[i])
        pyplot.show()

        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # plot raw pixel data
            pyplot.imshow(x_test[i])
        # show the figure
        pyplot.show()

        return x_train, y_train, x_test, y_test, self.num_classes

    def _tabular_data(self):

        kddcup99_all_data = fetch_kddcup99()
        feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                         'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                         'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                         'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                         'count',
                         'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                         'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                         'dst_host_same_srv_rate',
                         'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                         'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                         'dst_host_srv_rerror_rate']
        tabular_data_set = pd.DataFrame.from_dict(kddcup99_all_data['data'])
        tabular_data_set.columns = feature_names

        tabular_data_set = tabular_data_set.drop_duplicates(subset=feature_names, keep='first', inplace=True)
        traget = kddcup99_all_data['target']

        new_tabular_data_set, traget = self._sub_sampling(tabular_data_set, traget)
        list_of_columns = ['protocol_type', 'service', 'flag']
        X = self._set_tabular_df(data_to_work_on=new_tabular_data_set, list_of_columns=list_of_columns)
        return self._split_train_and_test_tabular_data(X, *self._create_value_mapping_for_target_tabular_data(traget))

    def _set_tabular_df(self, data_to_work_on, list_of_columns):
        list_of_dfs = []

        for cat_col in list_of_columns:
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(data_to_work_on[cat_col])
            column_names = ['{}_{}'.format(cat_col, col) for col in list(set(data_to_work_on[cat_col]))]
            tmp_df = pd.DataFrame(to_categorical(integer_encoded))
            tmp_df.columns = column_names
            tmp_df = tmp_df.reset_index()
            list_of_dfs.append(tmp_df)
            data_to_work_on = data_to_work_on.drop([cat_col], axis=1)

        data_to_work_on = data_to_work_on.reset_index()
        list_of_dfs.append(data_to_work_on)
        full_df = pd.concat(list_of_dfs, axis=1, ignore_index=True)
        return full_df

    def _create_value_mapping_for_target_tabular_data(self, traget):
        map_kddcup99_target_to_int = {value: key for key, value in enumerate(set(pd.Series(traget)))}
        tabular_label = [map_kddcup99_target_to_int[value] for value in list(pd.Series(traget))]
        return to_categorical(tabular_label), len(set(pd.Series(traget)))

    def _split_train_and_test_tabular_data(self, X, y, num_classes):

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
        print('Shape of train data tensor:', x_train.shape)
        print('Shape of train label tensor:', y_train.shape)
        print('Shape of test data tensor:', x_test.shape)
        print('Shape of test label tensor:', y_test.shape)
        print('Total number of classes:', num_classes)
        return x_train, y_train, x_test, y_test, num_classes

    def _sub_sampling(self, tabular_data_set, traget):
        print('Applying sub_sampling, total size Before sub_sampling: ', tabular_data_set.shape)
        values_to_sub_sample = [b'smurf.', b'neptune.', b'normal.']
        tabular_data_set['traget'] = traget
        tabular_data_set_base = tabular_data_set.loc[~tabular_data_set['traget'].isin(values_to_sub_sample)]
        tabular_data_set_to_sample = tabular_data_set.loc[tabular_data_set['traget'].isin(values_to_sub_sample)]
        dfs = []
        for val in values_to_sub_sample:
            dfs.append(self._sub_sample(tabular_data_set_to_sample, 'traget', val))

        dfs.append(tabular_data_set_base)
        full = pd.concat(dfs)

        y = full['traget']
        df = full.drop(columns=['traget'])

        print('Applying sub_sampling, total size After sub_sampling: ', df.shape)
        return df, y

    def _sub_sample(self, df, col, val, frac=0.09):
        return df[df[col] == val].sample(frac=frac, random_state=42)
