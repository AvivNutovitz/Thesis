import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import random
from keras.callbacks import CSVLogger, EarlyStopping
from datetime import datetime
from sklearn.model_selection import train_test_split
random.seed(42)


class ActiveLearning():
    """
    this class is incharge of the active learning flow
    """

    def __init__(self, conf, data_preprocesser, modeler, scorer, data_type):
        # al = active learning
        self.al_conf = conf['al_conf']
        self.data_preprocesser = data_preprocesser
        self.modeler = modeler
        self.data_type = data_type
        self.scorer = scorer
        self.stopping_criteria_train_ratio = self.al_conf.get('STOPPING_CRITERIA', {}).get('TRAIN_RATIO', 0.3)
        self.number_of_samples_in_one_loop = self.al_conf.get('NUMER_OF_SAMPELS_IN_ONE_LOOP', 300)
        self.epochs = self.al_conf.get('EPOCHS', 20)
        self.batch_size = self.al_conf.get('BATCH_SIZE', 100)
        self.validation_split = self.al_conf.get('VALIDATION_SPLIT', 0.3)
        self.output_folder = self.al_conf.get('OUTPUT_FOLDER', 'outputs')

    def setup(self):

        if self.data_type == 'base_text':

            # create data
            self.x_train, self.y_train, self.x_test, self.y_test, self.embedding_matrix, self.word_index = self.data_preprocesser.build(
                self.data_type)

            # create model
            self.model = self.modeler.build(self.data_type, len(self.word_index) + 1,
                                            self.data_preprocesser.dp_conf['text_conf'], self.embedding_matrix)

            # add word_index to the scorer
            self.scorer.word_index = self.word_index

        elif self.data_type == 'base_image':
            self.x_train, self.y_train, self.x_test, self.y_test, num_classes = self.data_preprocesser.build(self.data_type)

            self.model = self.modeler.build(self.data_type, self.x_train.shape[1:], num_classes)

        elif self.data_type == 'base_tabular':
            self.x_train, self.y_train, self.x_test, self.y_test, num_classes = self.data_preprocesser.build(self.data_type)

            self.model = self.modeler.build(self.data_type, self.x_train.shape[-1], num_classes)

        # set train, test ids data points tuples
        self.create_ids_first_time()

    def create_ids_first_time(self):
        all_X_data = np.concatenate((self.x_train, self.x_test))
        all_y_data = np.concatenate((self.y_train, self.y_test))

        self.knowns_data_points_ids = []
        self.knowns_data_points_tupples = []
        self.unknowns_data_points_ids = []
        self.unknowns_data_points_tupples = []

        for data_point_id, (x_data_point, y_data_point) in enumerate(zip(all_X_data, all_y_data)):
            if data_point_id < len(self.x_train):
                self.knowns_data_points_ids.append(data_point_id)
                self.knowns_data_points_tupples.append((data_point_id, x_data_point, y_data_point))
            else:
                self.unknowns_data_points_ids.append(data_point_id)
                self.unknowns_data_points_tupples.append((data_point_id, x_data_point, y_data_point))

    def get_number_of_information_points(self, list_of_shap_values_per_test_data_point, cutoff=0.001):
        # return the number of information_points regrds the cutoff
        list_of_results = []
        for shap_values in list_of_shap_values_per_test_data_point:
            # for every test data point shap values
            counter = 0
            for shap_value in list(shap_values):
                # keep only values that thier absolut value is grather then the cutoff
                if abs(shap_value) > cutoff:
                    counter += 1
            list_of_results.append(counter)
        return list_of_results

    def find_next_training_points(self, sorted_results_list):
        # return the indexes up to the number_of_exmples_to_return
        results_df = pd.DataFrame(columns=['sumple_index', 'confidence'])
        for (index, confidence) in sorted_results_list[0:self.number_of_samples_in_one_loop]:
            results_df.loc[index] = [index, confidence]
        results_df.to_csv(
            '{}/active_learning_results_{}_counter_{}_{}.csv'.format(self.output_folder, self.data_type, self.counter,
                                                                     self.scorer.score_type), index=False)
        return [indexes[0] for indexes in sorted_results_list][0:self.number_of_samples_in_one_loop]

    def calc_stopping_criteria(self):
        print("testing calc_stopping_criteria:")
        print("y train len is: {}".format(len(self.y_train)))
        print("y test len is: {}".format(len(self.y_test)))
        # for now use up to 30% of the data for training
        return (len(self.y_train)) / (len(self.y_train) + len(self.y_test)) < self.stopping_criteria_train_ratio

    def update_data_set_groups(self, list_of_data_points_ids_learned):

        tmp_list = []
        # remove ids from knowns_data_points_ids
        if len(list_of_data_points_ids_learned) > 0:
            for data_point_id in list_of_data_points_ids_learned:
                self.unknowns_data_points_ids.remove(data_point_id)

            for data_point_tupple in self.unknowns_data_points_tupples:
                if data_point_tupple[0] in list_of_data_points_ids_learned:
                    tmp_list.append(data_point_tupple)

            # remove data points from unknowns_data_points_tupples
            for tmp_data_point_tupple in tmp_list:
                self.unknowns_data_points_tupples.remove(tmp_data_point_tupple)

            # add ids to knowns_data_points_ids
            self.knowns_data_points_ids.extend(list_of_data_points_ids_learned)

            # add data points tupples to knowns_data_points_tupples
            self.knowns_data_points_tupples.extend(tmp_list)

            # update all: tuple - (id, x_data, y_data)
            self.x_train = np.array([id_data_point_tuple[1] for id_data_point_tuple in self.knowns_data_points_tupples])
            self.y_train = np.array([id_data_point_tuple[2] for id_data_point_tuple in self.knowns_data_points_tupples])

            self.x_test = np.array(
                [id_data_point_tuple[1] for id_data_point_tuple in self.unknowns_data_points_tupples])
            self.y_test = np.array(
                [id_data_point_tuple[2] for id_data_point_tuple in self.unknowns_data_points_tupples])

            print(len(self.x_train))
            print(len(self.y_train))
            print(len(self.x_test))
            print(len(self.y_test))

    def _run_training(self):
        csv_logger = CSVLogger('log_model_{}_{}.csv'.format(self.data_type, self.scorer.score_type), append=True,
                               separator=';')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=self.validation_split,
                                                          random_state=42)
        history = self.model.fit(x_train, y_train, epochs=self.epochs,
                                 batch_size=self.batch_size, verbose=2,
                                 callbacks=[csv_logger], validation_data=(x_val, y_val))
        self._plot_history([("{}_counter_{}_{}".format(self.data_type, self.counter, self.scorer.score_type), history)])
        self._save()

    def _save(self):
        model_weights_file_name = '{}/{}_{}_{}.h5'.format(self.output_folder, self.data_type, self.counter,
                                                          self.scorer.score_type)
        model_json_file_name = '{}/{}_{}_{}.json'.format(self.output_folder, self.data_type, self.counter,
                                                         self.scorer.score_type)

        self.model.save_weights(model_weights_file_name)
        model_json_string = self.model.to_json()
        with open(model_json_file_name, 'w') as f:
            json.dump(model_json_string, f)

    def create_shap_visual_outputs(self, ):
        pass

    def _plot_history(self, histories, keys=['categorical_accuracy', 'f1']):
        plt.figure(figsize=(16, 10))

        for name, history in histories:
            for key in keys:
                val = plt.plot(history.epoch, history.history['val_' + key], '--', label=key + '_Val_' + name.title())
                plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                         label=key + '_Train_' + name.title())

            plt.xlabel('Epochs')
            plt.ylabel(" /".join(keys).replace('_', ' '))
            plt.legend()

            plt.xlim([0, max(history.epoch)])
            plt.savefig('{}/model_plot_{}_counter_{}_{}.png'.format(self.output_folder, self.data_type, self.counter,
                                                                    self.scorer.score_type))
            plt.show()
        plt.close()

    def _get_index_data_predictions(self):
        all_test_data = np.array([test_tupple[1] for test_tupple in self.unknowns_data_points_tupples])
        test_predictions = self.model.predict(all_test_data).tolist()
        test_indexes = np.array([test_tupple[0] for test_tupple in self.unknowns_data_points_tupples])
        return [(test_index, test_data, test_prediction) for test_index, test_data, test_prediction in
                zip(test_indexes, all_test_data, test_predictions)]

    def main_active_learning_loop(self):

        self.counter = 0

        self._run_training()

        while self.calc_stopping_criteria():
            start_loop_time = datetime.now()

            test_index_data_and_predictions = self._get_index_data_predictions()

            self.scorer.set_data_and_model(self.knowns_data_points_tupples, self.x_test, self.y_test, self.model)

            sorted_scores_and_indexes = self.scorer.compute_scores(self.counter, self.output_folder,
                                                                   test_index_data_and_predictions)

            choosen_data_points_ids = self.find_next_training_points(sorted_scores_and_indexes)

            print("choosen_data_points_ids:")
            print(choosen_data_points_ids)
            print()

            self.update_data_set_groups(choosen_data_points_ids)

            print("start training again:")
            print()
            self._run_training()

            end_loop_time = datetime.now()

            print("Finish one active learning loop.")
            print("total time: {}".format(end_loop_time - start_loop_time))
            print()

            self.counter += 1