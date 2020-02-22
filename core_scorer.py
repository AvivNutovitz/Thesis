import numpy as np
import shap
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from datetime import datetime
from scipy.stats import ttest_ind, norm
import gc
random.seed(42)


class Scorer():

    def __init__(self, conf, problem_type):
        # score type are the inovation of the thises, score that are supported now are:
        # 1: (LC) least confidence - only based on model performance,
        #    take the examples that the model is least confidence in there classification.
        # 2: (SVV) shap values var - only baesd on shap values,
        #    take the examples that the variace of the shap values are lowset.
        # 3: (SLN) shap values var + least confidence + normalize information bins (my first idea),
        #    create a score baesd on lamdba * each parameter.
        # 4: (SMILC) find the mutal information score of shap values of the best in the class train sample to every test data point,
        #    combine with the least confidence, take examples that with the lowerst combined score.
        # 5: (WSKL) like SKL but with wight based on the confidence level of chi^2 test of the sum(shap values)^2,
        #    take examples with the highset wight SKL.
        # 6: (ND) nosie differences - aproxcimate shap value ideas.
        #    add nosie to every test data point, computue the differences betwwen the original data point confidence and the nosiy data point confidence

        self.conf = conf.get('score_conf', {})
        self.score_type = self.conf.get('score_type', 'LC')
        self.slice_test_data_points = self.conf.get('SLICE_TEST_DATA_POINTS', 500)
        # problem type supported now are text and image, needs to add tabuler data
        self.problem_type = problem_type

    # - set up
    def set_data_and_model(self, knowns_data_points_tupples, x_test, y_test, model):
        self.knowns_data_points_tupples = knowns_data_points_tupples
        self.x_train = np.array([id_data_point_tuple[1] for id_data_point_tuple in self.knowns_data_points_tupples])
        self.knowns_data_points_ids = [id_data_point_tuple[0] for id_data_point_tuple in
                                       self.knowns_data_points_tupples]
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    # - base function
    def compute_scores(self, counter, output_folder, test_index_and_predictions):
        self.counter = counter
        self.output_folder = output_folder
        final_score_results = []
        class_label_and_train_data_dict = None
        # return [(test_data_point_score, test_data_point_index)]

        if self.score_type == 'SMILC':
            print("calculate class_label_and_train_data_dict")
            start_calculate_time = datetime.now()
            class_label_and_train_data_dict = self._get_train_data_best_in_class_shap_values()
            end_calculate_time = datetime.now()
            print(
                f"Finish calculate class_label_and_train_data_dict, total time is: {end_calculate_time - start_calculate_time}")

        # slice to chunks
        test_index_and_predictions_slices = [test_index_and_predictions[i:i + self.slice_test_data_points] for i in
                                             range(0, len(test_index_and_predictions), self.slice_test_data_points)]
        print(f"Total chunks: {len(test_index_and_predictions_slices)}")

        for iteration, test_index_and_predictions_slice in enumerate(test_index_and_predictions_slices):
            print(len(test_index_and_predictions_slice))
            print(f"start chunk iteration: {iteration}")
            start_iteration_time = datetime.now()
            if self.score_type == 'LC':
                final_score_results.extend(self._find_least_confidece_examples(test_index_and_predictions_slice))
            elif self.score_type == 'SVV':
                final_score_results.extend(self._find_variance_shap_values(test_index_and_predictions_slice))
            elif self.score_type == 'SMILC':
                final_score_results.extend(
                    self._find_mutual_info_score_to_best_in_class_training_sample(test_index_and_predictions_slice,
                                                                                  class_label_and_train_data_dict))
            elif self.score_type == 'ND':
                final_score_results.extend(self._find_nosie_diffrences(test_index_and_predictions_slice))
            end_iteration_time = datetime.now()
            print(f"Finish itration {iteration}, total time is: {end_iteration_time - start_iteration_time}")

        final_score_results = sorted(final_score_results, key=lambda tup: tup[1])
        return final_score_results

    # - LC
    def _find_least_confidece_examples(self, test_index_data_and_predictions):
        list_of_results = []
        for test_tupple in test_index_data_and_predictions:
            # test_tupple: index, data, prediction
            # most confident prediction value from list of predictions
            max_prediction = max(test_tupple[2])
            list_of_results.append((test_tupple[0], max_prediction))
        return list_of_results

    # - SVV
    def _find_variance_shap_values(self, test_index_data_and_predictions):
        explainer = self._train_explainer()
        shap_values_and_index = self._build_shap_values_for_all_unknown_data_points(explainer,
                                                                                    test_index_data_and_predictions)
        test_data_points_shap_var_and_index = self._get_var_shap_values(shap_values_and_index)
        # first in the list are smaller variance
        del shap_values_and_index
        gc.collect()
        return test_data_points_shap_var_and_index

    # - helper function
    def _train_explainer(self):
        return shap.DeepExplainer(self.model, self.x_train)

    # - helper function
    def _build_shap_values_for_all_unknown_data_points(self, explainer, test_index_data_and_predictions):
        list_of_results = []
        #         assert len(test_index_data_and_predictions) == len(self.x_test)
        print("Start creating shap values time:")
        start_time = datetime.now()

        all_test_data = np.array([test_data_index[1] for test_data_index in test_index_data_and_predictions])
        all_shap_values = self._build_shap_values(explainer, all_test_data)

        print("Shap Values Shape (when turn into np array): {}".format(np.array(all_shap_values).shape))

        # plot
        if self.problem_type == 'base_image':
            self._plot_shap_values_for_images(all_test_data[0:5].astype('float'), all_shap_values[0:5])

        if self.problem_type == 'base_text':
            self._plot_shap_values_for_text(explainer, all_shap_values, test_index_data_and_predictions)

        # setup results
        for test_data_index, shap_values in zip(test_index_data_and_predictions, all_shap_values[0]):
            list_of_results.append((test_data_index[0], shap_values))

        del all_shap_values
        del all_test_data
        gc.collect()

        end_time = datetime.now()
        print("Finish creating shap values total time is:")
        print(end_time - start_time)
        assert len(test_index_data_and_predictions) == len(list_of_results)

        return list_of_results

    # - helper function for SVV and SLN
    def _build_shap_values(self, explainer, train_or_test_data, one_value=True):
        shap_values, indexes = explainer.shap_values(train_or_test_data, ranked_outputs=1, output_rank_order="min")
        return shap_values

    # - helper function for SVV and SLN
    def _get_var_shap_values(self, shap_values_and_index):
        # return the variance of shap_values and the index
        return [(shap_value_and_index[0], np.var(shap_value_and_index[1].ravel())) for shap_value_and_index in
                shap_values_and_index]

    # - plot text
    def _plot_shap_values_for_text(self, explainer, shap_values, test_index_data_and_predictions):
        shap.summary_plot(shap_values, self.x_test, feature_names=list(self.word_index.keys()), max_display=20,
                          show=False)
        plt.savefig(
            '{}/shap_value_plot_counter_{}_{}_{}.png'.format(self.output_folder, self.counter, self.problem_type,
                                                             self.score_type))

    # - plot image
    def _plot_shap_values_for_images(self, test_data, shap_values):
        shap.image_plot(shap_values, test_data, show=False)
        plt.savefig(
            '{}/shap_value_plot_counter_{}_{}_{}.png'.format(self.output_folder, self.counter, self.problem_type,
                                                             self.score_type))

    # - SKL
    def _find_mutual_info_score_to_best_in_class_training_sample(self, test_index_data_and_predictions,
                                                                 class_label_and_train_data_dict):
        list_of_results = []
        list_of_index_and_one_minus_p_values, test_shap_values_and_indexes = self._build_t_statistic(
            test_index_data_and_predictions)
        print(list_of_index_and_one_minus_p_values)
        list_of_index_and_mutual_info_scores = self._build_mutual_info_scores(class_label_and_train_data_dict,
                                                                              test_shap_values_and_indexes)
        print(list_of_index_and_mutual_info_scores)
        # multiply 1 - confidene with mutual_info_score
        for index_and_one_minus_p_value, index_and_mutual_info_score in zip(list_of_index_and_one_minus_p_values,
                                                                            list_of_index_and_mutual_info_scores):
            list_of_results.append(
                (index_and_one_minus_p_value[0], index_and_one_minus_p_value[1] * index_and_mutual_info_score[1]))

        # sort values get lower values first: low value is combination of weak mutual_info in the shap values
        # and weak confidence in the probability
        del list_of_index_and_one_minus_p_values
        del test_shap_values_and_indexes
        del list_of_index_and_mutual_info_scores
        gc.collect()
        return list_of_results

    # - helper function for SKL
    def _get_train_data_best_in_class_shap_values(self):
        explainer = self._train_explainer()
        train_predictions = self.model.predict(self.x_train)

        # key: class label,
        # value: (index of data point, train prediction)
        train_best_in_class_predictions_and_index_dict = defaultdict(list)
        for train_index, train_prediction in zip(self.knowns_data_points_ids, train_predictions):
            best_class_index = list(np.array(train_prediction).argsort())[-1]
            best_class_prediction = max(train_prediction)
            train_best_in_class_predictions_and_index_dict[best_class_index].append(
                (train_index, best_class_prediction))

        # key: class label,
        # value: (index of best data point)
        class_label_and_train_data_dict = {}
        for class_label, all_train_values in train_best_in_class_predictions_and_index_dict.items():
            best_train_index = max(all_train_values, key=lambda item: item[1])[0]
            # get the data based on the index
            for id_data_point_tuple in self.knowns_data_points_tupples:
                if id_data_point_tuple[0] == best_train_index:
                    class_label_and_train_data_dict[class_label] = np.array(id_data_point_tuple[1])
        return class_label_and_train_data_dict

    # - helper function for SMILC
    def _build_t_statistic(self, test_index_data_and_predictions):
        explainer = self._train_explainer()
        shap_values_and_indexes = self._build_shap_values_for_all_unknown_data_points(explainer,
                                                                                      test_index_data_and_predictions)

        list_of_results = []
        base_group = norm.rvs(size=len(shap_values_and_indexes[0][1]), random_state=1234)

        for shap_value_and_index in shap_values_and_indexes:
            t_stat, p_value = ttest_ind(shap_value_and_index[1].ravel(), base_group)
            list_of_results.append((shap_value_and_index[0], 1 - p_value))

        return list_of_results, shap_values_and_indexes

    # - helper function for SMILC
    def _build_mutual_info_scores(self, class_label_and_train_data_dict, test_shap_values_and_indexes):
        explainer = self._train_explainer()
        list_of_results = []
        all_train_shap_values = []

        # for all train best in class data compute thier shap values
        for _, train_data in class_label_and_train_data_dict.items():
            train_sample_sv = self._build_shap_values(explainer, np.array([train_data]))
            all_train_shap_values.append(train_sample_sv[0][0])

        # for every shap value in the test data find the mutual_info_score to every best in class train data shap values
        for test_shap_values_and_index in test_shap_values_and_indexes:

            test_vs_train_mutual_info_scores = []
            # compare each test shap values compute mutual_info_score agianst all train shap values
            for train_shap_values in all_train_shap_values:
                test_shap_values = test_shap_values_and_index[1].ravel() if self.problem_type == 'base_image' else \
                test_shap_values_and_index[1]
                train_shap_values = train_shap_values.ravel() if self.problem_type == 'base_image' else train_shap_values
                test_vs_train_mutual_info_scores.append(mutual_info_score(train_shap_values, test_shap_values))

            # get the max mutual_info_score
            max_mutual_info_score = max(test_vs_train_mutual_info_scores)
            list_of_results.append((test_shap_values_and_index[0], max_mutual_info_score))

            del test_vs_train_mutual_info_scores

        del all_train_shap_values
        gc.collect()
        return list_of_results

    # ND
    def _find_nosie_diffrences(self, test_index_data_and_predictions):
        # get original data best predictions
        base_predictions = self._find_least_confidece_examples(test_index_data_and_predictions)
        test_index_and_noise_predictions = []
        list_of_results = []
        # add noise to test data
        all_data_with_noise = np.array(
            [self.add_noise_to_data(test_data_index[1]) for test_data_index in test_index_data_and_predictions])
        # run prediction
        noise_raw_predictions = self.model.predict(all_data_with_noise).tolist()
        noise_predictions = [max(predictions) for predictions in noise_raw_predictions]
        # compute diffrences
        for base, noise in zip(base_predictions, noise_predictions):
            list_of_results.append((base[0], np.absolute(base[1] - noise)))
        return list_of_results

    # - helper function for ND
    def add_noise_to_data(self, target):
        target_dims = target.shape
        mask = np.random.randint(0, max(len(target.reshape(1, -1)[0]) * 0.1, 2), size=target.shape).astype(np.bool)
        if self.problem_type == 'base_text':
            target_tmp = target[mask]
            random.shuffle(target_tmp)
            target[mask] = target_tmp
            return target
        elif self.problem_type == 'base_image':
            noise = np.random.randint(-25, 25, (target_dims))
            target[mask] = noise[mask] + target[mask]
            return target