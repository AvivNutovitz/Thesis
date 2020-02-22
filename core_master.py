from core_data_preprocessing import Preprocessing
from core_modeler import Modeler
from core_active_learning import ActiveLearning
from core_scorer import Scorer


class Master():
    """
    this is the master class that incharge of all the program
    """

    def __init__(self, conf, data_type, on_server=True):
        self.conf = conf
        # data
        data_preprocessing = Preprocessing(conf, on_server)

        # model
        model_constractor = Modeler(conf)

        # scorer
        scorer = Scorer(conf, data_type)

        # active_learning
        self.active_learning = ActiveLearning(conf, data_preprocessing, model_constractor, scorer, data_type)
        self.active_learning.setup()

    def run(self):
        self.active_learning.main_active_learning_loop()