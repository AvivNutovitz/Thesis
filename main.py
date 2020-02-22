import gc
from core_master import Master

if __name__ == '__main__':

    text_conf = {
        'MAX_SEQUENCE_LENGTH': 50,
        'MAX_NB_WORDS': 10000,
        'EMBEDDING_DIM': 50,
        'HIDDEN_SIZE': 512
    }

    dp_conf = {
        'text_conf': text_conf
    }

    al_conf = {
        'STOPPING_CRITERIA': {
            'TRAIN_RATIO': 0.3,
        },
        'NUMER_OF_SAMPELS_IN_ONE_LOOP': 500,
        'EPOCHS': 30,
        'BATCH_SIZE': 100,
        'VALIDATION_SPLIT': 0.3,
        'OUTPUT_FOLDER': 'outputs',
    }
    score_conf = {
        'score_type': 'LC',
        'SLICE_TEST_DATA_POINTS': 500
    }

    conf = {
        'dp_conf': dp_conf,
        'm_conf': {},
        'al_conf': al_conf,
        'score_conf': score_conf
    }

    #     data_type = 'base_tabular'
    #     # LC, SMILC
    #     conf['score_conf']['score_type'] = 'LC'
    #     master_1 = Master(conf, data_type)
    #     master_1.run()
    #     del master_1
    #     gc.collect()

    data_type = 'base_image'
    # LC, SMILC, ND
    conf['score_conf']['score_type'] = 'LC'
    master_2 = Master(conf, data_type)
    master_2.run()
    del master_2
    gc.collect()

    data_type = 'base_text'
    # LC, SMILC, ND
    conf['score_conf']['score_type'] = 'LC'
    master_3 = Master(conf, data_type)
    master_3.run()
    del master_3
    gc.collect()

    data_type = 'base_image'
    # LC, SMILC, ND
    conf['score_conf']['score_type'] = 'ND'
    master_4 = Master(conf, data_type)
    master_4.run()
    del master_4
    gc.collect()

    data_type = 'base_text'
    # LC, SMILC, ND
    conf['score_conf']['score_type'] = 'ND'
    master_5 = Master(conf, data_type)
    master_5.run()
    del master_5
    gc.collect()