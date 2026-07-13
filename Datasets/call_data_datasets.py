import os
import json
from os import listdir
from os.path import splitext, isfile, join
from sklearn.model_selection import train_test_split
from DatasetsFunctions.ManageDatasets.CallDirectoriesDataset import call_directories_dataset

def call_data_datasets(dataset, random_split_seed):
    # Directories dataset
    DATA_PATH_TRAIN_IMG, DATA_PATH_TRAIN_MASK, DATA_PATH_VAL_IMG, DATA_PATH_VAL_MASK, DATA_PATH_TEST_IMG, DATA_PATH_TEST_MASK = call_directories_dataset(dataset)

    if dataset == 'PH2':
        batch_size = 8
        train_percent = 0.60
        val_percent = 0.20
        test_percent = 0.20

    elif dataset == 'CellTracking':
        batch_size = 8
        train_percent = 0.60
        val_percent = 0.20
        test_percent = 0.20

    assert (train_percent + val_percent + test_percent) == 1
    print(f'Train percent:{train_percent}, validation percent:{val_percent}, batch:{batch_size}, para el dataset {dataset}')

    indices_train = [splitext(file)[0][:] for file in listdir(DATA_PATH_TRAIN_IMG) if isfile(join(DATA_PATH_TRAIN_IMG, file)) and not file.startswith('.')]
    TOTAL_INDICES_TRAIN = len(indices_train)

    if DATA_PATH_TEST_IMG != '':
        INDICES_TEST = [splitext(file)[0][:] for file in listdir(DATA_PATH_TEST_IMG) if isfile(join(DATA_PATH_TEST_IMG, file)) and not file.startswith('.')]
    else:
        INDICES_TEST = []

    if DATA_PATH_VAL_IMG != '':
        INDICES_VAL = [splitext(file)[0][:] for file in listdir(DATA_PATH_VAL_IMG) if isfile(join(DATA_PATH_VAL_IMG, file)) and not file.startswith('.')]

    else:
        # New indices if val its not separated
        n_val = int(val_percent*len(indices_train))
        n_train = len(indices_train) - n_val

        indices_train, INDICES_VAL = train_test_split(indices_train, train_size=n_train, test_size=n_val, random_state=random_split_seed)

        n_test = int(test_percent*TOTAL_INDICES_TRAIN)
        n_train =  n_train - n_test

        assert (n_train + n_val + n_test) == TOTAL_INDICES_TRAIN
        if n_test != 0:
            indices_train, INDICES_TEST = train_test_split(indices_train, train_size=n_train, test_size=n_test, random_state=random_split_seed)

    # save diccionarie
    dir_diccionarios = r'DiccionariosCrossValidation'
    name_split = dataset + '_' + str(random_split_seed) + '.json'
    dir_full_splits = os.path.join(dir_diccionarios, name_split)

    # Indices dict
    dicc_indices = {}
    dicc_indices['test'] = INDICES_TEST
    dicc_indices['train'] = indices_train 
    dicc_indices['val'] = INDICES_VAL

    # If exists check its the same
    if os.path.exists(dir_full_splits):
        with open(dir_full_splits,'r') as file:
            dicc_indices_old = json.load(file)
            # Check experiments use same indices
            assert dicc_indices_old == dicc_indices
    else:
        # New diccionario if doesn't exist
        with open(dir_full_splits,'w') as outfile:
            json.dump(dicc_indices, outfile)
            print('Indices guardados correctamente')

    dict_data = {'PATH_TRAIN_IMG':DATA_PATH_TRAIN_IMG, 'PATH_TRAIN_MASK':DATA_PATH_TRAIN_MASK, 'PATH_VAL_IMG':DATA_PATH_VAL_IMG, 'PATH_VAL_MASK':DATA_PATH_VAL_MASK,
                'PATH_TEST_IMG':DATA_PATH_TEST_IMG, 'PATH_TEST_MASK':DATA_PATH_TEST_MASK, 'BATCH_SIZE':batch_size, 'INDICES_TRAIN':indices_train, 'INDICES_VAL':INDICES_VAL,
                'INDICES_TEST':INDICES_TEST}

    return dict_data
