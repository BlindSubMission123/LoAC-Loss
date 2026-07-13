def call_directories_dataset(dataset):
    if dataset == 'PH2':
        # Train
        DATA_PATH_TRAIN_IMG = r'/home/.../datasets/PH2_NewOrder/Images'
        DATA_PATH_TRAIN_MASK = r'/home/.../datasets/PH2_NewOrder/Masks'
        # Validation
        DATA_PATH_VAL_IMG = ''
        DATA_PATH_VAL_MASK = ''
        # Test
        DATA_PATH_TEST_IMG = ''
        DATA_PATH_TEST_MASK = ''

    if dataset == 'CellTracking':
        # Train 
        DATA_PATH_TRAIN_IMG = r'/home/.../datasets/DatasetsCellTracking/Images'
        DATA_PATH_TRAIN_MASK = r'/home/.../datasets/DatasetsCellTracking/GroundTruth'
        # Validation
        DATA_PATH_VAL_IMG = ''
        DATA_PATH_VAL_MASK = ''
        # Test
        DATA_PATH_TEST_IMG = ''
        DATA_PATH_TEST_MASK = ''

    return DATA_PATH_TRAIN_IMG, DATA_PATH_TRAIN_MASK, DATA_PATH_VAL_IMG, DATA_PATH_VAL_MASK, DATA_PATH_TEST_IMG, DATA_PATH_TEST_MASK
