####################################################
#                    Imports                       #
####################################################

import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from Training.TestMini.Segmentor_TEST_mini import *
from Models.UNet.UNetBatchNorm import UNetBatchNorm
from DatasetsFunctions.DatasetsClasses.DatasetPH2 import PH2_Loader
from DatasetsFunctions.ManageDatasets.call_data_datasets import call_data_datasets

####################################################
#                   CrossValidation                #
####################################################

# Diccionario modelos
dict_model = {'UNet_PH2': lambda: UNetBatchNorm(n_channels=3, n_classes=2),}
conjunto = 'test'

GENERAL_DIR = r'EXP'    
name_checkpoint = r'dataset_loss_lrvalue_seedinitvalue_randomsplitvalue_dicepromval.ckpt'

dataset = name_checkpoint.split('_')[0]
dir_checkpoint = os.path.join(GENERAL_DIR,name_checkpoint)

if conjunto == 'val':
    iter = 'MetricasPerLossVal'
else:
    iter = 'MetricasPerLossTest'


# String information
if dataset in ['THP1_Cells','DRIVE_DAUG']:
    loss_func = dir_checkpoint.split('/')[-1].split('_')[2]
    seed_init_str = dir_checkpoint.split('_')[5][8:]
    random_split_str = dir_checkpoint.split('_')[6][11:]
else:
    loss_func = dir_checkpoint.split('/')[-1].split('_')[1] #+ '_' + dir_checkpoint.split('/')[-1].split('_')[2]
    seed_init_str = dir_checkpoint.split('_')[3][8:]
    random_split_str = dir_checkpoint.split('_')[4][11:]

Seeds = {int(seed_init_str):[int(random_split_str)]}
device = torch.device('cuda:0')
normalize_img = 'Custom'

print(Seeds)

for seed_initial_params in Seeds:
    List_randomsplitseeds = Seeds[seed_initial_params]

    for random_split_seed in List_randomsplitseeds:
        DICT_DATA_DATASET = call_data_datasets(dataset, random_split_seed)

        ####################################################
        #                   DataLoaders                    #
        ####################################################
        if conjunto == 'val':
            indices=DICT_DATA_DATASET['INDICES_VAL']
        else:
            indices=DICT_DATA_DATASET['INDICES_TEST']

        if dataset == 'PH2':
            test_dataset = PH2_Loader(img_path=DICT_DATA_DATASET['PATH_TRAIN_IMG'], mask_path=DICT_DATA_DATASET['PATH_TRAIN_MASK'], indices=indices, transform=False, 
                                    conjunto = "validation")
            
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False, persistent_workers=True)
            
        logger = CSVLogger(save_dir=GENERAL_DIR, name=iter, version=loss_func)
        trainer = pl.Trainer(logger=logger)
        model = dict_model['UNet_PH2']()

        # Cargamos modelo
        model.eval()
        checkpoints = torch.load(dir_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoints["state_dict"].items() if k.startswith("model.")})
        print('Pesos cargados correctamente')

        # Cargamos segmentador
        save_dir_final = os.path.join(GENERAL_DIR, iter, loss_func)
        segmentor = Segmentor_TEST_mini(model = model, model_type='UNet', save_dir=save_dir_final)
        trainer.test(segmentor, test_loader)

        # Clean
        torch.cuda.empty_cache()

        del trainer
        del model
        del checkpoints
        del segmentor
        del test_loader