####################################################################
#                            Imports                               #
####################################################################
# Torch
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

# Modelos
from Training.Train.Segmentor_mini import *
from Model.UNetBatchNorm import UNetBatchNorm


# Directorios y datasets
from Training.Segmentor_TRAIN import *
from Datasets.DatasetPH2 import PH2_Loader
from Datasets.call_data_datasets import call_data_datasets

# Loss
from Losses.LoAC import *
from Losses.CEDice import *
from Losses.ABeDice import *
from Losses.NAC_Loss import *
from Losses.CrossEntropy import *
from Losses.DiceForeground import *

####################################################################
#                        Diccionarios                              #
####################################################################
# Diccionario monitor
dict_monitor = {"val_dice":"max", "val_loss":"min"}

# Diccionario modelos
dict_model = {'UNet_PH2': lambda: UNetBatchNorm(n_channels=3, n_classes=2),}

dict_loss = {
    'CEDice': lambda: CEDice(),
    'CE': lambda: CrossEntropySeg(),
    'Dice': lambda: DiceForeground(),
    'NAC': lambda: NAC_LossS(NUM_CLASS=2),
    'LoAC': lambda: LoAC(n_classes=2, mu=1,alpha=1),
    'ABeDice': lambda: ABeDice( n_classes=2, alpha=2, beta=3),
    }

# Folds
DIR_FOLDS_PH2 = r'Checkpoints'


dict_checkpoints = {'Prueba1':DIR_FOLDS_PH2}

                    # Model, Schedular, Dataset, Lr
dict_experimento = {'Prueba2.5':{'PARAMETROS_JUSTOS':['UNet_PH2','val_dice','PH2', 1e-4],
                    # Init seed parameters, Fold
                    'Seeds':{60:[100]},
                    # Losses
                    'Losses_exp':['CE','Dice','NAC','ABeDice','LoAC']},}


####################################################################
#                         Train Loop                               #
####################################################################
for key in dict_experimento:
    # Control / Equal / Variables
    max_epochs = 2000
    early_stopping_bool = True
    consider_resize = False
    transform = True

    #dir
    Checkpoint_dir_general = dict_checkpoints[key]

    # PARAMETROS JUSTOS
    model_type = dict_experimento[key]['PARAMETROS_JUSTOS'][0]
    monitor =  dict_experimento[key]['PARAMETROS_JUSTOS'][1]
    dataset = dict_experimento[key]['PARAMETROS_JUSTOS'][2]
    learning_rate = dict_experimento[key]['PARAMETROS_JUSTOS'][3]

    # mode del schedular
    mode = dict_monitor[monitor]

    ####################################################################
    #                           ASSERTS                                #
    ####################################################################
    if model_type not in dict_model.keys():
        raise Exception(f'Modelo no contemplado')
    if monitor not in ["val_dice", "val_loss"] or type(max_epochs) != int or type(early_stopping_bool) != bool:
        raise Exception(f'Elige correctamente un monitor u opcion de early stopping')
    if dataset not in ['PH2']:
        raise Exception(f'Dataset no contemplado')
    if type(learning_rate) != float:
        raise Exception(f'Learning rate erronea')

    ####################################################################
    #                     Directorios y seeds                          #
    ####################################################################
    # Dicts for crossvalidation
    Dict_seeds = dict_experimento[key]['Seeds']
    List_Losses_exps = dict_experimento[key]['Losses_exp']

    for seed_initial_params in Dict_seeds:
        # Randoms seeds for split
        Lista_randoms_seed_for_split = Dict_seeds[seed_initial_params]

        for random_split_seed in Lista_randoms_seed_for_split:
            ####################################################################
            #                          Training                                #
            ####################################################################
            # Inicializamos params
            DICT_DATA_DATASET = call_data_datasets(dataset, random_split_seed)

            for loss_function_name in List_Losses_exps:
                seed_everything(seed_initial_params, workers=True)

                torch.use_deterministic_algorithms(True,warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                experimento = loss_function_name
                print(loss_function_name)

                ####################################################################
                #                        DataLoader                                #
                ####################################################################
                # Dataset
                train_dataset = PH2_Loader(img_path=DICT_DATA_DATASET['PATH_TRAIN_IMG'], mask_path=DICT_DATA_DATASET['PATH_TRAIN_MASK'], indices=DICT_DATA_DATASET['INDICES_TRAIN'], transform=transform,
                                            conjunto = "train")
                val_dataset = PH2_Loader(img_path=DICT_DATA_DATASET['PATH_TRAIN_IMG'], mask_path=DICT_DATA_DATASET['PATH_TRAIN_MASK'], indices=DICT_DATA_DATASET['INDICES_VAL'], transform=False,
                                    conjunto = "validation")
                
                num_workers = 4
                train_loader = DataLoader(train_dataset, batch_size=DICT_DATA_DATASET['BATCH_SIZE'], num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=False)
                val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=False)
                ####################################################################
                #                         Modelo                                   #
                ####################################################################
                # Cargamos Modelo
                model = dict_model[model_type]()
                loss_function = dict_loss[loss_function_name]()

                name = f"{dataset}_{experimento}_lr{learning_rate}_seedinit{seed_initial_params}_randomsplit{random_split_seed}"
                logger = CSVLogger(save_dir=Checkpoint_dir_general, name=name)
                check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=Checkpoint_dir_general, filename= name +  f"_{{{monitor}:0.4f}}",
                                                                            monitor=monitor, mode = mode, save_top_k =1, verbose=True, save_weights_only=True, auto_insert_metric_name=False)
                progress_bar = pl.callbacks.TQDMProgressBar()

                if early_stopping_bool == True:
                    #patience = 60
                    early_stop = EarlyStopping(monitor=monitor, mode=mode, patience=50, min_delta=1e-4, verbose=True)
                    PARAMS = {"enable_progress_bar" : True, "logger":logger, "callbacks" : [check_point, early_stop, progress_bar],
                            "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":max_epochs} 
                else:
                    PARAMS = {"enable_progress_bar" : True, "logger":logger, "callbacks" : [check_point, progress_bar],
                        "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":max_epochs}

                # Segmentator
                trainer = pl.Trainer(accelerator="gpu", devices=1, deterministic="warn", benchmark=False,precision="32", **PARAMS)
                segmentor = Segmentator_TRAIN(model=model, model_type=model_type, loss_function=loss_function, loss_function_name=loss_function_name, 
                                        learning_rate=learning_rate, monitor = monitor, mode = mode, batch_size=DICT_DATA_DATASET['BATCH_SIZE'])
                
                trainer.fit(segmentor, train_loader, val_loader)

                # Clean
                del segmentor
                del trainer
                del train_loader
                del val_loader
                del model
                del loss_function
                del check_point
                del progress_bar
                if early_stopping_bool:
                    del early_stop
                torch.cuda.empty_cache()

            del DICT_DATA_DATASET
