import numpy as np
import torch

import pickle
import os

import utils.directory
import utils.checkpoint

class Checkpoint(object):
    """
    Early stopping to stop the training when the accuracy (or other metric) does not improve 
    after certain epochs.

    Parameters
    ----------
        patience (int): how many epochs to wait before stopping when loss is not improving.

        min_delta (int): minimum difference between new loss and old loss for new loss to be 
            considered as an improvement.

        path (str): Path for the checkpoint to be saved to.

        metric_eval (str): Name of the metric used.
    """

    def __init__(self, CONFIG, min_delta=0, verbose=True):

        # Network 
        self.rnn_type = CONFIG.rnn_type
        self.input_size = CONFIG.input_size
        self.hidden_size = CONFIG.hidden_size
        self.num_layers = CONFIG.num_layers
        self.num_classes = CONFIG.num_classes
        self.dropout = CONFIG.dropout
        self.activation = CONFIG.activation
        self.batch_norm = CONFIG.batch_norm

        # Early stopping
        self.patience = CONFIG.patience
        self.metric_eval = CONFIG.metric_eval

        self.save_file = CONFIG.save_file + CONFIG.experiment + CONFIG.name_model

        # Me sirven?
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0 # Deberia guardarlo para saber en cual early stoppinh va
        self.best_acc = None
        self.early_stop = False

        self.val_acc_min = np.Inf

        self.train_loss = []
        self.val_loss = []
        self.train_acc = [] 
        self.val_acc = []
        self.checkpoint_early = None
        self.checkpoint_training = None

    def save_init(self, loss_list, optimizer, scheduler, lr):       
        '''Guarda los parametros utilizados'''
        ## Files where the results will be saved 
        utils.directory.create_dir(self.save_file)

        dictionary_model = {'rnn_type': self.rnn_type,
                            'input_size': self.input_size,
                            'hidden_size': self.hidden_size,
                            'num_layers': self.num_layers,
                            'num_classes': self.num_classes,
                            'dropout': self.dropout,
                            'activation': self.activation,
                            'batch_norm': self.batch_norm,
                            'loss_list': loss_list,
                            'optimizer': optimizer,
                            'scheduler': scheduler,
                            'lr': lr,
                            'patience': self.patience,
                            'metric_eval': self.metric_eval
                            }

        with open(self.save_file + '/parameters.model', 'wb') as f:
            pickle.dump(dictionary_model, f)

    def save_training(self, epoch, model, optimizer, lr_scheduler):
        self.type = 'training' 
        self.save_checkpoint(epoch, model, optimizer, lr_scheduler)     

    def early_stopping(self, epoch, model, optimizer, lr_scheduler):
        """
        Train_loss, train_acc, 
        val_loss y val_acc: Listas que contienen el valor promedio de cada epoca
        """
        self.type = 'early_stopping'

        if self.best_acc is None:
            self.best_acc = self.val_acc[-1]
            self.save_checkpoint(epoch, model, optimizer, lr_scheduler)

        elif self.val_acc[-1] > self.best_acc + self.min_delta:
            self.best_acc = self.val_acc[-1]
            self.counter = 0 # reset counter if validation loss improves
            self.save_checkpoint(epoch, model, optimizer, lr_scheduler)

        elif self.val_acc[-1] < self.best_acc + self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")

            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        '''Saves model when metric evaluated decrease.'''

        checkpoint = {'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    #'scheduler': lr_scheduler.state_dict(),
                    'train_loss': self.train_loss,
                    'val_loss': self.val_loss,
                    'train_acc': self.train_acc,
                    'val_acc': self.val_acc,
                    'counter': self.counter
                    }

        if self.type == 'early_stopping':
            if self.verbose:
                print(f'Saving model... {self.metric_eval} improved from {self.val_acc_min:.5f} to {self.val_acc[-1]:.5f}')

            self.create_checkpoint_file(self.checkpoint_early)
            self.checkpoint_early = self.save_file + f'/best_result/ckpt-{epoch:03d}-{self.val_acc[-1]:.4f}.pt'
            checkpoint_path = self.checkpoint_early
            self.val_acc_min = self.val_acc[-1] 

        elif self.type == 'training':
            self.create_checkpoint_file(self.checkpoint_training)
            self.checkpoint_training = self.save_file + f'/save_training/ckpt-{epoch:03d}.pt'
            checkpoint_path = self.checkpoint_training 

        torch.save(checkpoint, checkpoint_path)        

    def create_checkpoint_file(self, checkpoint_path):
        if checkpoint_path is not None:
            utils.directory.remove_dir(checkpoint_path)
        else:
            if self.type == 'early_stopping': path_aux = self.save_file + '/best_result'
            elif self.type == 'training': path_aux = self.save_file + '/save_training'
            utils.directory.create_dir(path_aux)
