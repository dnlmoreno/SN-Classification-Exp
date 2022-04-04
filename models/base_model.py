import time

import numpy as np
#import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle
import torch
#from torch.utils.tensorboard import SummaryWriter
#import torchvision

# Files.py
import models.sne_models
import utils.checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class Model(object):
    def __init__(self, CONFIG):
        super(Model, self).__init__()
        # Load and save model
        self.name_model = CONFIG.name_model
        self.param_path = CONFIG.load_file + CONFIG.experiment + CONFIG.name_model
        #self.training = CONFIG.training

        self.epochs_trained = 1
        if CONFIG.load_model:
            model_path = self.param_path + CONFIG.model_path
            self.epochs_trained, CONFIG = self.load_checkpoint(self.param_path, model_path, CONFIG)
            print(f"The model was successfully loaded for training.\n")

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

        # Model
        if CONFIG.load_model == False: 
            self.model = self.__choose_model()
            self.checkpoint = utils.checkpoint.Checkpoint(CONFIG)

        #self.writer = SummaryWriter(f"runs/{CONFIG.experiment}")

    def summary(self):
        print(self.model)

    def compile(self, loss_list, optimizer, lr, metric_eval, scheduler=None, save=True):
        self.loss = self.__choose_loss_function(loss_list) # choose loss function
        self.optimizer = self.__choose_optimizer(optimizer, lr) # choose optimizer
        if scheduler is not None:
            self.lr_scheduler = self.__choose_lr_scheduler(scheduler) # Se le ingresa el optimizer
        else:
            self.lr_scheduler = scheduler
        
        self.metric_eval = metric_eval

        # save init paremeters
        if save:
            self.checkpoint.save_init(loss_list, optimizer, scheduler, lr)

    def fit(self, train_dataloader, val_dataloader, train_type, num_epochs, verbose):
        self.model.to(device)

        for epoch in range(self.epochs_trained, num_epochs+1):
            # Training epoch
            start_epoch = time.time()

            train_loss, train_acc = self.__train_one_epoch(train_dataloader, train_type)
            val_loss, val_acc = self.__val_one_epoch(val_dataloader)
            
            end_epoch = time.time()

            # loss promedio de la epoca
            epoch_train_loss, epoch_val_loss = np.mean(train_loss), np.mean(val_loss)
            epoch_train_acc, epoch_val_acc = np.mean(train_acc), np.mean(val_acc)

            if verbose:
                print(f'\nEpoch {epoch}/{num_epochs} '
                    + f'- time: {(end_epoch-start_epoch):.3f} seg\n'
                    + f'loss = {epoch_train_loss:.5f}, '
                    + f'val_loss = {epoch_val_loss:.5f}, '
                    + f'acc = {epoch_train_acc:.5f}, '
                    + f'val_acc = {epoch_val_acc:.5f}, ') 

                #self.writer.add_scalars('Loss', {'Training': epoch_train_loss, 
                #                                'Validation': epoch_val_loss},
                #                                epoch)

                #self.writer.add_scalars('Accuracy', {'Training': epoch_train_acc,
                #                                    'Validation': epoch_val_acc},
                #                                    epoch)

            # guarda el loss promedio de cada epoca
            self.checkpoint.train_loss.append(epoch_train_loss), self.checkpoint.val_loss.append(epoch_val_loss)
            self.checkpoint.train_acc.append(epoch_train_acc), self.checkpoint.val_acc.append(epoch_val_acc)

            # if accuracy improves save the best model
            self.checkpoint.early_stopping(epoch, self.model, self.optimizer, self.lr_scheduler)

            # guarda el modelo cada cierta cantidad de epochs
            if epoch % 1 == 0:
                self.checkpoint.save_training(epoch, self.model, self.optimizer, self.lr_scheduler)

            if self.checkpoint.early_stop:
                #self.writer.close()
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        #self.writer.close()
    
    def evaluate(self, test_dataloader, batch_size):
        loss_list, acc_list = [], []
        self.model.eval() # No activate dropout and batchNorm

        with torch.no_grad():
            for batch_idx, (data, targets, _) in enumerate(test_dataloader):
                # Get data to cuda if possible
                X_test_sorted = data.to(device)
                y_test_sorted = torch.stack(targets).to(device)

                #batch_size = X_test.batch_sizes[0].item()
                batch_size = test_dataloader.batch_size

                scores_sorted = self.model(X_test_sorted, batch_size) # NN raw output is [3.2, 1.3, 0.2, 0.8]
                loss = self.loss[0](scores_sorted, y_test_sorted) # the softmax probability output is [0.775, 0.116, 0.039, 0.07]
                #loss_batch = self.loss[0](torch.nn.functional.log_softmax(scores_batch, dim=1), y_test) 
                loss_list.append(loss)

                # metrics 
                acc = self.__get_metrics(y_test_sorted, scores_sorted)
                acc_list.append(acc)

        loss = torch.mean(torch.stack(loss_list)).item()
        acc = np.mean(acc_list)

        return loss, acc

    def predict(self, test_dataloader, batch_size):
        y_pred_prob, y_pred = [], []
        self.model.eval() # No activate dropout and batchNorm

        with torch.no_grad():
            for batch_idx, (data, targets, idx_sorted) in enumerate(test_dataloader):
                # Get data to cuda if possible
                X_test_sorted = data.to(device)

                #batch_size = X_test.batch_sizes[0].item()
                batch_size = test_dataloader.batch_size

                # forward/predict
                scores_sorted = self.model(X_test_sorted, batch_size)

                # Vuelve las predicciones y etiquetas al orden normal
                # sirve para evaluar las metricas
                scores = torch.zeros(size=scores_sorted.size())
                for i in range(len(idx_sorted)):
                    scores[idx_sorted[i]] = scores_sorted[i]

                y_pred_prob_batch = torch.nn.functional.softmax(scores, dim=1).to(device)
                y_pred_prob.append(y_pred_prob_batch)

                y_batch_pred = torch.argmax(y_pred_prob_batch, axis=1)
                y_pred.append(y_batch_pred)

        y_pred_prob = torch.cat((y_pred_prob),dim=0).cpu().numpy()
        y_pred = torch.cat((y_pred),dim=0).cpu().numpy()

        return y_pred_prob, y_pred

    def load_checkpoint(self, param_path, model_path, CONFIG):

        with open(param_path + '/parameters.model', 'rb') as f:
            dictionary_model = pickle.load(f)

        ## Parametros tienen que coincidir con el metodo save_init en checkpoint
        # Network
        CONFIG.rnn_type = self.rnn_type = dictionary_model['rnn_type']
        CONFIG.input_size = self.input_size = dictionary_model['input_size']
        CONFIG.hidden_size = self.hidden_size = dictionary_model['hidden_size']
        CONFIG.num_layers = self.num_layers = dictionary_model['num_layers']
        CONFIG.num_classes = self.num_classes = dictionary_model['num_classes']
        CONFIG.dropout = self.dropout = dictionary_model['dropout']
        CONFIG.activation = self.activation = dictionary_model['activation']
        CONFIG.batch_norm = self.batch_norm = dictionary_model['batch_norm']
        # Compile
        CONFIG.loss_list = dictionary_model['loss_list']
        CONFIG.optimizer = dictionary_model['optimizer']
        #scheduler = dictionary_model['scheduler']
        CONFIG.lr = dictionary_model['lr']
        # Early stopping
        CONFIG.patience = dictionary_model['patience']
        CONFIG.metric_eval = dictionary_model['metric_eval']
        
        # Model load
        checkpoint = torch.load(model_path)

        epochs_trained = checkpoint['epoch']
        self.model = self.__choose_model() # self.name_model se elige manualmente
        self.model.load_state_dict(checkpoint['model_state'])
        
        self.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval, save=False)
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        #self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.lr_scheduler = None # momentaneo

        # To checkpoint
        self.checkpoint = utils.checkpoint.Checkpoint(CONFIG)
        self.checkpoint.train_loss = checkpoint['train_loss']
        self.checkpoint.val_loss = checkpoint['val_loss']
        self.checkpoint.train_acc = checkpoint['train_acc']
        self.checkpoint.val_acc = checkpoint['val_acc']

        # early stopping values
        self.checkpoint.val_acc_min = max(self.checkpoint.val_acc)
        self.checkpoint.best_acc = self.checkpoint.val_acc[-1]
        self.checkpoint.counter = checkpoint['counter']

        return epochs_trained, CONFIG

    def __train_one_epoch(self, train_dataloader, train_type):
        self.model.train()

        if train_type == 'normal':
            return self.__train_normal(train_dataloader)
        elif train_type == 'kd':
            return self.__train_kd(train_dataloader)

    def __val_one_epoch(self, val_dataloader):
        val_loss, val_acc = [], []
        self.model.eval() # No activate dropout and batchNorm

        with torch.no_grad():
            for batch_idx, (data, targets, idx_sorted) in enumerate(val_dataloader):
                # Get data to cuda if possible
                X_val_sorted = data.to(device)
                y_val_sorted = torch.stack(targets).to(device)

                batch_size = val_dataloader.batch_size

                # forward
                scores_sorted = self.model(X_val_sorted, batch_size)
                loss = self.loss[0](scores_sorted, y_val_sorted)
                #loss = self.loss[0](torch.nn.functional.log_softmax(scores, dim=1), y_val) 

                # Vuelve las predicciones y etiquetas al orden normal
                # sirve para evaluar las metricas
                scores = torch.zeros(size=scores_sorted.size())
                y_val = torch.zeros(size=y_val_sorted.size()) 
                for i in range(len(idx_sorted)):
                    scores[idx_sorted[i]] = scores_sorted[i]
                    y_val[idx_sorted[i]] = y_val_sorted[i]
            
                # metrics 
                acc = self.__get_metrics(y_val, scores)

                # save metrics for step
                val_loss.append(loss.item()) # standard Python number
                val_acc.append(acc)

        return val_loss, val_acc


    def __train_normal(self, train_dataloader):
        train_loss, train_acc = [], []
        y_pred = []

        for batch_idx, (data, targets, idx_sorted) in enumerate(train_dataloader):
            # Get data in cuda if it was possible
            X_train_sorted = data.to(device)
            y_train_sorted = torch.stack(targets).to(device)

            batch_size = train_dataloader.batch_size

            # forward
            scores_sorted = self.model(X_train_sorted, batch_size)
            loss = self.loss[0](scores_sorted, y_train_sorted)
            #loss = self.loss[0](torch.nn.functional.log_softmax(scores, dim=1), y_train) 

            # Vuelve las predicciones y etiquetas al orden normal
            # sirve para evaluar las metricas
            scores = torch.zeros(size=scores_sorted.size())
            y_train = torch.zeros(size=y_train_sorted.size()) 
            for i in range(len(idx_sorted)):
                scores[idx_sorted[i]] = scores_sorted[i]
                y_train[idx_sorted[i]] = y_train_sorted[i]           

            # backward
            self.optimizer.zero_grad() # Quizas tenga que poner [0]
            loss.backward()

            # gradient descent or adam step
            self.optimizer.step()

            # metrics 
            acc = self.__get_metrics(y_train, scores)

            # save metrics for step
            train_loss.append(loss.item()) # standard Python number
            train_acc.append(acc)

        return train_loss, train_acc


    def __train_kd(self, train_dataloader, unlabeled_dataloader):
        # rate
        rate = unlabeled_dataloader.batch_size/train_dataloader.batch_size
        train_loss, train_acc = [], []

        for batch_idx, (labeled_data, unlabeled_data) in enumerate(zip(train_dataloader, unlabeled_dataloader)):
            # Get data in cuda if it was possible
            X_train, y_train = labeled_data.to(device)
            X_unl, y_soft = unlabeled_data.to(device)

            maxv, _ = y_soft.max(dim=1)
            valid_idx = torch.where(maxv > self.teacher.threshold)
            X_unl = X_unl[valid_idx]
            y_soft = y_soft[valid_idx]

            # forward
            score_student = self.model(X_train)
            score_student_soft = self.model(X_unl)

            hard_loss = self.loss[0](score_student, y_train)
            soft_loss = self.loss[1](torch.nn.functional.log_softmax(score_student_soft/self.T, dim=1), y_soft).float() * rate
            loss = self.alpha * soft_loss * self.T**2/rate + (1-self.alpha) * hard_loss #.double()

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            self.optimizer.step()

            # metrics 
            acc = self.__get_metrics(y_train, score_student) # Metrica solo del estudiante?

            # save metrics for step
            train_loss.append(loss.item()) # standard Python number
            train_acc.append(acc)

        return train_loss, train_acc


    def __choose_model(self):
        """Selecciona el modelo de clasificación de supernovas"""
        if self.name_model.lower() == 'charnock':
            model = models.sne_models.CharnockModel(self.rnn_type, self.input_size, 
                                                   self.hidden_size, self.num_layers,
                                                   self.num_classes, self.dropout, 
                                                   self.activation, self.batch_norm)
        elif self.name_model.lower() == 'rapid':
            model = models.sne_models.RapidModel(self.rnn_type, self.input_size, 
                                                self.hidden_size, self.num_layers,
                                                self.num_classes, self.dropout, 
                                                self.activation, self.batch_norm)
        elif self.name_model.lower() == 'supernnova':
            model = models.sne_models.SuperNNovaModel(self.rnn_type, self.input_size, 
                                                    self.hidden_size, self.num_layers,
                                                    self.num_classes, self.dropout, 
                                                    self.activation, self.batch_norm)
        elif self.name_model.lower() == 'donoso':
            model = models.sne_models.DonosoModel(self.rnn_type, self.input_size, 
                                                self.hidden_size, self.num_layers,
                                                self.num_classes, self.dropout, 
                                                self.activation, self.batch_norm)
        
        return model.to(device)

    def __choose_loss_function(self, loss_list):
        criterion = []
        for loss in loss_list:
            if loss.lower() == 'crossentropy':
                criterion.append(torch.nn.CrossEntropyLoss().to(device))
            elif loss.lower() == 'kldiv':
                criterion.append(torch.nn.KLDivLoss().to(device))
            elif loss.lower() == 'cce':
                # it need (torch.nn.functional.log_softmax(y_pred), y_true)
                criterion.append(torch.nn.NLLLoss().to(device))

        return criterion

    def __choose_optimizer(self, optim, lr):
        if optim.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim.lower() == 'rms':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

        return optimizer

    def __choose_lr_scheduler(self, scheduler):
        if scheduler.lower() == 'cosineannealing': # ARREGLAR
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 50, 1e-6)

        return lr_scheduler

    def __get_metrics(self, y_true, scores):
        y_pred = torch.argmax(scores, axis=1)

        # Try if y_true is list or one-hot 
        try:
            acc = (y_true == y_pred).sum().item() / len(y_true)
        except RuntimeError:
            y_true = torch.argmax(y_true, dim=1)
            acc = (y_true == y_pred).sum().item() / len(y_true)       

        if self.metric_eval.lower() == 'f1_score':
            #f1_score_custom = utils.F1Score('macro')(y_val_pred, y_val)
            f1_score = f1_score(y_true=y_true.tolist(), y_pred=y_pred.tolist(), 
                                average='macro', pos_label=None)   

            return acc, f1_score     

        return acc

    def metrics(self, y_true, y_pred):
        y_true = torch.argmax(y_true, axis=1).tolist()

        prec = precision_score(y_true=y_true, y_pred=y_pred, average=None, pos_label=None)
        prec_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro', pos_label=None)
        rec = recall_score(y_true=y_true, y_pred=y_pred, average=None, pos_label=None) 
        rec_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro', pos_label=None)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', pos_label=None)   

        return prec, prec_macro, rec, rec_macro, f1
        
