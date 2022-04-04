import torch
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import utils.plots

def print_metrics(y_test, y_pred, y_pred_prob, loss, dict_labels, one_hot, save=''):
    if one_hot:
        y_true_onehot = y_test
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true_onehot = torch.nn.functional.one_hot(y_test)
        y_true = y_test

    y_true, y_pred = utils.plots.replace_for_visualization(y_true, y_pred, dict_labels)

    auc = roc_auc_score(y_true_onehot, y_pred_prob, average='macro')
    acc = accuracy_score(y_true, y_pred)
    acc_balanced = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, digits=3)

    print('Loss:', loss)
    print("Accuracy:", "%0.5f" % acc)
    print("Accuracy balanced:","%0.5f" %  acc_balanced)
    
    print('AUC:', auc)
    print("macro precision: ","%0.5f" %  prec_macro)
    print("macro recall: ","%0.5f" %  rec_macro)
    print("macro F1: ","%0.5f" %  f1)

    print(f'\n{report}')

    if save != '':
        # SE GUARDAN LOS RESULTADOS Y PREDICCIONES DEL MODELO
        with open(save + '_results.txt', 'w') as f:
            f.write('Test loss: {0}\n'.format(loss))
            f.write('Test accuracy: {0}\n'.format(acc))
            f.write('Accuracy balanced: {0}\n'.format(acc_balanced))
            f.write('AUC: {0}\n'.format(auc))
            f.write('Macro precision: {0}\n'.format(prec_macro))
            f.write('Macro recall: {0}\n'.format(rec_macro))
            f.write('F1: {0}\n'.format(f1))
            f.write('Report: {0}\n'.format(report))