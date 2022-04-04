from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_learning_curves(train, val, metric, labels_curves, color_curves, loc='upper right', path_save=''):
    curves = [train, val]
    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(11,8))

    for i in range(len(curves)):
        plt.plot(epochs, curves[i], color=color_curves[i], label=f'{labels_curves[i]} {metric.lower()}') #, linewidth=2.5, markersize=7)

    plt.xticks(fontsize = 15)
    plt.yticks(size = 15) # Arreglar este tamaño

    plt.title(f'Training and validation {metric.lower()}', size='20', pad=15)
    plt.xlabel('Epochs', fontsize='20', labelpad=15)
    plt.ylabel(f'{metric}', fontsize='20', labelpad=15)

    legend = plt.legend(loc=loc, shadow=True, fontsize='15')
    legend.get_frame().set_facecolor('white')

    if path_save != '':
        #plt.savefig("../6. Results/spcc/three_filter/[16,16]/grafico_loss.png", bbox_inches='tight')
        #path = os.path.abspath(model_root+'_grafico_loss.png')
        #plt.savefig(path, bbox_inches='tight')
        pass

    plt.show()


def plot_cm(y_true, y_pred, label_order, dict_labels, figsize=(12, 10), save_path=None):
    y_true, y_pred = replace_for_visualization(y_true, y_pred, dict_labels)
    cm_matrix = confusion_matrix(y_true, y_pred, 
                                 labels=label_order)
                 
    #cm_classes = reverse_replace(dict_labels, cm_classes)
    plot_confusion_matrix(cm_matrix, label_order, figsize, save_path)


def plot_confusion_matrix(cm, classes, figsize,
                          plot_name,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 17)
    plt.yticks(tick_marks, classes, fontsize = 17)

    #fmt = '.2f' if normalize else 'd'
    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%d"%  (cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 16)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    
    if plot_name is not None:
        plt.savefig(plot_name, bbox_inches='tight')
        
    plt.show()


def replace_for_visualization(y_true, y_pred, dict_labels):
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for idx in range(len(y_true)):
        y_true[idx] = dict_labels[y_true[idx]]
        y_pred[idx] = dict_labels[y_pred[idx]]

    return y_true, y_pred