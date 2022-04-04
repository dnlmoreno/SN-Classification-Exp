import argparse


def get_args(input_size, num_classes):

    parser = argparse.ArgumentParser()  # description="SNIa classification")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to be used")
    parser.add_argument("--fold", type=int, default=0, 
                        help="Fold to be used")

    # Data --> no me esta sirviendo
    parser.add_argument("--num_augment", type=int, default=5,
                        help="Numero de aumentaciones de los datos")

    #••••••••••••••••••••••••••••#
    #•••••••    MODELO    •••••••#
    #••••••••••••••••••••••••••••#

    # Data Loader
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Tamanio de cada mini-batch")

    # Compile
    parser.add_argument("--loss_list", type=list,
                        default=['crossentropy'],
                        choices=[['crossentropy']],
                        help="Losses function a utilizar")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str,
                        default='adam', help="Optimizador")

    # Network arquitecture
    parser.add_argument("--rnn_type", type=str, default='LSTM',
                        choices=['RNN', 'LSTM'], help="Unidad recurrente")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Probabilidad aplicada en la salida de cada RNN")
    ### Valores determinados por la data usada ###
    parser.add_argument("--input_size", type=int,
                        default=input_size, help="Numero de features")
    parser.add_argument("--num_classes", type=int,
                        default=num_classes, help="Numero de clases")

    parser.add_argument("--hidden_size", type=int,
                        default=16, help="Numero de neuronas")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Numero de capas ocultas")
    parser.add_argument("--activation", type=str, default='tanh',
                        help="Función de activación para la capa final")
    parser.add_argument("--batch_norm", type=bool,
                        default=False, help="Batch normalization")

    # Fit
    parser.add_argument("--type_train", type=str,
                        choices=['normal', 'kd'],
                        default='normal', help="Tipo de entrenamiento a utilizar")    
    parser.add_argument("--num_epochs", type=int,
                        default=200, help="Numero de epocas")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Imprime la fase de entrenamiento")

    # Early stopping
    parser.add_argument("--patience", type=int, default=200,
                        help="Numero de epocas para activar early stoping")
    parser.add_argument("--metric_eval", type=str, default='accuracy',
                        help="Numero de epocas para activar early stoping")
    parser.add_argument("--name_model", type=str, default='charnock',
                        help="Nombre del modelo a utilizar")

    # Load and save experiment
    parser.add_argument("--experiment", type=str, default='/exp_1_charnock_results/', 
                        help="Nombre del experimento a realizar")
    parser.add_argument("--load_file", type=str, default='save_results',
                        help="file to load params from")
    parser.add_argument("--save_file", type=str, default='save_results',
                        help="file to save params to / load from if not loading from checkpoint")

    # Pre-trained model
    parser.add_argument("--load_model", type=bool,
                        default=False, help="Load pre-trained model") 
    parser.add_argument("--model_path", type=str, default='/best_result/ckpt-181-0.9502.pt', 
                        help="relative path del modelo desde el experimento")


    args = parser.parse_args(args=[])

    return args
