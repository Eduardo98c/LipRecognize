# 0. Carica i pacchetti keras necessari
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os  # drectory library
import cv2  # image processing library
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import TimeDistributed
from keras import applications, regularizers
from keras import optimizers
from keras.models import Model
import keras.backend as K
import seaborn as sns

# Fix random seed
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(3)

timesteps = 20  # Numero di frame in input per la LSTM
n_labels = 256  # Numero di Dataset_input Labels
Learning_rate = 0.0001  # Ottimizzatore learning rate in questo caso per adam
batch_size = 32
validation_ratio = 0.2
num_epochs = 50  # valore precedente : 50

img_row = 192  # Transfer model input size ( MobileNet )
img_col = 192  # Transfer model input size ( MobileNet )
img_channel = 3  # RGB

lastLayer = img_row


def Create_validation_set(validation_path, n_labels):
    x_val = []
    y_val = []

    # caricamento del dataset
    for i in range(n_labels):
        nb = 0

        # conteggio del numero di dataset in ogni classe
        for root, dirs, files in os.walk(validation_path + '/' + str(i) + '/'):
            for name in dirs:
                nb = nb + 1
        print("Numero di dataset nella label", i, ":", nb)

        # carica i dati dalle immagini
        for j in range(nb):
            temp = []
            for k in range(timesteps):
                name = validation_path + '/' + str(i) + '/' + str(j) + '/' + str(k) + '.jpg'
                img = cv2.imread(name)
                res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
                temp.append(res)
            x_val.append(temp)
            y_val.append(i)

    x_val = np.array(x_val)
    y_val = np.eye(n_labels)[y_val]

    return x_val, y_val


def Train_dataset(dataset_input):
    # 1. Creazione Datasets
    # definisco delle liste vuote per il caricamento (dati, label e numero totale di dati)
    data = []
    label = []
    Totalnb = 0
    num_feature = 0

    # caricamento del dataset
    for i in range(n_labels):
        nb = 0

        # conteggio del numero di dataset in ogni classe
        for root, dirs, files in os.walk(dataset_input + '/' + str(i) + '/'):
            for name in dirs:
                nb = nb + 1
        print("Numero di dataset nella label", i, ":", nb)
        Totalnb = Totalnb + nb

        # carica i dati dalle immagini
        for j in range(nb):
            temp = []
            for k in range(timesteps):
                name = dataset_input + '/' + str(i) + '/' + str(j) + '/' + str(k) + '.jpg'
                img = cv2.imread(name)
                res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
                num_feature = num_feature + 1
                temp.append(res)
            label.append(i)
            data.append(temp)
    print("Numero totale di dati:", Totalnb)
    print("Numero totale di feature:", num_feature)

    # Convertire l'elenco in array numpy, per l'uso in Keras
    train_label = np.eye(n_labels)[label]  # One-hot encoding by np array function
    train_data = np.array(data)
    print("Dataset_input_video shape is", train_data.shape, "(size, timestep, column, row, channel)")
    print("Label shape is", train_label.shape, "(size, label onehot vector)")

    # Rimescolamento del set di dati per la funzione di input fit,
    # se non lo si fa, non si può addestrare interamente il modello
    x = np.arange(train_label.shape[0])
    np.random.shuffle(x)

    # è necessario un rimescolamento(shuffle) dello stesso ordine
    train_label = train_label[x]
    train_data = train_data[x]

    # Dichiara i dati per l'addestramento e la convalida,è possibile separare il set di test da questo
    x_train = train_data[0:Totalnb, :]
    y_train = train_label[0:Totalnb]

    x_val, y_val = Create_validation_set('Dataset2/Validation_Set_Normalized', n_labels=n_labels)  # 13% del dataset

    # Dichiarazione dell'input layer per l'architettura CNN+LSTM
    video = Input(shape=(timesteps, img_col, img_row, img_channel))

    # Carica il modello di apprendimento del trasferimento che desideri
    model = applications.MobileNet(input_shape=(img_col, img_row, img_channel), weights="imagenet", include_top=False)
    model.trainable = False

    # FC Dense Layer

    x = model.output

    # Flatten: Appiattisce l'input in un vettore unidimensionale
    x = Flatten()(x)

    # L'obiettivo principale di ReLU è introdurre la non linearità nel modello, 
    # in modo da consentire al modello di apprendere relazioni complesse nei dati.
    # se il valore di input è positivo, la funzione ReLU restituisce 
    # lo stesso valore, altrimenti restituisce zero

    # Dataset 1: kernel regularizer = 0.00275;  Dataset 2: kernel regularizer = 0.1
    x = Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.00275))(x)
    x = Dropout(0.3)(x)
    cnn_out = Dense(lastLayer, activation="relu")(x)

    # Costruisce il modello CNN
    Lstm_inp = Model(model.inputs, cnn_out)

    # Distribuisce l'output della CNN per fasi temporali
    encoded_frames = TimeDistributed(Lstm_inp)(video)

    # Costruisce il modello LSTM

    # Numero di unità LSTM all'interno del layer LSTM (iperparametro scelto in base ai dati di input)
    encoded_sequence = LSTM(256)(encoded_frames)
    hidden_Drop = Dropout(0.3)(encoded_sequence)

    # Dataset 1: kernel regularizer = 0.00275;  Dataset 2: kernel regularizer = 0.1
    hidden_layer = Dense(lastLayer, activation="relu", kernel_regularizer=regularizers.l2(0.00275))(
        encoded_sequence)  # (encoded_sequence)

    # La funzione Softmax è utile per ottenere una distribuzione di probabilità tra le diverse classi, 
    # consentendo di effettuare una classificazione finale basata sulla classe con la probabilità più alta

    outputs = Dense(n_labels, activation="softmax")(hidden_layer)

    # Contruzione del modello CNN+LSTM
    model = Model([video], outputs)

    # 3. Settaggi di apprendimento del modello

    # Compilazione del modello
    adam = keras.optimizers.Adam(Learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    # 4. Addestramento del modello

    # Early stopping utile a terminare l'addestramento appena il
    # val_accuracy non aumenta per N volte di fila di una certa percentuale
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)

    hist = model.fit(x_train, y_train, batch_size=batch_size, validation_split=validation_ratio, shuffle=True,
                     epochs=num_epochs, callbacks=[es])

    # hist = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), shuffle=True,epochs=num_epochs,callbacks=[es])

    # Visualizzazione dell'architettura del modello
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot

    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    from keras.utils import plot_model

    plot_model(model, to_file='model.png')

    # 5. Conferma del processo di addestramento
    import matplotlib.pyplot as plt

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    acc_ax.set_ylabel('accuracy')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()

    # 6. Calcola la matrice di confusione
    y_pred = model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=1)

    y_true = np.argmax(y_val, axis=1)

    # 7. Calcola le metriche di precisione, richiamo e f1-score
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)

    # Mostra il grafico
    plt.show()

    # Crea il DataFrame per le metriche
    df = pd.DataFrame(report).transpose()
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)

    # Salva il DataFrame in un file CSV
    df.to_csv('metric.csv', index_label='Label')

    print(report)

    # Accuracy score
    from sklearn.metrics import accuracy_score
    print('accuracy is', accuracy_score(y_pred, y_true))

    # 6. Salva il modello
    model.save(
        'Dataset1/Models_normalized_60_frames/EarlyStopModels/Lib_Reading_20Frame_192_Modello3_Normalized_ES_L2reg.h5')
