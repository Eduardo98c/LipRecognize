# Load keras package needed

import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import cv2
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from Preprocessing import Preprocessing

from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

timesteps = 20

import concurrent.futures


class FrameErrato:
    def __init__(self, nome_frame, classe_predetta, classe_reale, immagine, index):
        self.nome_frame = nome_frame
        self.classe_predetta = classe_predetta
        self.classe_reale = classe_reale
        self.immagine = immagine
        self.index_img = index

    def salvare_immagine(self, percorso_salvataggio):
        cv2.imwrite(percorso_salvataggio, self.immagine)


def get_object_by_index(objects_list, index):
    for obj in objects_list:
        if obj.index_img == index:
            return obj
    return None


def preprocess_frame(frame, img_row, img_col):
    Prep = Preprocessing()
    frame = Prep.preprocessing_meanShift(frame)
    res = cv2.resize(frame, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
    return res


def predict_subject_fromVideo_MultiThreading(input_video, path_model, img_row, img_col):

    # Carica il modello
    model = load_model(path_model)

    # Inizializza il contatore di classi
    class_counts = [0] * 256

    # Apre il video
    cap = cv2.VideoCapture(input_video)

    # Legge i frames
    frame_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            future = executor.submit(preprocess_frame, frame, img_row, img_col)
            futures.append(future)

            frame_count += 1

        # Estrai i risultati dai futuri e utilizza il predittore
        for future in concurrent.futures.as_completed(futures):
            input_data = np.array([future.result()] * timesteps)
            input_data = np.expand_dims(input_data, axis=0)

            # Utilizza il predittore
            y_data = model.predict(input_data, batch_size=1)

            # Predice la classe dal modello
            y_classes = y_data.argmax(axis=-1)
            predicted_class = y_classes[0]

            # Aggiorna il contatore di classi
            class_counts[predicted_class] += 1

    # Release video capture
    cap.release()

    # Calcola la maggioranza e predice la classe del video
    majority_class = np.argmax(class_counts)

    return majority_class, class_counts


def predict_subject_fromVideo(input_video, path_model, img_row, img_col):
    # Carica il modello
    model = load_model(path_model)

    # Inizializza il contatore di classi
    class_counts = [0] * 256

    # Apre il video
    cap = cv2.VideoCapture(input_video)

    # Legge i frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessa i frames
        Prep = Preprocessing()
        frame = Prep.preprocessing_meanShift(frame)

        res = cv2.resize(frame, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
        input_data = np.array([res] * timesteps)
        input_data = np.expand_dims(input_data, axis=0)

        # Utilizza il predittore
        y_data = model.predict(input_data, batch_size=1)

        # Predice la classe dal modello
        y_classes = y_data.argmax(axis=-1)
        predicted_class = y_classes[0]

        # Aggiorna il contatore di classi
        class_counts[predicted_class] += 1

        frame_count += 1

    # Release video capture
    cap.release()

    # Calcola la maggioranza e predice la classe del video
    majority_class = np.argmax(class_counts)

    return majority_class, class_counts


# Predice da un'immagine, la classe di appartenenza
def predict_image_lips(input_image, path_model, img_row, img_col):
    # Carica il modello
    model = load_model(path_model)

    # Preprocessa l'immagine di input
    # img = cv2.imread(input_image)
    # img = input_image
    # Prep = Preprocessing()
    # img = Prep.preprocessing_meanShift(img)

    # res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
    input_data = np.array([input_image] * timesteps)
    input_data = np.expand_dims(input_data, axis=0)

    # Usa la probabilità della classe del modello
    y_data = model.predict(input_data, batch_size=20)

    # Usa la classe del modello
    y_classes = y_data.argmax(axis=-1)

    return y_classes[0]


def predict_dataset_lips(dataset_input, path_model, img_row, img_col, n_labels):
    # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):

    model = load_model(path_model)

    data = []
    label = []
    Totalnb = 0
    data_nb = [0] * n_labels

    countFrame = 0

    # Carica il Training_set
    for i in range(n_labels):
        nb = 0
        for root, dirs, files in os.walk(dataset_input + '/' + str(i) + '/'):  # set directory
            for name in dirs:
                nb = nb + 1

        print("Numero di dataset nella label", i, ":", nb)
        Totalnb = Totalnb + nb
        data_nb[i] = data_nb[i] + nb
        for j in range(nb):
            temp = []
            for k in range(timesteps):
                name = dataset_input + '/' + str(i) + '/' + str(j) + '/' + str(k) + '.jpg'
                img = cv2.imread(name)
                res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
                temp.append(res)
            label.append(i)
            data.append(temp)

    print("Total Number of Data is", Totalnb)
    label = np_utils.to_categorical(label, n_labels)
    data = np.array(data)

    x = np.arange(label.shape[0])
    np.random.shuffle(x)
    label = label[x]
    data = data[x]

    npp = 0
    n = {}

    for i in range(label.shape[0]):

        # Testa il modello per ciascuna sequenza temporale dell'immagine
        train_data_one = data[i]
        train_data_one = np.expand_dims(train_data_one, axis=0)

        # Usa la probabilità della classe del modello
        y_data = model.predict(train_data_one, batch_size=1)


        # Usa la classe del modello
        y_classes = y_data.argmax(axis=-1)

        # risultato di stampa etichetta di input della classe/modello previsto ( One_hot Code )
        print(y_classes, label[i])

        if label[i][y_classes] == 1:
            npp = npp + 1
            key = tuple(y_classes.tolist())  # Convert array to tuple
            if key in n:
                n[key] += 1
            else:
                n[key] = 1


    y_pred = model.predict(data)
    y_pred = np.argmax(y_pred, axis=1)

    y_true = np.argmax(label, axis=1)

    # 7. Calcola le metriche di precisione, richiamo e f1-score
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)


    conf_matrix = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')

    fig, ax1 = plt.subplots(figsize=(15, 15))
    conf_matrix.ax_.tick_params(labelsize=1.5, pad=5)
    conf_matrix.plot(ax=ax1)

    conf_matrix_norm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', normalize='true')
    conf_matrix_norm.ax_.tick_params(labelsize=1.5, pad=5)
    fig, ax2 = plt.subplots(figsize=(15, 15))
    conf_matrix_norm.plot(ax=ax2)

    # Aggiungi titolo e etichette agli assi
    plt.title('Matrice di Confusione Normalizzata')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')


    # Crea il DataFrame per le metriche
    df = pd.DataFrame(report).transpose()
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)

    # Salva il DataFrame in un file CSV
    df.to_csv('metric.csv', index_label='Label')

    #print(report)

    labels = list(report.keys())
    labels.remove('accuracy')
    labels.remove('macro avg')
    labels.remove('weighted avg')

    metrics = ['precision', 'recall', 'f1-score']
    scores = {}

    for label_i in labels:
        scores[label_i] = {metric: report[label_i][metric] for metric in metrics}

    precision = [scores[label]['precision'] for label in labels if isinstance(label, str)]
    recall = [scores[label]['recall'] for label in labels if isinstance(label, str)]
    f1_score = [scores[label]['f1-score'] for label in labels if isinstance(label, str)]

    # Crea il grafico a barre
    fig, ax = plt.subplots(figsize=(12, 6))
    # Calcola la posizione delle barre
    x = np.arange(len(labels))

    # Larghezza delle barre e spazio tra le barre
    bar_width = 0.3
    bar_spacing = 0.05

    # Crea il grafico a barre
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, precision, bar_width, align='edge', label='Precision')
    ax.bar(x + bar_width + bar_spacing, recall, bar_width, align='edge', label='Recall')
    ax.bar(x + 2 * (bar_width + bar_spacing), f1_score, bar_width, align='edge', label='F1-score')

    # Imposta i ticks sull'asse x
    ax.set_xticks(x + bar_width + bar_spacing)
    ax.set_xticklabels(labels)

    # Aggiungi una legenda
    ax.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    # Calcola l'accuratezza e stampa i risultati
    total_samples = label.shape[0]
    accuracy = npp / total_samples

    # Stampa i risultati totali
    print("Number of correct answers is", npp, "into", total_samples, "Test Acc is", accuracy)
    print("Number of correct answers for each class is", ", ".join(str(n.get(tuple([i]), 0)) for i in range(n_labels)))
    print("Accuracy of each class is", ", ".join(str(n.get(tuple([i]), 0) / data_nb[i]) for i in range(n_labels)))



def predict_dataset_lips_with_folder(Test_Set_input, path_model, img_row, img_col, n_labels, output_predict_dir):
    # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):

    model = load_model(path_model)

    # Creazione della cartella "failure_frames" se non esiste
    failure_frames_dir = output_predict_dir+"/failure_frames"
    if not os.path.exists(failure_frames_dir):
        os.makedirs(failure_frames_dir)

    csv_dir = output_predict_dir+"/CSV"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    data = []
    label = []
    Totalnb = 0
    data_nb = [0] * n_labels

    failure_frames_data = []

    index_failure = []

    # Carica il Training_set
    for i in range(n_labels):
        nb = 0
        for root, dirs, files in os.walk(Test_Set_input + '/' + str(i) + '/'):
            for name in dirs:
                nb = nb + 1

        print("Numero di dataset nella label", i, ":", nb)
        Totalnb = Totalnb + nb
        data_nb[i] = data_nb[i] + nb
        for j in range(nb):
            temp = []
            for k in range(300):
                if os.path.exists(Test_Set_input + '/' + str(i) + '/' + str(j) + '/' + str(k) + '.jpg'):
                    name = Test_Set_input + '/' + str(i) + '/' + str(j) + '/' + str(k) + '.jpg'
                    img = cv2.imread(name)
                    res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
                    temp.append(res)
            label.append(i)
            data.append(temp)

    print("Total Number of Data is", Totalnb)
    label = np_utils.to_categorical(label, n_labels)
    data = np.array(data)

    x = np.arange(label.shape[0])
    np.random.shuffle(x)
    label = label[x]
    data = data[x]

    npp = 0
    n = {}

    list_y_classes = []
    class_correct = [0] * n_labels
    class_total = [0] * n_labels

    for i in range(label.shape[0]):
        frames = data[i]  # Frames per la label corrente
        num_groups = len(frames) // timesteps  # Numero di gruppi di 20 frames

        correct_predictions = 0  # Contatore delle previsioni corrette per il campione i
        start = None
        end = None

        for group in range(num_groups):
            start = group * timesteps
            end = start + timesteps
            group_frames = frames[start:end]  # Prendi il gruppo di 20 frames

            group_data = []
            for frame in group_frames:
                group_data.append(frame)

            train_data_one = np.array(group_data)
            train_data_one = np.expand_dims(train_data_one, axis=0)

            y_data = model.predict(train_data_one, batch_size=1)
            y_classes = y_data.argmax(axis=-1)

            list_y_classes.append(y_classes)

            print(y_classes, label[i])
            if label[i][y_classes] == 1:
                correct_predictions += 1

                key = tuple(y_classes.tolist())
                if key in n:
                    n[key] += 1
                else:
                    n[key] = 1

        if correct_predictions >= (num_groups / 2):  # Controlla se la maggioranza delle previsioni è corretta
            npp += 1
            class_correct[label[i].argmax()] += 1

        class_total[label[i].argmax()] += 1

        if correct_predictions < num_groups:  # Controlla se la previsione è errata per almeno un gruppo
            index_failure.append(i)
            for group in range(num_groups):
                start = group * timesteps
                end = start + timesteps
                for k in range(start, end):
                    y_classes_group = list_y_classes[group]
                    if label[i].argmax() != y_classes_group.item():  # Controlla se la previsione è errata per il frame
                        frameErrato_temp = FrameErrato(nome_frame=f"{k}.jpg",
                                                       classe_predetta=y_classes_group.item(),
                                                       classe_reale=label[i].argmax(), immagine=data[i][k], index=i)
                        failure_frames_data.append(frameErrato_temp)

        list_y_classes.clear()
    # Calcola l'accuratezza per ogni classe
    class_accuracy = [class_correct[j] / class_total[j] if class_total[j] != 0 else 0 for j in range(n_labels)]

    # Calcola l'accuratezza e stampa i risultati
    total_samples = label.shape[0]
    accuracy = npp / total_samples

    # Stampa i risultati totali
    print("Number of correct answers is", npp, "out of", total_samples, "Test Acc is", accuracy)
    print("Number of correct answers for each class is", ", ".join(str(class_correct[i]) for i in range(n_labels)))
    print("Accuracy of each class is", ", ".join(str(class_accuracy[i]) for i in range(n_labels)))

    # failure_frames_data = np.array(failure_frames_data)
    # Salva i frame non identificati correttamente

    for frame_errato in failure_frames_data:
        label_failure_dir = os.path.join(failure_frames_dir, "classe_reale_"+ str(frame_errato.classe_reale))
        os.makedirs(label_failure_dir, exist_ok=True)

        class_predicted_path= os.path.join(label_failure_dir,"classe_predetta_"+str(frame_errato.classe_predetta))

        if not os.path.exists(class_predicted_path):
            os.makedirs(class_predicted_path)

        frame_path = os.path.join(class_predicted_path, frame_errato.nome_frame)
        frame = frame_errato.immagine
        cv2.imwrite(frame_path, frame)

    # Crea il file CSV
    csv_file_path = os.path.join(csv_dir, 'failure_frames.csv')
    df = pd.DataFrame({
        'Frame': [frameErrato.nome_frame for frameErrato in failure_frames_data],
        'Predicted Class': [frameErrato.classe_predetta for frameErrato in failure_frames_data],
        'Real Class': [frameErrato.classe_reale for frameErrato in failure_frames_data]
    })
    df.to_csv(csv_file_path, index=False, sep=';')
