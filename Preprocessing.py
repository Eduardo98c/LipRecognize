import concurrent.futures
import os
from typing import List

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import shutil

# lista di estensioni video supportate
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder

video_extensions = [".mp4", ".avi", ".mkv", ".mov"]


class Preprocessing:

    def __init__(self):
        pass

    def _extract_label(self, filename):
        count = 0
        nameFileOut = ''
        for digit in filename:
            if count < 4:
                if digit == '_':
                    count += 1
                else:
                    nameFileOut += digit

        return nameFileOut

    def _oneHotEncoding(self, listLabels):

        # Inizializza il OneHotEncoder
        # encoder = OneHotEncoder()

        # codifica univoca per le label
        encoder = LabelEncoder()

        # Trasforma la lista mondimensionale in array 2D con una sola colonna
        labels_2d = np.array(listLabels).reshape(-1, 1)

        # # Esegue la codifica one-hot della matrice bidimensionale
        encoded_data = encoder.fit_transform(labels_2d)

        # Print the encoded data
        # print(encoded_data.toarray())

        return encoded_data  # .toarray()

    def define_number_label(self, dataset_dir):

        # Elenco delle cartelle label presenti nel dataset
        labels = os.listdir(dataset_dir)

        # Rinomina le label da 0 a N
        for i, label_name in enumerate(labels):
            old_label_dir = os.path.join(dataset_dir, label_name)
            new_label_name = str(i)
            new_label_dir = os.path.join(dataset_dir, new_label_name)
            os.rename(old_label_dir, new_label_dir)

            # Sposta i video nella label principale, elimina la cartella "video" e le altre sottocartelle
            video_dir = os.path.join(new_label_dir, 'video')

            # Verifica se la cartella "video" esiste
            if os.path.exists(video_dir):
                # Sposta i video nella label principale
                for sub_dir_name in os.listdir(video_dir):
                    sub_dir = os.path.join(video_dir, sub_dir_name)
                    if os.path.isdir(sub_dir):
                        for video_name in os.listdir(sub_dir):
                            video_path = os.path.join(sub_dir, video_name)
                            new_video_path = os.path.join(new_label_dir, video_name)
                            shutil.move(video_path, new_video_path)

                # Elimina la cartella "video" e le altre sottocartelle
                shutil.rmtree(video_dir)

            # Numerazione dei video all'interno della label principale
            video_counter = 0
            for video_name in sorted(os.listdir(new_label_dir)):
                video_path = os.path.join(new_label_dir, video_name)
                if os.path.isfile(video_path):
                    new_video_name = str(video_counter) + '.avi'  # Esempio: 0.avi, 1.avi, 2.avi, ...
                    new_video_path = os.path.join(new_label_dir, new_video_name)
                    os.rename(video_path, new_video_path)
                    video_counter += 1

        print("Operazioni completate!")

    def Preprocessing_label(self, directory_input_path_video, directoryOutput):

        # Crea la cartella per il dataset
        if not os.path.exists(directoryOutput):
            os.makedirs(directoryOutput)
        i = 0
        listLabels = []
        listFilePath = []
        for filename in os.listdir(directory_input_path_video):
            # ottiene il percorso completo del file
            file_path = os.path.join(directory_input_path_video, filename)

            # controlla se il file ha un'estensione video supportata
            if os.path.isfile(file_path) and os.path.splitext(filename)[1] in video_extensions:
                # il file è un video, quindi possiamo fare qualcosa con esso
                print("Nome file: ", filename)
                print("Percorso file: ", file_path)

                labelTemp = self._extract_label(filename)

                listLabels.append(labelTemp)
                listFilePath.append(file_path)
                i = i + 1

        labelsEncoded = self._oneHotEncoding(listLabels)

        for label, file_path in zip(labelsEncoded, listFilePath):

            label_i = str(label)  # .replace('. ', '').replace('.', '').replace('\n ', '')
            print(label_i)
            print('file_path: ' + file_path)
            print(directoryOutput + '/' + label_i)
            if not os.path.exists(directoryOutput + '/' + label_i):
                os.makedirs(directoryOutput + '/' + label_i)

            shutil.copy(file_path, directoryOutput + '/' + label_i + '/')

    def Preprocessing_datasetMultiThread(self, directory_input_path_video, datasetOutputNormalized, sampling_rate, preprocessing_bool):

        # Crea la cartella per creare il dataset normalizzato
        if not os.path.exists(datasetOutputNormalized):
            os.makedirs(datasetOutputNormalized)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dir in os.listdir(directory_input_path_video):
                dir_path = os.path.join(directory_input_path_video, dir)
                dir_path_normalized = os.path.join(datasetOutputNormalized, dir)

                if not os.path.exists(dir_path_normalized):
                    os.makedirs(dir_path_normalized)

                video_files = [
                    (filename, os.path.join(dir_path, filename))
                    for filename in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, filename))
                       and os.path.splitext(filename)[1] in video_extensions
                ]

                i = 0
                for filename, file_path in video_files:
                    executor.submit(
                        self._preprocess_video,
                        file_path,
                        i,
                        dir_path_normalized,
                        sampling_rate,
                        preprocessing_bool
                    )
                    i += 1

    def Preprocessing_dataset(self, directory_input_path_video, datasetOutputNormalized, sampling_rate, preprocessing_bool):

        # Crea la cartella per creare il dataset normalizzato
        if not os.path.exists(datasetOutputNormalized):
            os.makedirs(datasetOutputNormalized)

        for dir in os.listdir(directory_input_path_video):

            dir_path = os.path.join(directory_input_path_video, dir)

            # Crea la cartella per creare la label_i nel dataset normalizzato
            dir_path_normalized = os.path.join(datasetOutputNormalized, dir)
            if not os.path.exists(dir_path_normalized):
                os.makedirs(dir_path_normalized)

            i = 0
            for filename in os.listdir(dir_path):

                # ottiene il percorso completo del file
                file_path = os.path.join(dir_path, filename)

                # controlla se il file ha un'estensione video supportata
                if os.path.isfile(file_path) and os.path.splitext(filename)[1] in video_extensions:
                    # il file è un video, quindi possiamo fare qualcosa con esso
                    print("Nome file: ", filename)
                    print("Percorso file: ", file_path)

                    self._preprocess_video(file_path, i, dir_path_normalized, sampling_rate, preprocessing_bool)
                    i = i + 1

    def _preprocess_video(self, input_video, index_video, directoryOutputNormalized, sampling_rate, preprocessing_bool):

        dir_video = str(index_video)

        num_frames_path = os.path.join(directoryOutputNormalized, dir_video)

        # Crea la cartella per i frame preprocessati
        if not os.path.exists(num_frames_path):
            os.makedirs(num_frames_path)

        print('dir:' + num_frames_path)

        # Apri il file video
        cap = cv2.VideoCapture(input_video)

        # Verifica se il video è stato aperto correttamente
        if not cap.isOpened():
            print("Errore nell'apertura del file video")

        # Leggi il primo frame
        ret, frame = cap.read()

        # Inizializza il contatore per i frame
        frame_count = 0
        frame_index = 0

        # Ciclo per processare tutti i frame del video
        while ret:
            # Applica il preprocessing
            if (sampling_rate is None) or (frame_index % sampling_rate == 0):

                if preprocessing_bool:
                   frame = self.preprocessing_meanShift(frame)


                # Salva l'immagine preprocessata nella cartella ''Dataset_normalized/' + labelName + '/'
                frame_path = os.path.join(num_frames_path + '/', '{}.jpg'.format(frame_count))

                cv2.imwrite(frame_path, frame)

                frame_count += 1

            # Incrementa il contatore dei frame
            frame_index += 1

            # Leggi il frame successivo
            ret, frame = cap.read()

        # Rilascia le risorse
        cap.release()
        cv2.destroyAllWindows()

    def preprocessing_meanShift(self, frame):

        # Applicazione del filtro di sfocatura gaussiano
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Converti l'immagine sfocata in HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Applica l'equalizzazione dell'istogramma al canale di luminanza (Value)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Converti l'immagine HSV in BGR
        equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Applicazione del filtro PyrMeanShift
        res = cv2.pyrMeanShiftFiltering(equalized, 20, 30)

        # Applicazione dell'otsu sulla componente luminanza (grigio)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)

        # Applicazione della maschera ottenuta all'immagine originale
        res = cv2.bitwise_and(res, res, mask=threshold)

        # res = cv2.Canny(res, 30, 100)
        return res

    def preprocessing_otsu(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        gray = cv2.GaussianBlur(gray, (5, 5), 2, borderType=cv2.BORDER_REPLICATE)

        # Applica l'algoritmo di Otsu per la prima soglia
        ret1, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Applica l'algoritmo di Otsu per la seconda soglia
        ret2, thresh2 = cv2.threshold(gray, ret1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        res = cv2.bitwise_or(gray, thresh1, thresh2)

        # res = cv2.Canny(res, 40, 70)
        cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        return res

    # Metodo per salvare tutti i frame originali dei video della directory "Train"
    def salva_immagini(self, directory_path_video):
        for filename in os.listdir(directory_path_video):
            # ottiene il percorso completo del file
            file_path = os.path.join(directory_path_video, filename)
            # controlla se il file ha un'estensione video supportata
            if os.path.isfile(file_path) and os.path.splitext(filename)[1] in video_extensions:
                # il file è un video, quindi possiamo fare qualcosa con esso
                print("Nome file: ", filename)
                print("Percorso file: ", file_path)

                self._salva_immagini_originali(file_path)

    def _salva_immagini_originali(self, path_video):
        # Apri il file video
        cap = cv2.VideoCapture(path_video)

        # Verifica se il video è stato aperto correttamente
        if not cap.isOpened():
            print("Errore nell'apertura del file video")

        # Crea la cartella per salvare le immagini
        if not os.path.exists('immagini_originali'):
            os.makedirs('immagini_originali')

        # Leggi il primo frame
        ret, frame = cap.read()

        # Inizializza il contatore per i frame
        frame_count = 0

        # Ciclo per processare tutti i frame del video
        while ret:
            # Salva l'immagine nella cartella 'immagini_originali'
            cv2.imwrite('immagini_originali/frame{:05d}.jpg'.format(frame_count), frame)
            # Leggi il frame successivo
            ret, frame = cap.read()

            # Incrementa il contatore dei frame
            frame_count += 1

        # Rilascia le risorse
        cap.release()
