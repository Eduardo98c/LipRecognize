# import Train
import cv2
import pydot
import tensorflow as tf
import visualkeras as visualkeras
from IPython.core.display import SVG
from matplotlib import pyplot as plt
from keras.models import Model
from pygments.lexers import graphviz
from keras.utils.vis_utils import model_to_dot, plot_model
import mayavi.mlab as mlab
from PIL import Image
from keras.models import load_model
import pyvista as pv
import networkx as nx

from Preprocessing import Preprocessing as Prep
import TrainLip_CNN_LSTM
import Predict_lips

# Path Training set Dataset 1
path_training_set_video_dataset1 = "Dataset1/Training_set_video"
path_training_set_dataset1 = "Dataset1/Training_set"
path_training_set_normalized_30_frames_dataset1 = "Dataset1/Training_set_normalized_25_30_frames"
path_training_set_normalized_60_frames_dataset1 = "Dataset1/Training_set_Normalized_60_frames"

# Path Test set Dataset1
path_test_set_dataset1 = "Dataset1/Test_set"
path_test_set_normalized_dataset1 = "Dataset1/Test_set_normalized"

# Path Models dataset1
path_model_20Frame_192_normalized = "Dataset1/Models_normalized_60_frames/Prove_overfitting/Lib_Reading_20Frame_192_Modello1_Normalized.h5"
path_model_20Frame_192_modello2_original = "Dataset1/Models_frames_original/Modello2_Original/Lib_Reading_20Frame_192_Modello2_Original.h5"

# Path Training set Dataset2
path_training_set_video_dataset2 = "Dataset2/Training_Set"
path_training_set_normalized_dataset2 = "Dataset2/Training_Set_Normalized"

# Path Validation set Dataset2
path_validation_set_dataset2 = "Dataset2/Validation_Set"
path_validation_set_normalized_dataset2 = "Dataset2/Validation_Set_Normalized"

# Path Test set Dataset2
path_test_set_dataset2 = "Dataset2/Test_Set"
path_test_set_normalized_dataset2 = "Dataset2/Test_Set_Normalized"

if __name__ == '__main__':
    Prep = Prep()

    # Prep.define_number_label('Dataset2/Test_Set')
    # Prep.Preprocessing_label('Train', 'Training_set_video')

    # sampling_rate: 5 = 60 frames, 10 = 30 frames
    # Prep.Preprocessing_datasetMultiThread('Dataset2/Training_Set', 'Dataset2/Training_Set_Normalized', None, False)

    # Addestramento modello per dataset 1
    # TrainLip_CNN_LSTM.Train_dataset(path_training_set_normalized_60_frames_dataset1)

    # Addestramento modello per dataset 2
    # TrainLip_CNN_LSTM.Train_dataset(path_training_set_normalized_dataset2)

    # Predict Dataset 1
    # Predict_lips.predict_dataset_lips(path_test_set_normalized_dataset1,'Dataset1/Models_normalized_60_frames/EarlyStopModels/Modello2_Normalized_ES44_L2reg/Lib_Reading_20Frame_192_Modello2_Normalized_ES44_L2reg.h5',192, 192, 256)
    model = load_model(
        'Dataset1/Models_normalized_60_frames/EarlyStopModels/Modello2_Normalized_ES44_L2reg/Lib_Reading_20Frame_192_Modello2_Normalized_ES44_L2reg.h5')
    print(model.summary())
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

    # Genera una rappresentazione grafica dell'architettura del modello
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    visualkeras.layered_view(model=model, legend=True, spacing=60, draw_funnel=True,to_file='grafico3D_CNN.png').show()  # write and show
    visualkeras.graph_view(model= model ,show_neurons=True,to_file='grafico_reteNeurale_CNN.png').show()


    # Predict_lips.predict_dataset_lips_with_folder(path_test_set_normalized_dataset1,"Dataset1/Models_normalized_60_frames/EarlyStopModels/Lib_Reading_20Frame_192_Modello2_Normalized_ES44_L2reg.h5", 192, 192, n_labels=256, output_predict_dir="outputError_dataset1")

    # Predict Dataset 2

    # Predict_lips.predict_dataset_lips('Dataset2/Test_Set_Normalized', 'Dataset2/Models_normalized/EarlyStopModels/Modello5_Normalized_ES15_2_L2reg/Lib_Reading_20Frame_192_Modello5_Normalized_ES15_2_L2reg.h5', 192, 192, 43)

    # Predict_lips.predict_dataset_lips_with_folder(path_test_set_normalized_dataset2,"Dataset2/Models_normalized/EarlyStopModels/Lib_Rading_20Frame_192_Modello4_Normalized_ES18_L2reg.h5",192, 192, n_labels=43, output_predict_dir="output_predict")

    # class_predicted = Predict_lips.predict_image_lips('Dataset2/Test_Set_Normalized/35/0/109.jpg',"Dataset2/Lib_Rading_20Frame_192_Modello1_Normalized.h5", 192, 192)
    # print(class_predicted)
    # classe_di_maggioranza, contatoreClassi = Predict_lips.predict_subject_fromVideo_MultiThreading('Test/1_1_1_11_30_3_m.avi', path_model_20Frame_192_normalized, 192, 192)
    # print(classe_di_maggioranza)
    # print(contatoreClassi)

    print("fine")
