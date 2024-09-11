# LipRecognize

Machine learning Biometrics project carried out by Eduardo Autore, Francesco Pio Iannuziello, Alessandro Macaro


To run the project follow the steps:

 Navigate to the "LipRecognize" folder and create a virtual environment for Python:

   python3 -m venv venv (Mac)

   python -m venv venv (Windows)
   
 Once the virtual environment has been created, perform the following steps:

   Go to "Add Configuration" in PyCharm (python configuration)

   Select "Add New Configuration" 

   Then set Target type to Script patch,

   Select the "app.py" file in Target.

   Run the commands to activate the virtual environment:

   venv \ Scripts \ activate (Windows)

   source venv / bin / activate (Mac)
   
 Install the libraries with the following commands:

   pip install -r requirements.txt 

 
 To launch project use:
    
   python3 main.py or run Main in run/debug configuration


Istructions for Preprocessing:
        
        1) methods of Class 'Preprocessing' : 
             
           - Preprocessing_label(self, directory_input_path_video, directoryOutput)
             
                - takes an input video dataset (type 1 Babele) with this structure:  DatasetName-> '1_1_1_11_30_1_m.avi', '1_1_1_11_30_2_m.avi'...    
             
                - outputs a video dataset(with label) with this structure: DatasetOut-> (Label 0...N) -> video_i.avi...N (example: '1_1_1_11_30_1_m.avi' TO '1_1_1_11_30_5_m.avi').

           
           - define_number_label(self, dataset_dir)

                - Takes an input dataset (type2 VidTimit) with this structure: VidTimit-Video-m->fadg0...->video->sa1,sa2...->nameVideo.avi)

                - outputs a video dataset(with label) with this structure: DatasetOut-> (Label 0...N) -> video_i.avi...N (example: '1_1_1_11_30_1_m.avi' TO '1_1_1_11_30_5_m.avi').

             
           - preprocessing_meanShift(self, frame)
             
                - takes an input frame (frame_i of N frames in a single video)
                 
                - outputs a preprocessed frame with 
                     - Blurring with 'cv2.GaussianBlur(frame, (5, 5), 0)' 
                     - Histogram equalization  
                     - Clustering method 'cv2.pyrMeanShiftFiltering(equalized, 20, 30)' for color segmentation
                     - Adaptive thresholding 'cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)'
                
                - this method is used in _preprocess_video for preprocessing of N_frames of N_video


           - _preprocess_video(self, input_video, index_video, directoryOutputNormalized, sampling_rate, preprocessing_bool):
              
                - Takes an input Video with his 'index_video', his 'directoryOutputNormalized' (same structure of input dataset with 'label'), 
                  the percentage of images to be processed and a boolean to decide whether or not to preprocess the images. 
                
                - Outputs a percentage of images(preprocessed or not) in the same structure (Dataset->label->frame_0...N) 
                 
                - This method is private and it is used in 'Preprocessing_dataset' and 'Preprocessing_datasetMultiThread'


           - Preprocessing_dataset(self, directory_input_path_video, datasetOutputNormalized, sampling_rate, preprocessing_bool)

                - Takes an input dataset with label and with video for each label (dataset Directory), the output directory where to create the normalized dataset,  
                  the percentage of images to be processed and a boolean to decide whether or not to preprocess the images

                - Output a Normalized dataset with percentage of images(preprocessed or not) for each label in the same structure (Dataset->label->frame_0...N) 

           
           - Preprocessing_datasetMultiThread(self, directory_input_path_video, datasetOutputNormalized, sampling_rate, preprocessing_bool)
        
                
                - same functionality as 'Preprocessing_dataset' but in multi thread, with greater efficiency but with a higher cost


        2) if you want to use the methods of the Preprocessing class correctly, instantiate an object of type 'Preprocessing' in main:

           Example:
             
                if __name__ == '__main__':
               
                   Prep = Preprocessing()
                   
                   Prep.Preprocessing_label('Datset1/Train', 'Dataset1/Training_set')

                   # sampling_rate: 5 = 60 frames, 10 = 30 frames
                   Prep.Preprocessing_datasetMultiThread('Dataset1/Training_Set', 'Dataset1/Training_Set_Normalized', sampling_rate = 5, True)



Istructions for Training:
      
      1) Preliminary settings:
         
          To correctly set the training function, the following variables must be set:

            - timesteps:  Number of input frames for LSTM 
             
            - n_labels:   Number of Dataset_input Labels
            
            - Learning_rate = 0.0001: Learning rate optimizer in this case for adam with very low starting value

            - batch_size = 32 : The number of training examples that are used in a single iteration during the model optimization process

            - validation_ratio = 0.2 : number of validation examples as a percentage of the dataset if you don't want to create a custom validation set
            
            - num_epochs = 50: Number of times the entire training set is presented to the model during the training process

            - img_row :  Number of rows of images to be trained, in case of dimensions greater than 224, the value will be reset to 224

            - img_col :  Number of columns of images to be trained, in case of dimensions greater than 224, the value will be reset to 224

            - img_channel: Number of channels of the images to train, if the images are in color, we enter 3, if the images are in grayscale, we enter 1 channel


  
      2) Functions of file 'TrainLip_CNN_LSTM.py':
       

         - Create_validation_set(validation_path, n_labels)
          
              - Takes validation_set directory and number of labels of validation_set
              
              - Output 2 array: 'x_val' and 'y_val', which respectively contain the images and their labels.



         - Train_dataset(dataset_input):
         
              - Takes as input a dataset with the following structure: Dataset->label->frame_0...N 
               
              - As an output, it saves the model in the directory designated in 'model.save('dir_save')' 
                and prints the model graph (to be saved manually once printed) 
                with information on loss, validation loss, accuracy and validation accuracy of the model



       3) To properly train a model, call the appropriate function in main, 
          inserting as input the directory of the dataset to be used for training:
 
             Example:
             
                if __name__ == '__main__':
                
                   TrainLip_CNN_LSTM.Train_dataset("Dataset1/Training_set_Normalized_60_frames")
        

              


Istructions for Predictions:
   
     1) Methods of 'Predict_lips.py':


        - predict_dataset_lips(Test_Set_input, path_model, img_row, img_col, n_labels):

           - Takes as input the test set on which to make the prediction, the path of the model to be used for the prediction, 
             the dimensions of the images used for the training and the number of labels of the test set.


           - Outputs the accuracy results on the test set in general, the number of correct answers for each class and the accuracy for each class 
               
           - In this method, the prediction is made by taking, for each sample of the test set, 
             the first N frames (which correspond to the number of timesteps passed to the LSTM) 
             and the prediction is based on this sample. 


        - predict_dataset_lips_with_folder(Test_Set_input, path_model, img_row, img_col, n_labels, output_predict_dir)
     
           - Takes as input the test set on which to make the prediction, the path of the model to be used for the prediction, 
             the dimensions of the images used for the training, the number of labels of the test set 
             and the output predict directory which will contain all bad frames in the following directory structure:
                
                outputError_dataset1->CSV
                outputError_dataset1->failure_frames->real_class_i->predicted_class_j->frame_k


           - Outputs the accuracy results on the test set in general, the number of correct answers for each class the accuracy for each class,
             the outputError directory and the CSV file that contains the bad frame, its prediction and its real class.

           
           - In this method, the prediction is made by taking all the frames of each label of the test set 
             and dividing them into N groups of size 'timesteps' (the same with which the model was trained) 
             and checking that the majority of the predictions are correct, if the majority of the predictions are not correct,
             then all the incorrect frames are saved in the list of objects of type 'FrameErrato'
  
     

      2) To correctly perform a prediction, call the appropriate predict functions in main,
         passing as input the path of the test set, the path of the model, 
         the number of rows and columns of the images and the number of labels of the test set:
         
            Example:
             
                if __name__ == '__main__':

                   Predict_lips.predict_dataset_lips_with_folder("Dataset1/Test_set_normalized","Dataset1/Models_normalized_60_frames/EarlyStopModels/Lib_Reading_20Frame_192_Modello1_Normalized_ES44_L2reg.h5", 192, 192, n_labels=256,                    output_predict_dir="outputError_dataset1")
                   

                   Predict_lips.predict_dataset_lips("Dataset1/Test_set_normalized","Dataset1/Models_normalized_60_frames/EarlyStopModels/Lib_Reading_20Frame_192_Modello1_Normalized_ES44_L2reg.h5",192, 192, n_labels=256)  
                   
                   
        