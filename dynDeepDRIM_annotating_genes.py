from __future__ import print_function

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="example")

#from numba import jit, cuda
import tensorflow.keras as keras
import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


parser = argparse.ArgumentParser(description="")

parser.add_argument('-test_data_path', required=True, default=None)
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
parser.add_argument('-annotation_name', type=str,required=True, default=None)
parser.add_argument('-trained_model_path', type=str,required=True, default=None)

args = parser.parse_args()


class dynDeepDRIM_annotate_gene_function:
    def __init__(self, output_dir=None,test_data_path=None, predict_output_dir=None, load_model_path =None):
        # ###################################### parameter settings
        self.data_augmentation = False   
 
        self.model_name = 'keras_cnn_trained_model_DeepDRIM.h5'
        self.output_dir = output_dir
        if output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
    
        self.test_data_path = test_data_path

        self.num_classes = 2

        self.x_test = None
        self.y_test = None
        self.z_test = None
        
        self.load_model_path = os.path.join(load_model_path,self.model_name)
        self.predict_output_dir = predict_output_dir
        

    def load_data_TF2(self,data_path):  
        import random
        import numpy as np
        xxdata_list = []
        yydata = []
        xdata = np.load(data_path)
     
 
        for k in range(int(xdata.shape[0]/2)):

            xxdata_list.append(xdata[2*k,:,:,:,:]) 
            xxdata_list.append(xdata[2*k+1,:,:,:,:])  
                
            yydata.append(1)
            yydata.append(0)
                
        yydata_array = np.array(yydata)
        yydata_x = yydata_array.astype('int')
        
        return((np.array(xxdata_list),yydata_x))



    def update_test_data(self,save_name_folder):
        
        (self.x_test, self.y_test) = self.load_data_TF2(self.test_data_path)

        print(self.x_test.shape, 'x_test tensors')

        if self.output_dir is not None:
            self.save_dir = os.path.join(self.output_dir, save_name_folder) 
        else:
            self.save_dir="."

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        

    def get_single_image_model(self, x_train):
   
        input_img = keras.layers.Input(shape=x_train.shape[1:])
        x=keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu')(input_img)
           
        x=keras.layers.Conv2D(16, (3, 3),activation='relu')(x)
        
        x=keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x=keras.layers.Dropout(0.5)(x)
       
        x=keras.layers.Flatten()(x)
      
        model_out=keras.layers.Dense(512)(x)
          
        return keras.Model(input_img,model_out)


    def get_pair_image_model(self,x_train):
        input_img = keras.layers.Input(shape=x_train.shape[1:])
        x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
       
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
       
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Flatten()(x)

        model_out = keras.layers.Dense(512)(x)

        return keras.Model(input_img,model_out)

        
    def DeepDRIM_sub(self,x_train):
        ############
        # for concatenate
        print("x shape", x_train.shape)

        n=x_train.shape[1]
        x1=x_train[:, 0, :, :,np.newaxis]

        x2=x_train[:, 1:n, :, :,np.newaxis]
        x2_1=x2[:,0,:,:,:]


        single_image_model=self.get_single_image_model(x1)
        input_img_single = keras.layers.Input(shape=x1.shape[1:])
        single_image_out = single_image_model(input_img_single)

        pair_image_model = self.get_pair_image_model(x2_1)

        input_img = keras.layers.Input(shape=x2.shape[1:])
        pair_image_out_list=[]
        input_img_whole_list=[]
        input_img_whole_list.append(input_img_single)
        input_img_multi_list=[]
        for i in range(0,n-1):
            input_img_multi = keras.layers.Input(shape=x2_1.shape[1:])
            input_img_multi_list.append(input_img_multi)
            input_img_whole_list.append(input_img_multi)
            pair_image_out=pair_image_model(input_img_multi)

            pair_image_out_list.append(pair_image_out)
        merged_vector=keras.layers.concatenate(pair_image_out_list[:], axis=-1)#modify this sentence to merge
        merged_model=keras.Model(input_img_multi_list,merged_vector)
        merged_out=merged_model(input_img_multi_list)
        combined_layer = keras.layers.concatenate([single_image_out, merged_out], axis=-1)
        combined_layer = keras.layers.Dropout(0.5)(combined_layer)

        combined = keras.layers.Dense(512, activation='relu')(combined_layer)
        
        return keras.Model(input_img_whole_list, combined)
        
    def DeepDRIM_cat(self,x_train):
        x_first = x_train[:,:,0,:,:]
        x_drim = x_train[:,:,0,:,:,np.newaxis]
        
        model_list = []
        deepDrim_input_list=[]
        deepDrim_output_list=[]
        for i in range(x_train.shape[2]):
            
            single_DeepDRIM = self.DeepDRIM_sub(x_first) # model
   
            deepDrim_in =[] # deepdrim input
            for j in range(x_train.shape[1]): 
                deepDrim_in.append(keras.layers.Input(shape=x_drim.shape[2:])) 
            
            deepDrim_out=single_DeepDRIM(deepDrim_in) # deepdrim output
            
            model_list.append(single_DeepDRIM)    
            deepDrim_input_list.append(deepDrim_in)
            deepDrim_output_list.append(deepDrim_out)
            
        cat_deepDrim_output = tf.convert_to_tensor(deepDrim_output_list)
        cat_deepDrim_output = tf.transpose(cat_deepDrim_output, (1, 0, 2))        
        cat_deepDrim_output = keras.layers.Flatten()(cat_deepDrim_output)
           
        combined = keras.layers.Dense(512, activation='relu')(cat_deepDrim_output)
        combined = keras.layers.Dropout(0.5)(combined)
        combined = keras.layers.Dense(128, activation='relu')(combined)
        combined = keras.layers.Dropout(0.5)(combined)   
        combined = keras.layers.Dense(1, activation='sigmoid')(combined)
       
        return keras.Model(deepDrim_input_list, combined)
        

    def construct_model(self, x_train):       
        model = self.DeepDRIM_cat(x_train)       
        self.model = model



    def test_model_predict(self, annotation_name):

        self.update_test_data(annotation_name)     
        self.construct_model(self.x_test)
       
        x_test = self.x_test
        y_test = self.y_test
          
        n = x_test.shape[1]
        if self.load_model_path is not None:
            self.model.load_weights(self.load_model_path)   
 
        x_test_list=[]

        for i in range(x_test.shape[2]):
                x_test_singletime_list = []
                for j in range(0,n):    
                    x_test_singletime_list.append(x_test[:,j,i,:,:,np.newaxis]) 
                x_test_list.append(x_test_singletime_list)
                
        score = self.model.predict(x_test_list)
        np.savetxt(os.path.join(self.save_dir,'y_predict.csv') , score, delimiter=",")



def main():

    ins = dynDeepDRIM_annotate_gene_function(test_data_path=args.test_data_path,output_dir=args.output_dir,load_model_path=args.trained_model_path)
    ins.test_model_predict(annotation_name= args.annotation_name)


if __name__ == '__main__':
    
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    
    main()
