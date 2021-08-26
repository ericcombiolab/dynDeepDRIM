from __future__ import print_function
# Usage  python train_with_labels_three_fold.py number_of_data_parts_divided NEPDF_pathway number_of_category
# command line in developer's linux machine :
# module load cuda-8.0 using GPU
#srun -p gpu --gres=gpu:1 -c 2 --mem=20Gb python train_with_labels_three_foldx.py 9 /home/yey3/cnn_project/code3/NEPDF_data 3 > results.txt
#######################OUTPUT
# it will generate three-fold cross validation results

#CUDA_VISIBLE_DEVICES='0' python gpu_print.py

import argparse

import sys
import os

parser = argparse.ArgumentParser(description="example")

from numba import jit, cuda
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
from scipy import interp
import pandas as pd



parser = argparse.ArgumentParser(description="")

parser.add_argument('-num_batches', type=int, required=True, default=None, help="Number of TF or the number of x file.")
parser.add_argument('-data_path', required=True, default=None, help="The path that includes x file, y file and z file.")
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
parser.add_argument('-cross_validation_fold_divide_file', default=None, help="A file that indicate how to divide the x file into three-fold. The file include three line, each line list the ID of the x files for the folder (split by ',')")

args = parser.parse_args()


class direct_model1_squarematrix:
    def __init__(self, num_batches=5, output_dir=None, data_path=None, predict_output_dir=None):
        # ###################################### parameter settings
        self.data_augmentation = False
        # num_predictions = 20
        self.batch_size = 32  # mini batch for training
        # num_classes = 3   # ### categories of labels
        self.epochs = 200  # ### iterations of trainning, with GPU 1080, 200 for KEGG and Reactome, depends on specific tasks for GTRD, we actually selected
        # the best epochs and learning rate by a test on the first three TF in list
        # length_TF =3057  # number of divide data parts
        # num_predictions = 20
        self.model_name = 'keras_cnn_trained_model_DeepDRIM.h5'
        self.output_dir = output_dir
        if output_dir is not None:
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
        #####
        self.num_batches = num_batches  # number of data parts divided
        self.data_path = data_path
        self.num_classes = 2
        self.whole_data_TF = [i for i in range(self.num_batches)]
        ###################################################
        self.x_train = None
        self.y_train = None
        self.z_train = None
        self.count_set_train = None
        self.x_test = None
        self.y_test = None
        self.z_test = None
        self.count_set = None
        self.load_model_path = None
        self.predict_output_dir = predict_output_dir


    def load_data_TF2(self,indel_list,data_path, num_of_pair_ratio=1):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
        import random
        import numpy as np
        xxdata_list = []
        yydata = []
        zzdata = []
        count_set = [0]
        count_setx = 0
        for i in indel_list:#len(h_tf_sc)):
            xdata = np.load(data_path+str(i)+'_xdata.npy')
            ydata = np.load(data_path+str(i)+'_ydata.npy')
            zdata = np.load(data_path+str(i)+'_zdata.npy')

            num_of_pairs = round(num_of_pair_ratio*len(ydata))
            all_k_list = list(range(len(ydata)))
            select_k_list = all_k_list[0:num_of_pairs]

            count=0
            for k in select_k_list:
                y_data = ydata[k]
            #for TF_predict
                if ydata[k] != 2:                     
                    xxdata_list.append(xdata[k,:,:,:,:]) # time course
                    yydata.append(y_data)
                    zzdata.append(zdata[k])
                    count+=1                  
            count_setx = count_setx + count
            count_set.append(count_setx)  # [start_batch/tf0,start_batch/tf1....start_batch/tfn,end_batch/tfn +1]
            print(i,'\t',count)
                  
        yydata_array = np.array(yydata)
        yydata_x = yydata_array.astype('int')
        print(np.array(xxdata_list).shape)
        return((np.array(xxdata_list),yydata_x,count_set,np.array(zzdata)))

    def update_test_train_data(self, test_indel,epochs,num_of_pair_ratio=1):
        print("len test_indel",test_indel)
        if type(test_indel)!=list:
            test_TF = [test_indel]  #
        else:
            test_TF = test_indel
        train_TF = [i for i in self.whole_data_TF if i not in test_TF]  #
        #####################################################################
        (self.x_train, self.y_train, self.count_set_train,self.z_train) = self.load_data_TF2(train_TF, self.data_path,num_of_pair_ratio)
        (self.x_test, self.y_test, self.count_set,self.z_test) = self.load_data_TF2(test_TF, self.data_path,num_of_pair_ratio)
        print(self.x_train.shape, 'x_train samples')
        print(self.x_test.shape, 'x_test samples')
        if self.output_dir is not None:
            self.save_dir = os.path.join(self.output_dir, str(test_indel) + '_saved_models' + str(epochs))  ## the result folder
        else:
            self.save_dir="."
        if self.num_classes > 2:
            self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
            self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        print(self.y_train.shape, 'y_train samples')
        print(self.y_test.shape, 'y_test samples')
        ############
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def get_single_image_model(self, x_train):
        ############
        print("x_train.shape in single image",x_train.shape)

        input_img = keras.layers.Input(shape=x_train.shape[1:])
        x=keras.layers.Conv2D(32, (3, 3), padding='same',activation='relu')(input_img)
       
        x=keras.layers.Conv2D(32, (3, 3),activation='relu')(x)
    
        x=keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x=keras.layers.Dropout(0.25)(x)

        x=keras.layers.Flatten()(x)
       
        model_out=keras.layers.Dense(512)(x)

        return keras.Model(input_img,model_out)

    def get_pair_image_model(self,x_train):
        ############
        print("x_train.shape in multi image", x_train.shape)

        input_img = keras.layers.Input(shape=x_train.shape[1:])
        x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)    
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)

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

        combined = keras.layers.Dense(512)(combined_layer)
        
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
        
        if self.num_classes < 2:
            print('no enough categories')
            sys.exit()
        elif self.num_classes == 2:
            
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            #sgd = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
        checkpoint1 = ModelCheckpoint(filepath=self.save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                      monitor='val_loss',
                                      verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        checkpoint2 = ModelCheckpoint(filepath=self.save_dir + '/weights.hdf5', monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='auto', period=1)
        callbacks_list = [checkpoint2, early_stopping]
        self.model = model
        self.callbacks_list = callbacks_list

    def test_model(self,model,x_test,y_test,z_test,save_dir,history,test_indel):
        # Score trained model.
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        y_predict = model.predict(x_test)
        np.save(save_dir + '/end_y_test.npy', y_test)
        np.savetxt(save_dir + '/end_y_test.csv', y_test, delimiter=",")
        np.save(save_dir + '/end_y_predict.npy', y_predict)
        np.savetxt(save_dir + '/end_y_predict.csv', y_predict, delimiter=",")
        print(z_test)
        df = pd.DataFrame(z_test)
        df.to_csv(save_dir + '/end_z_test.csv')
        ############################################################################## plot training process
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.grid()
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid()
        plt.savefig(save_dir + '/end_result.pdf')
        ###############################################################  evaluation without consideration of data separation
        if self.num_classes == 2:  ## here we only focus on three category tasks
            plt.figure(figsize=(10, 6))
            #for i in range(3):
                #y_test_x = [j[i] for j in y_test]
                #y_predict_x = [j[i] for j in y_predict]
            y_test_x = [j for j in y_test]
            y_predict_x = [j for j in y_predict]

            fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
                #plt.subplot(1, 3, i + 1)
            plt.plot(fpr, tpr)
            plt.grid()
            plt.plot([0, 1], [0, 1])
            plt.xlabel('FP')
            plt.ylabel('TP')
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            auc = np.trapz(tpr, fpr)
            print('AUC:', auc)
                #plt.title('label' + str(i) + ', AUC:' + str(auc))
            plt.savefig(save_dir + '/end_2labels.pdf')
            
            y_testy = y_test
            y_predicty = y_predict
            fig = plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1])
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.xlabel('FP')
            plt.ylabel('TP')
            # plt.grid()
            AUC_set = []
            s = open(save_dir + '/divided_interaction.txt', 'w')
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)  # 3068
            for jj in range(len(self.count_set) - 1):  # len(count_set)-1):
                if self.count_set[jj] < self.count_set[jj + 1]:
                    print(test_indel, jj, self.count_set[jj], self.count_set[jj + 1])
                    y_test = y_testy[self.count_set[jj]:self.count_set[jj + 1]]
                    y_predict = y_predicty[self.count_set[jj]:self.count_set[jj + 1]]
                    # Score trained model.
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    # Print ROC curve
                    plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
                    auc = np.trapz(tpr, fpr)
                    s.write(str(jj) + '\t' + str(self.count_set[jj]) + '\t' + str(self.count_set[jj + 1]) + '\t' + str(auc) + '\n')
                    print('AUC:', auc)
                    AUC_set.append(auc)

            mean_tpr = np.median(tprs, axis=0)
            mean_tpr[-1] = 1.0
            per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
            mean_auc = np.trapz(mean_tpr, mean_fpr)
            plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
            plt.title(str(mean_auc))
            plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
            plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
            plt.legend(loc='lower right')
            plt.savefig(save_dir + '/divided_interaction_percentile.pdf')
            del fig
            fig = plt.figure(figsize=(5, 5))
            plt.hist(AUC_set, bins=50)
            plt.savefig(save_dir + '/divided_interaction_hist.pdf')
            del fig
            s.close()


    def train_and_test_model(self,num_of_pair_ratio=1):
        for test_indel in range (self.num_batches):                                      #  for GTRD leave-one-TF-out CV
            self.update_test_train_data(test_indel,self.epochs,num_of_pair_ratio)
            self.construct_model(self.x_train)
            model = self.model
            callbacks_list = self.callbacks_list
            x_train = self.x_train
            y_train = self.y_train
            x_test = self.x_test
            y_test = self.y_test
            z_test = self.z_test
            history = None
            if not self.data_augmentation:
                print('Not using data augmentation.')
                n = x_train.shape[1]
                x_train_list = []
                for j in range(0, n):
                    x_train_list.append(x_train[:, j, :, :, np.newaxis])
                history = model.fit(x_train_list, y_train,batch_size=self.batch_size,epochs=self.epochs,validation_split=0.2,
                          shuffle=True, callbacks=callbacks_list)
            # Save model and weights
            model_path = os.path.join(self.save_dir, self.model_name)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)
            n2 = x_test.shape[1]
            x_test_list = []
            for j in range(0,n2):
                x_test_list.append(x_test[:,j,:,:,np.newaxis])
            self.test_model(model,x_test_list,y_test,z_test,self.save_dir,history,test_indel)


    def train_and_test_model_dividePart_assignTForder(self,indel_list0,indel_list1,indel_list2,num_of_pair_ratio=1):

        indel_list0 = [int(i) for i in indel_list0]
        indel_list1 = [int(i) for i in indel_list1]
        indel_list2 = [int(i) for i in indel_list2]
        
        divide_part = 3#

        for i in range(0, divide_part):
            if i==0:
                test_indel=indel_list0
            elif i==1:
                test_indel=indel_list1
            elif i==2:
                test_indel=indel_list2

            self.update_test_train_data(test_indel,self.epochs,num_of_pair_ratio)
            self.construct_model(self.x_train)
            model = self.model
            callbacks_list = self.callbacks_list
            x_train = self.x_train
            y_train = self.y_train
            x_test = self.x_test
            y_test = self.y_test
            z_test = self.z_test
            history = None
            n = x_train.shape[1]
            if self.load_model_path is not None:
                model.load_weights(self.load_model_path)
            if not self.data_augmentation:
                print('Not using data augmentation.')

                x_train_list=[]

                for i in range(x_train.shape[2]):
                    x_train_singletime_list = []
                    for j in range(0,n):    
                        x_train_singletime_list.append(x_train[:,j,i,:,:,np.newaxis]) 
                    x_train_list.append(x_train_singletime_list)
                
                history = model.fit(x_train_list, y_train,batch_size=self.batch_size,epochs=self.epochs,validation_split=0.2,
                          shuffle=True, callbacks=callbacks_list)
            # Save model and weights
            model_path = os.path.join(self.save_dir, self.model_name)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)
            x_test_list=[]

            for i in range(x_test.shape[2]):
                    x_test_singletime_list = []
                    for j in range(0,n):    
                        x_test_singletime_list.append(x_test[:,j,i,:,:,np.newaxis])
                    x_test_list.append(x_test_singletime_list)
                
            self.test_model(model,x_test_list,y_test,z_test,self.save_dir,history,test_indel)

def load_indel_lists_from_file(cross_validation_fold_divide_file):
    s = open(cross_validation_fold_divide_file)
    cross_fold = []

    for line in s:
        line=line.strip()
        separation = line.split(',')
        indel_list = []
        for i in range(0, len(separation)):
            indel_list.append(separation[i])
            #self.whole_data_TF.append(separation[i])
        cross_fold.append(indel_list)

    indel_list0=cross_fold[0]
    indel_list1=cross_fold[1]
    indel_list2=cross_fold[2]
    print('indel_list0',indel_list0)
    print('indel_list1',indel_list1)
    print('indel_list2',indel_list2)
    return indel_list0,indel_list1,indel_list2



def main():

    tcs = direct_model1_squarematrix(num_batches=args.num_batches,
        data_path=args.data_path,
        output_dir=args.output_dir)
    indel_list0,indel_list1,indel_list2=load_indel_lists_from_file(args.cross_validation_fold_divide_file)

    tcs.train_and_test_model_dividePart_assignTForder(indel_list0,indel_list1,indel_list2)



if __name__ == '__main__':

    if args.cross_validation_fold_divide_file is not None:
        main()
    else:
        print("Require input cross_validation_fold_divide_file")


