import numpy as np
import os
import cv2 as cv2
import pandas as pd
import xml.etree.ElementTree as ET
from keras.src.utils import to_categorical
from numpy import matlib
from sklearn.preprocessing import LabelEncoder
import Obj_Seg
from AOOA import AOOA
from BOA import BOA
from FOA import FOA
from GMEO import GMEO
from Global_Vars import Global_vars
from Mask_RCNN import Mask_RCNN
from Model_A_MGA_GCY9_GELAN import Model_A_MGA_GCY9_GELAN
from Model_DMSAU_Net import Model_DMSAU_Net
from Model_SA_YOLOV8 import Model_SA_YOLOV8
from PROPOSED import PROPOSED
from Plot_results import *

no_of_dataset = 2

# Read the dataset 1
an = 0
if an == 1:
    Images = []
    Targets = []
    Annotated = []
    Bounds = []
    croped = []
    Directory = './Datasets/Dataset_1/'
    Images_Dir = Directory + 'JPEGImages/'
    List_Images = os.listdir(Images_Dir)
    Annote_Dir = Directory + 'Annotations/'
    classes_names = pd.read_csv(Directory + 'Classes.txt.txt', sep='\t')
    for i in range(len(List_Images)):
        filename = List_Images[i]
        image_file = Images_Dir + List_Images[i]
        xml_file = Annote_Dir + filename.split('.')[0] + '.xml'
        if os.path.exists(xml_file):
            print(i + 1, len(List_Images))

            image = cv.imread(image_file)
            image = np.uint8(image)
            Image = cv.resize(image, (128, 128))

            tree = ET.parse(xml_file)
            root = tree.getroot()
            sub_root = list(root)
            object_list = list(sub_root[len(sub_root) - 1])
            bou_box_list = list(object_list[len(object_list) - 1])
            Bound = []
            for child in bou_box_list:
                Bound.append(int(child.text))
            top_left_x = Bound[0]
            top_left_y = Bound[1]
            bot_right_x = Bound[2]
            bot_right_y = Bound[3]
            cropped_img = image[top_left_y:bot_right_y + 1, top_left_x:bot_right_x + 1]
            crop_img = cv.resize(cropped_img, (128, 128))
            bbox = image.copy()
            cv.rectangle(bbox, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (255, 0, 0), 2)
            bbox = cv.resize(bbox, (128, 128))
            Targets.append((filename[2:5]))
            Bounds.append(Bound)
            Annotated.append(bbox)
            croped.append(crop_img)
            Images.append(Image)
    Targets = np.asarray(Targets)
    from keras.src.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    targ = encoder.fit_transform(Targets)
    uni, unicount = np.unique(Targets, return_counts=True)
    labels = to_categorical(targ)
    classes_names = np.asarray(classes_names)
    class_no = []
    class_names = []
    for m in range(classes_names.shape[0]):
        splitting = classes_names[m, 0].split('    ')
        class_no.append(splitting[0])
        class_names.append(splitting[1])

    class_no = np.asarray(class_no).astype('str')
    missing_values = class_no[~np.isin(class_no, uni)]  # 031-wheat blossom midge, 060-Viteus vitifoliae,

    np.save('Images_1.npy', Images) # Save Image 1
    np.save('Bounds_1.npy', Bounds) # Save Bounds 1
    np.save('Target_1.npy', labels) # Save Target 1

# Read the dataset 2
an = 0
if an == 1:
    Images = []
    Targets = []
    Annotated = []
    Bounds = []
    croped = []
    Directory = './Datasets/Dataset_2/'
    Dataset_dir = os.listdir(Directory)
    count = 0
    for n in range(len(Dataset_dir)):
        if Dataset_dir[n] == 'test' or Dataset_dir[n] == 'train' or Dataset_dir[n] == 'valid':
            train_test_dir = Directory + Dataset_dir[n]

            annotate_text = train_test_dir + '/_annotations.txt'
            annotate = pd.read_csv(annotate_text, sep='\t', header=None)

            Class_text = train_test_dir + '/_classes.txt'
            classes = pd.read_csv(Class_text, sep='\t', header=None)

            annotate = np.asarray(annotate)
            for i in range(len(annotate)):
                print(n, len(Dataset_dir), i, len(annotate), count)
                splitting = annotate[i][0].split(' ')
                if len(splitting) > 1:
                    tar = splitting[1].split(',')[-1]
                    Bound = splitting[1].split(',')[:-1]

                    image_path = train_test_dir + '/' + splitting[0]
                    image = cv.imread(image_path)
                    img = cv.resize(image, (128, 128))
                    img = np.uint8(img)

                    top_left_x = int(Bound[0])
                    top_left_y = int(Bound[1])
                    bot_right_x = int(Bound[2])
                    bot_right_y = int(Bound[3])
                    cropped_img = image[top_left_y:bot_right_y, top_left_x:bot_right_x]
                    crop_img = cv.resize(cropped_img, (128, 128))
                    bbox = image.copy()
                    cv.rectangle(bbox, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (255, 0, 0), 2)
                    bbox = cv.resize(bbox, (128, 128))

                    count += 1
                    Images.append(img)
                    Targets.append(tar)
                    Bounds.append(Bound)
                    Annotated.append(bbox)
                    croped.append(crop_img)

    Targets = np.asarray(Targets)
    encoder = LabelEncoder()
    targ = encoder.fit_transform(Targets)
    uni, unicount = np.unique(Targets, return_counts=True)
    labels = to_categorical(targ)

    np.save('Images_2.npy', Images) # Save Images 2
    np.save('Bounds_2.npy', Bounds) # Save Bounds 2
    np.save('Target_2.npy', labels) # Save Target 2


# Optimization for Detection
an = 0
if an == 1:
    BESTSOL = []
    FITNESS = []
    for n in range(no_of_dataset):
        Feat = np.load('Images_' + str(n+1) + '.npy', allow_pickle=True) # Load Images
        Target = np.load('Target_' +  str(n+1) + '.npy', allow_pickle=True) # Load Target
        Global_vars.Feat = Feat
        Global_vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden neuron count in YoloV9 , No of Epochs in YoloV9 , Activation Function in YoloV9
        xmin = matlib.repmat(np.asarray([5, 5, 1]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 50, 5]), Npop, 1)
        fname = Obj_Seg
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("AOOA...")
        [bestfit1, fitness1, bestsol1, time1] = AOOA(initsol, fname, xmin, xmax, Max_iter)

        print("FOA...")
        [bestfit2, fitness2, bestsol2, time2] = FOA(initsol, fname, xmin, xmax, Max_iter)

        print("BOA...")
        [bestfit3, fitness3, bestsol3, time3] = BOA(initsol, fname, xmin, xmax, Max_iter)

        print("GMEO...")
        [bestfit4, fitness4, bestsol4, time4] = GMEO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        FITNESS.append(fitness)
        BESTSOL.append(BestSol_CLS)

    np.save('Fitness.npy', np.asarray(FITNESS)) # Save Fitness
    np.save('BestSol.npy', np.asarray(BESTSOL)) # Save Bestsol

# Detection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n+1) + '.npy', allow_pickle=True) # Load Images
        Target = np.load('Target_' +  str(n+1) + '.npy', allow_pickle=True) # Load Target
        BestSol = np.load('BestSol.npy', allow_pickle=True) # Load Bestsol
        for j in range(BestSol.shape[0]):
            print(j)
            sol = BestSol[j, :]
            Eval, pred0 = Model_A_MGA_GCY9_GELAN(Images, Target, sol=sol)
        Eval_1, Method_1 =  Mask_RCNN(Images, Target)
        Eval_2, Method_2 = Model_SA_YOLOV8(Images, Target)
        Eval_3, Method_3 = Model_DMSAU_Net(Images, Target)
        Eval_4, Method_4 = Model_A_MGA_GCY9_GELAN(Images, Target)
        Eval_5, Proposed = Model_A_MGA_GCY9_GELAN(Images, Target, sol=BestSol[-1, :])
        Seg = [Method_1, Method_2, Method_3, Method_4, Proposed]
        np.save('Detected' + str(n+1) + '.npy', Seg)
        np.save('Det_img' +  str(n+1) + '.npy',Proposed)

plot_conv() # Plot Convergence Graph
plot_Proposed() # plot Proposed Graph
plot_BS() # plot Batchsize Variation Graph
plot_Epoch() # plot Epoch Variation Table
Image_Results() # plot Image Results
Sample_images() # plot Sample Images

