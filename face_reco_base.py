#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dlib
import cv2
import bz2
import os
from model import create_model

from urllib.request import urlopen

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from utils import load_image

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import time
import copy

import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

# Recognize and label unknown images
from utils import display_cv2_image

import imageio
from sklearn.manifold import TSNE


RECOGNIZE_UNKNOWN_FACES = False
MIN_DLIB_SCORE = 1.1
MIN_SHARPNESS_LEVEL = 30
TEST_FACE_RECOGNIZER = False
MIN_CONFIDENCE_SCORE = 0.3


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 


class FaceRecognizer():
    def __init__(self):
        dst_dir = 'models'
        dst_file = os.path.join(dst_dir, 'landmarks.dat')
        
        if not os.path.exists(dst_file):
            os.makedirs(dst_dir)
            download_landmarks(dst_file)

        # Create CNN model and load pretrained weights (OpenFace nn4.small2)
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')
        self.metadata = self.load_metadata('faces')
        
        # Initialize the OpenFace face alignment utility
        self.alignment = AlignDlib('models/landmarks.dat')

        # Get embedding vectorsf
        # self.embedded = np.zeros((self.metadata.shape[0], 128))
        self.embedded = np.zeros((0, 128))

        # Train images
        custom_metadata = self.load_metadata("faces")
        self.metadata = np.append(self.metadata, custom_metadata)
        self.update_embeddings()
        self.train_images()


    # Download dlib face detection landmarks file
    def download_landmarks(self, dst_file):
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        decompressor = bz2.BZ2Decompressor()
        
        with urlopen(url) as src, open(dst_file, 'wb') as dst:
            data = src.read(1024)
            while len(data) > 0:
                dst.write(decompressor.decompress(data))
                data = src.read(1024)

        
    def load_metadata(self, path):
        ds_store = ".DS_Store"
        metadata = []
        dirs = os.listdir(path)
        if ds_store in dirs:
            dirs.remove(ds_store)
        for i in dirs:
            subdirs = os.listdir(os.path.join(path, i))
            if ds_store in subdirs:
                subdirs.remove(ds_store)
            for f in subdirs:
                metadata.append(IdentityMetadata(path, i, f))
        return np.array(metadata)


    # Align helper functions
    def get_face_thumbnail(self, img):
        return self.alignment.getLargestFaceThumbnail(96, img, self.alignment.getLargestFaceBoundingBox(img), 
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    
    def get_all_face_thumbnails_and_scores(self, img):
        return self.alignment.getAllFaceThumbnailsAndScores(96, img, 
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


    def get_face_vector(self, img, is_thumbnail = False):
        if not is_thumbnail:    
            img = self.get_face_thumbnail(img)
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        return self.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    
    def get_face_vectors(self, img):
        face_thumbnails, scores, face_types = self.get_all_face_thumbnails_and_scores(img)
        face_vectors = []
        for face_img in face_thumbnails:
            # scale RGB values to interval [0,1]
            face_img = (face_img / 255.).astype(np.float32)
            # obtain embedding vector for image
            vector = self.nn4_small2_pretrained.predict(np.expand_dims(face_img, axis=0))[0]
            face_vectors.append(vector)
        return face_vectors, face_thumbnails, scores, face_types
    
    
    # Train classifier models

    def train_images(self, train_with_all_samples = False):
        self.targets = np.array([m.name for m in self.metadata])
        start = time.time()
    
        self.encoder = LabelEncoder()
        self.encoder.fit(self.targets)
    
        # Numerical encoding of identities
        y = self.encoder.transform(self.targets)
    
        if train_with_all_samples == False:
            train_idx = np.arange(self.metadata.shape[0]) % 2 != 0
        else:
            train_idx = np.full(self.metadata.shape[0], True)
            
        self.test_idx = np.arange(self.metadata.shape[0]) % 2 == 0
    
        # 50 train examples of 10 identities (5 examples each)
        X_train = self.embedded[train_idx]
        # 50 test examples of 10 identities (5 examples each)
        X_test = self.embedded[self.test_idx]
    
        y_train = y[train_idx]
        y_test = y[self.test_idx]
    
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        self.svc = LinearSVC() #class_weight='balanced')
    
        self.knn.fit(X_train, y_train)
        self.svc.fit(X_train, y_train)
    
        acc_knn = accuracy_score(y_test, self.knn.predict(X_test))
        acc_svc = accuracy_score(y_test, self.svc.predict(X_test))
    
        if train_with_all_samples == False:
            print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
        else:
            print('Trained classification models with all image samples')
            
        end = time.time()
        print("train_images took {} secs".format(end-start))


    def update_embeddings(self):
        for i, m in enumerate(self.metadata):
            print("loading image from {}".format(m.image_path()))
            img = load_image(m.image_path())
            is_thumbnail = "customer_" in m.image_path()
            vector = self.get_face_vector(img, is_thumbnail)
            vector = vector.reshape(1,128)
            self.embedded = np.append(self.embedded, vector,axis=0)
        
    def label_cv2_image_faces(self, rgb_img, face_bbs, identities):    
        # Convert RGB back to cv2 RBG format
        img = rgb_img[:,:,::-1]
    
        for i, bb in enumerate(face_bbs):
            # Draw bounding rectangle around face
            cv2.rectangle(img, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(img, (bb.left(), bb.bottom() - 35), (bb.right(), bb.bottom()), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, identities[i], (bb.left() + 6, bb.bottom() - 6), font, 1.0, (255, 255, 255), 1)
        return img
    
    def display_cv2_image_faces(self, rgb_img, face_bbs, identities):
        img = label_cv2_image_faces(rgb_img, face_bbs, identities)
        display_cv2_image(img)
        
    def display_plt_image_faces(self, img, face_bbs, identities, subplot=111):
        plt.subplot(subplot)
        plt.figure()
        plt.imshow(img)
        for bb in face_bbs:
            plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
        # TODO: Print identities in correct order
        plt.title(f'Recognized as {identities}')


    def save_unknown_face(self, face_vector, face_thumbnail):
        print("Saving unknown face...")
        dirs = os.listdir("faces")
        customer_dirs = [dir for dir in dirs if "customer_" in dir]
        if len(customer_dirs) > 0:
            dir_indexes = [int(dir.split("_")[1]) for dir in customer_dirs]
            curr_index = max(dir_indexes) + 1
        else:
            curr_index = 1
                    
        # Save image to customer dir
        # TODO: Remove requirement for double-creation of all data
        customer_dir = "customer_{}".format(curr_index)
        dir_path = os.path.join("faces", customer_dir)
        os.mkdir(dir_path)
        for i in range (0,8):
            customer_file = "customer_{}_{}.jpg".format(curr_index, i+1)
            file_path = os.path.join(dir_path, customer_file)
            imageio.imwrite(file_path, face_thumbnail)
            metadata = np.append(self.metadata, IdentityMetadata("custom", customer_dir, customer_file))
            embedded = np.append(self.embedded, face_vector.reshape(1,128), axis=0)
    
        print("Saved unknown face")    

    def distance(self, emb1, emb2):
        return np.sum(np.square(emb1 - emb2))
    
    def get_distances(self, vector):
        distances = []
        for embed in self.embedded:
            distances.append(distance(embed,vector))
        return distances
    
    def get_sharpness_level(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(grey, cv2.CV_64F).var()
    
    
    def identify_image_faces(self, example_image):
        vectors, thumbnails, dlib_scores, face_types = self.get_face_vectors(example_image)
        
        identities = []
        saved_unknown = False
        for i, vector in enumerate(vectors):
            vector = vector.reshape(1,128)
            confidence_scores = self.svc.decision_function(vector)    
            if (confidence_scores.max() < MIN_CONFIDENCE_SCORE):
                sharpness_level = self.get_sharpness_level(thumbnails[i])
                example_identity = "Unknown"
                #example_identity = "Unknown ({:0.2f}, {}, {:0.2f})".format(dlib_scores[i], face_types[i], sharpness_level)
                print("Unknown face - dlib score={:0.2f}, face_type={}, sharpness_level={:0.2f}".format(dlib_scores[i], face_types[i], sharpness_level))
                if RECOGNIZE_UNKNOWN_FACES:
                    # Only save (and train) a good-quality and front-facing face
                    if dlib_scores[i] >= MIN_DLIB_SCORE and face_types[i] == 0 and sharpness_level >= MIN_SHARPNESS_LEVEL:
                        saved_unknown = True
                        print("Saving unknown face")
                        save_unknown_face(vector, thumbnails[i])
            else:
                example_prediction = self.svc.predict(vector)
                example_identity = self.encoder.inverse_transform(example_prediction)[0]
            identities.append(example_identity)
            
        # Add to training model if any unknown faces were saved
        if saved_unknown:
            train_images()
            
        # Detect faces and return bounding boxes
        face_bbs = self.alignment.getAllFaceBoundingBoxes(example_image)
        
        return face_bbs, identities
        
    def display_unknown_image(self, image_path):
        img = load_image(image_path)
     
        face_bbs, identities = self.identify_image_faces(img)
        #display_cv2_image_faces(img, face_bbs, identities)
        self.display_plt_image_faces(img, face_bbs, identities)
    
    def display_image_prediction(self, example_idx): 
        example_image = load_image(self.metadata[self.test_idx][example_idx].image_path())
        example_prediction = self.knn.predict([self.embedded[self.test_idx][example_idx]])
        example_identity = self.encoder.inverse_transform(example_prediction)[0]
    
        # Detect face and return bounding box
        #bb = alignment.getLargestFaceBoundingBox(example_image)
    
        plt.imshow(example_image)
        #plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
        plt.title(f'Recognized as {example_identity}')
    
    def visualize_dataset(self):
        X_embedded = TSNE(n_components=2).fit_transform(self.embedded)
        plt.figure()
        
        for i, t in enumerate(set(self.targets)):
            idx = self.targets == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)   
        
        plt.legend(bbox_to_anchor=(1, 1));


# Create class and test it
if TEST_FACE_RECOGNIZER:
    face_recognizer = FaceRecognizer()
    
    example_idx =6
    
    face_recognizer.display_image_prediction(example_idx)
    
    face_recognizer.display_unknown_image("sample/sample01.jpg")
    
    face_recognizer.visualize_dataset()




