
# Facial recognition in videos

# This requires face_reco_base.py to be run first

import cv2
import matplotlib.pyplot as plt
import imageio
import pylab
import numpy as np
import copy
from utils import load_image
from face_reco_image import FaceImage

CREATE_ANIMATED_GIF = True
CREATE_MP4_VIDEO  = False
DISPLAY_SAMPLE_FRAMES = False
TRAIN_WITH_ALL_SAMPLES = True
USE_FULL_LABELS = True
TEST_FACE_VIDEO = False

class FaceVideo():
    def __init__(self, filename):
        self.face = FaceImage()
        if TRAIN_WITH_ALL_SAMPLES == True:
            self.face.face_recognizer.train_images(train_with_all_samples=True)
        self.temp_file = "temp.jpg"
        self.vid = imageio.get_reader(filename,'ffmpeg')
        self.orig_fps = self.vid.get_meta_data()["fps"]
        

    # Tag video frames with face labels 
    def label_image(self, example_image):
        self.face.face_recognizer.identify_image_faces(example_image)
        img = label_cv2_image_faces(example_image, face_bbs, identities)
        # Convert cv2 RBG back to RGB format
        img = img[:,:,::-1]        
        return img


    def create_animated_gif(self, outputfile):
        # Extract video frames for animated GIF
        frame_interval_secs = 1
        frame_interval_frames = int(frame_interval_secs * self.orig_fps)
        
        num_frames = len(self.vid)
        frames = [i for i in range(0, num_frames, frame_interval_frames)]
        
        video_images = []
        
        for frame in frames:
            image = self.vid.get_data(frame)
            video_images.append(np.array(image))
        
        
        labeled_images = []
        
        for i, video_image in enumerate(video_images):
            print("Processing {} of {} video frames".format(i+1, len(video_images)))
            # TODO: Figure out how to do in-memory transform instead of using temp file
            imageio.imwrite(self.temp_file, video_image)
            #imageio.imwrite("test{}.jpg".format(i+1), video_image)
            video_image2 = load_image(self.temp_file)
            
            if USE_FULL_LABELS:
                img2 = copy.deepcopy(video_image2)
                labeled_image = self.face.detect_face(img2)
            else:
                labeled_image, metadata2, embedded2 = self.label_image(video_image2)
            labeled_images.append(labeled_image)
        
        # Create animated GIF
        playback_frame_duration_secs=1
        
        print("Creating animated GIF...")
        
        with imageio.get_writer(outputfile, mode='I', duration=playback_frame_duration_secs) as writer:
            for image in labeled_images:
                writer.append_data(image)
        
        print("Created animated GIF")


    # Tag video frames with face labels for MP4 video
    def create_mp4_video(self, outputfile):
        vidnew=[]
        for i, image in enumerate (self.vid):
            
            ## label faces in video frame
            print("Processing {} of {} video frames".format(i+1, len(self.vid)))
            # TODO: Figure out how to do in-memory transform instead of using temp file
            imageio.imwrite(self.temp_file, image)
            video_image2 = load_image(self.temp_file)
            
            if USE_FULL_LABELS:
                img2 = copy.deepcopy(video_image2)
                labeled_image = self.face.detect_face(img2)
            else:
                labeled_image = self.label_image(video_image2)
            
            #r = np.random.randint(-10,10,2)
            #n = cv2.rectangle(image,(600+r[0],400+r[1]),(700+r[0],300+r[1]),(0,255,0),3)
            
            ## append facial recognition return image to new list
            vidnew.append(labeled_image)
            
        # Create MP4 video
        writer = imageio.get_writer(outputfile, fps=self.orig_fps)
        
        for im in vidnew:
            writer.append_data(im)
        writer.close()

# Test class

if TEST_FACE_VIDEO:
    labeled_video = FaceVideo("sample/sample02.mp4")
    
    if CREATE_ANIMATED_GIF == True:
        labeled_video.create_animated_gif("sample02gif.gif")
        
    if CREATE_MP4_VIDEO == True:
        labeled_video.create_mp4_video('sample02vid.mp4')
