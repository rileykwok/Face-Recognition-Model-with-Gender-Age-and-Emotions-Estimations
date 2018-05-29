# Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations
Capstone Project by [Bertrand Lee](https://github.com/bertrandlee) and [Riley Kwok](https://github.com/rileykwok)

## Table of Contents
+ [About](#about)
+ [Technical System Architecture](#technical-system-architecture)
+ [Results](#results)
+ [References](#references)
+ [Installations](#installations)

## About

We created a prototype face recognition system based on three pre-trained CNN models that is able to identify faces and predict their gender, age and emotions in an image or video. 

The prototype is designed to be implemented in the retail industry for different applications:
- Personalized Customer Experience

   By linking facial recognition system to a loyalty program, customer information and purchasing history could be displayed on the employee's smart devices. Based on the information, the salesperson could provide a premium, personalised experience to the customer, e.g. greet the customer by name and based on his previous purchases, recommend his favourite products or new products suitable for his taste. 

- Collect Customer Demographics

   For unregistered customers, i.e. face not registered to the system, the system is able to track their unique face vectors and gather useful demographic information including gender and age estimations. Their purchase records of these unregistered customers can also be tracked by cameras at the checkout counter.

   With this demographic data, the company would be able to improve their customer segmentation and generate consumer-centered marketing strategies and more targeted in-store events.

- Enhanced Store Traffic Analytics

   Face recognition system can also collect data about the number of shoppers that visit the store and their foot traffic patterns. It can also track the conversion rate, or in other words, the percentage of shoppers that end up making a purchase. Furthermore, the system can identify hot spot areas where shoppers linger, or where there is a lot of foot traffic. 

   The benefits that this bring include being able to analyse the foot traffic patterns to create more targeted and effective visual merchandising and product assortment. identify which in-store marketing promotions are most popular, and help to create more accurate per-store revenue forecasts.

- Tracking Customer Emotions

   A face recognition system can also provide valuable feedback to in-store promotions via tracking customers’ emotional response, enabling the retailer to improve product assortment, visual merchandising, and tailor more effective promotion campaigns.

- Improve Store Security

   Face recognition system will be very helpful in preventing in-store crimes such as shop liftings. The store’s security team could be alerted instantly if a person previously flagged for shoplifting enters the store, so that person can be monitored more closely by store security.

   The significant savings resulting from reduced in-store theft can then be passed on to customers in the form of lower product prices.

## Technical System Architecture
| Stages| Face Detection | Face Alignment  |Face Feature <br>Extraction|Face Classification|
| --- |-------------| -----|-----|-----|
|**ML Models**|||**OpenFace.nn4** <br>*(Identity)*<br>**WideResNet**<br>*(Age, Gender)*<br>**mini_XCEPTION**<br>*(Emotions)* | SVM Classifier |
| **Libraries**| Dlib|OpenCV|Keras<br>TensorFlow |Scikit Learn |
|**Language**| Python|Python|Python|Python|

## Results
[Left] System detected new customer face at promotion booth and shows the estimated gender, age and emotion.
<br>[Right] Video shows how a known customer is being identified and how his emotion is being tracked. 

<img src="https://github.com/rileykwok/Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations/blob/master/sample/promotion1.png" width="300" height="300">       <img src="https://github.com/rileykwok/Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations/blob/master/sample/supermarket.gif" width="300">


We have trained the system on some celebrities and some of our friends, we noted that in some cases that the 2 face vectors of 2 different people were even located closer then each other (euclidean distance) than two photos of one person. From what we tested, using 5+ photos per person with clean, clear, front facing faces would produce better results for identity estimates. 

The SVM classifier is also prefered over KNN classifier as they produce slightly better estimation. It also provides a confidence score per estimate which allows us to set a threshold to categorize known or new faces. The confidence score threshold we experimented that works best is 0.3.

<img src="https://github.com/rileykwok/Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations/blob/master/sample/visualisation-t-SNE.png" width="500">

## References
Face Detection: [DLib](http://dlib.net/)
<br>Face Alignment: [OpenCV:](https://opencv.org/)
<br>Face Recognition Models: 
<br>[OpenFace/ Facenet nn4.small model](https://cmusatyalab.github.io/openface/models-and-accuracies/#model-definitions)
<br>[Oarriaga/ mini_XCEPTION Emotion Model](https://github.com/oarriaga/face_classification)
<br>[WideResNet Age_Gender_Model](https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)
<br>Face Recogition Explained and Codes: [Karasserm](http://krasserm.github.io/2018/02/07/deep-face-recognition/)
<br>Age/Gender Prediction Explained and Codes: [Chengwei](https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)

## Installations
To run the model, please install the required python packages using
```
pip install -r requirements.txt
```
**To run real-time demo:**
```
python face_reco_demo.py
```
**To run face recognition model:**<br>
```
python face_reco_base.py
python face_reco_image.py
python face_reco_video.py
```
**Image:**
```
face = FaceImage()
display_labeled_image(face, "sample/sample01.jpg")
```
<img src="https://github.com/rileykwok/Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations/blob/master/sample/sample01result.jpg" width="400">

**Video to video:**
```
labeled_video = FaceVideo("sample/sample02.mp4")
labeled_video.create_mp4_video("sample02vid.mp4")
```
**Video to GIF:**
```
labeled_video = FaceVideo("sample/sample02.mp4")
labeled_video.create_animated_gif("sample02gif.gif")
```
<img src="https://github.com/rileykwok/Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations/blob/master/sample/sample02gif.gif" width="400">

**To train new faces for face identification:**<br>
Import photos to `faces/name_of_person/001.jpg` and run the codes again.

**To visualise face vectors of trained faces (using t-SNE dimension reduction):**
```
face_recognizer.visualize_dataset()
```
