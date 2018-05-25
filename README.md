# Face-Recognition-Model-with-Gender-Age-and-Emotions-Estimations
Capstone Project by [Bertrand Lee](https://github.com/bertrandlee) and [Riley Kwok](https://github.com/rileykwok)

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
With Donald Trump's and Barack Obama's faces trained in the SVM classifer, the face recognition system is able to identify them in the photo with emotion predicted. For untrained faces, e.g. Melania Trump's, Michelle Obama's and the guard's faces, the system classfied them as 'unknown' and predicted their gender, age and emotion.

<photo>

This shows how a customer may respond emotionally to different products in-store and how the system is able to track his emotions.
<gif>

## References
Face Detection and Alignment: 
[DLib](http://dlib.net/) and
[OpenCV:](https://opencv.org/)
<br>Face Recognition Model: 
<br>[OpenFace/ Facenet nn4.small model](https://cmusatyalab.github.io/openface/models-and-accuracies/#model-definitions)
<br>[Emotion Prediction Model](https://github.com/oarriaga/face_classification)
<br>[Age/Gender Prediction Model](https://www.dlology.com/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)






