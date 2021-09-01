# Economic profiling using Maps/Satellite-Images-Pepper-Global-UK

Problem Description

Deriving an alternate/secondary source of customer data with customer latitude-longitude as input, from developing countries with India being the main region of focus for Pepper-Global Financial Services using Computer Vision techniques from the Open-Source Maps/satellite images depending on whichever is free and easy to use based on their usage in the past research through extensive literature review. This data could be useful for Pepper in the future for economic profiling of the regions where the clients reside.
Using existing academic research as inspiration, source proxies for economic indicators and predictors found in successful local economy models in order to create Pepper’s own economic profile in the future.

Background of the project

Pepper Global is a Financial Services company with a globally diversified business portfolio. Pepper takes ownership, provides loan servicing, makes investments, manages and builds lending in an innovative manner. In developing countries such as India, Pepper has very limited access to the customer data which includes the customer address, customer name, phone number, date of loan sanction and the date when the loan became nonperforming.
Pepper invests across regions of UK, Europe, Southeast Asia and Australia. The Pepper India business specializes in non-Performing loans (NPL’s) or bad loans, third party loan servicing constituting of eighty- five percent of their Indian business. NPLs are the loans that the customer has either stopped paying or has never paid. Pepper buys the books of non-performing loan from the banks or financial institutions and tries to reach out or trace the customer to recover the maximum possible loan amount, willing to let go of the interest. Pepper-global is currently operating out of three regions in India. In India pepper majorly deals with loan servicing of unsecured personal debts, eighty-five percent of which are educational loans where the customer has stopped paying for more than 5 years and some of them have been written off by the banks. Apart from pure servicing, Pepper can also buy the books of loans from the bank, in which case it becomes an investor. Pepper needs to do an evaluation of such NPL’s and predict the probability of their recovery along with the estimated amount that could be recovered.
This certainty of loan repayment would enhance pepper’s commission on the book of loans that it services for the banks and make it easier for them to buy the books of loans which are having higher probabilities of repayment. Pepper wants to further expand its business in India into newer regions and become a loan provider and get involved in performing loans rather than just doing pure loan servicing of NPL’s. Building an alternate source of customer data would help Pepper in deciding on the regions of expansion by figuring out the impact of geography of a location on the loan recoverability.
In order to analyze if the investment in the non-performing loan books is profitable for the organization, Pepper needs to build a secondary data source from the customer address/ latitude-longitude they receive, as like any other organization their success relies on the risk and profitability assessment of the loans. This would help Pepper in minimizing the risk associated with the purchase of NPLs for loan servicing and continue growing their business in developing countries. Pepper’s profitability would largely increase with the ability to predict the chances of these loan recovery along with the estimated time it would take to recover and the percentage of loan that could be successfully recovered. This would additionally enable Pepper to move from non-performing to performing loans in the future in such countries.

Project Scope

In this project, the two major objectives are the literature review of the past research on satellite images and open-source map images for deriving economic indicators from such data which could be used as a secondary source of data for Pepper-Global to build an economic profile in the future. Based on the accessibility, ease of use and availability of open-source map images and satellite images, we would choose one amongst satellite images/open-source maps in agreement with Pepper-Global. We then apply computer vision object detection techniques which is one of the major applications of a convolution neural network to detect the various objects in these images along with the count of the occurrences of each class of object. The methodology used should be scalable in future to extract similar dataset for new map images with the customer latitude longitude details.

Project Objectives

The major objectives of this project can be summarized as under:
• Exploring the use of open-source Maps and satellite images to derive the economic indicators using only latitude-longitude or zip code as input by extensively referring the past research papers available.
• Evaluate the challenges and benefits of using Satellite images and open-source maps for analyzing the data.

Use either Open-Source Maps or Satellite images whichever is more convenient, for extracting local economy indicators.
• Brainstorming some of the economic indicators which could be used for economic profiling of a region.
• Evaluating the economic features that have been described in the past research papers
• Identify the computer vision, geo spatial analytics or machine learning methodologies used to
derive economic indicators from open-source maps and or satellite images.
• Build an alternative data source of a region using computer vision or geo spatial analytical
techniques which could be used by Pepper-Global in future to do the economic profiling.

The OSM data is useful due to the below reasons:

• The data is completely open source and crowdfunded with an editable access.
• These maps are updated regularly by its subscribed users who can add new points of interest to
the maps(SinghSehra et al., 2013).
• OSM can reach the developing regions of the world with the assistance of their global
volunteers. To obtain this kind of data would be tough for any commercial map providers(Kounadi, 2009).
One limitation of using OSM images is that it might not be very accurate across all cities, and we should be cautious to select the regions for which there is reliable data available. However, Pepper-Global operates out of the big cities in India and worldwide and OSM data is easily available for such regions.

Object Detection

Object classification and localization are the two major tasks under object detection. To recognize an object within an image, it is first localized to find the specific area where the object resides within the image and then boundaries called bounding boxes is created around the object and then object classification classifies/labels the objects into different classes. This is achieved with the help of region- based convolution neural network (RCNN) where a base neural network architecture is used for the classification task and then the region containing an object is extracted for localization. Therefore, Object detection can be summed up as convolution (classification) plus region (localization) (Ahn et al., 2020).
RCNN was the first object detection algorithm developed using computer vision, but it was proposing 2000 regions for each image, and this was a major issue. To overcome this problem, Fast RCNN and later faster RCNN was developed which reversed the process by first doing convolution to check whether a class is available and then localize it for every feature.

Object Segmentation

The pixel-wise classification of an object is called Semantic segmentation(Tyagi, 2021).Semantic segmentation is the most widely used object detection technique used to analyze the images captures through satellite sensors (Wang et al., 2020);(Chen et al., 2013). These images are used for object classification by first manually labeling or annotating the objects in the images(Guo et al., 2018).
Contrary to drawing bounding boxes like in object localization, in case of satellite images which are taken from top/vertical view it is essential to map the objects in the shape of polygons to generate pixels in the shape of an object and assigning a color to each object, thereby taking a pixel-based approach called segmentation.
Apart from object localization and classification, there are other image processing and computer vision methods that could be utilized to detect satellite images. Semantic segmentation first requires partitioning the images into different semantic segments to detect pixels with similar characteristics and cluster objects belonging to the same class in a group. The segmentation algorithms then classify the pixels into different categories so that every pixel is labeled after segmenting (Chen et al., 2013) .This technique has many widely used use cases such as land use and land cover classification and self-driving cars (can write about instance and other segmentation as well).
There are several state-of-the-art image segmentation algorithms used in computer vision: Mask RCNN, You Only Look Once (YOLO (V1-V5)), SSD, Mobile Net are amongst the most effective and widely used. Instead of generating pixels for the bounding boxes as in case of the RCNN methods, the segmentation methods have an additional layer added on top of the Faster RCNN architecture and it is this layer or network which is responsible for generating pixels in the shape of each object.

Layers in Object Detection Network:

An object detection neural network consists of backbone (trained on a popular dataset such as ImageNet) and a head (used for predicting the classes to which the object belongs). There are various state of the art CNN algorithms like VGG, Resnet-50, Resnet-101, Dense Net which could serve as the backbone architecture for object detection. The head could either be a one stage object detector (for dense predictions) or a two-stage object detector (for sparse predictions). The R-CNN series including faster R-CNN are popular two stage detectors and YOLO (v1 - v4), SSD and Retina Net are amongst the most popular one stage object detectors. In recent times a new layer has been introduced between the backbone and the head of the network to collect features from different stages. This layer is called as Neck. Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) are some of most popular neck layers. The YOLO v4 model uses CSPDarknet53 as its backbone, SPP and PAN architectures as its neck layers and the YOLO v3 architecture as its head.

YOLO v4

The YOLO v4 model was released on 23 April 2020. YOLO v4 is one of the lightest (least number of model parameters/weights), fastest (less training time required), and most accurate (higher accuracy for object detection and object tracking tasks) state of the art model developed till date. It is based on the transformer architecture (encoder-decoder layers) which was a breakthrough in Deep Learning. All the State-of the art models developed later including Google’s BERT and Open AI’s GPT models were built using the transformer architecture. This uses the famous attention-based mechanism used in deep learning network layers which was based on the research paper attention is all you need released by Google(Bochkovskiy, Wang and Liao, 2020).
YOLO v4 algorithm improved the real time accuracy of object detection along with their processing on large scale GPU’s (General Processing Units) which enabled such processes to be used on a mass scale at very affordable prices. The state-of-the-art transfer learning YOLO model has been trained on the popular Microsoft COCO dataset for detection accuracy, and ImageNet (ILSVRC 2012 val) dataset for classification accuracy; both of which have a very large image corpus for model training. The below image shows that the performance of YOLO v4 model when compared to some of the other state of the art object detectors outperforms the other models both in terms of speed (frames per second) and percentage accuracy (Bochkovskiy, Wang and Liao, 2020).

Improvements in YOLO v4 architecture(Sinigardi, 2020):
• It used the state-of-the-art models like EfficientNet, CSP and PRN
• A few hidden layers were added to this architecture to improve the object detection accuracy:
[conv_lstm]/[conv_rnn], [sam], [Gaussian_yolo], fixed[reorg3d], fixed[batchnorm]
• The new layers added made the model capable of accurate object detection on videos.
• Some Activation functions were added to the model architecture: Swish, Mish and
Norm_Chan_Softmax.
• To process higher number of images in each training step, the ability for training using CPU-RAM
was added to increase the mini-batch size which determines the number of images that will be
passed in each training iteration.
• By training our own model weights using XNOR-net model, the binary neural network
performance for object detection on CPU’s could be increased by two to four times
• By merging the convolutional and Batch normalization layers it improved the convolution neural
network performance by seven percent.
• Calculation of anchors were added for training.
• Setting random parameter as 1 in the configuration file, the memory allocation could be
optimized.
• The rectified calculation for performance metrics such as mean average precision (mAP), F1,
Recall and Intersection over union (IOU was added using the command darknet detector map)

The display of charts for average loss and mAP during training was added in this model which did not exist earlier.

Post processing methods; Non-Maximal Separation:

Activation functions are an integral component of any Deep learning architecture which determines which neural networks would be activated. Without substantial increase in the computational costs, it takes care of major issues like vanishing and exploding gradient (which prevents the weights of a neural network to change and reach an optimal value). The popular sigmoid and hyper tangent (tanh) functions were known to have the vanishing gradient issue. In 2010, the ReLU, parametric-ReLU, Swish were popular activation functions introduced to deal with such issues (Nair and Hinton, 2018).
Non-Maximal Separation (NMS) is a popular post processing method which is effectively used to separate out the bounding boxes that badly predicts the same object and retains only the ones with the highest response accuracy (each bounding box displays the percentage accuracy of detecting an object). The NMS also acts as an optimizer for the activation functions.

Data Augmentation

A mixture of data augmentation techniques was used for training the YOLO model such as Mixup, CutOut, CutMix etc.to increase the variance of the input image as any machine learning model needs to be trained with data having large variance. Additionally, techniques like different types of distortions (scaling, rotating, flipping, cropping) including geometric and photometric were used which are common for any object detection task. All these makes the model robust to images obtained under different environmental, lighting and noise conditions. All the data augmentation techniques used were pixel- based augmentation techniques which retained all the original pixel information even after augmentation(DeVries and Taylor, 2017).

Experiments

The Microsoft Azure cloud platform, provided by Pepper, is a Linux based operating system and uses MS Azure General Processing Units (GPU) has been used for training our YOLO architecture. The YOLO v4 model was originally trained on a single GPU with 8-16 GB-VRAM without using other expensive devices, to keep the model light weighted with fewer parameters/weights and allowing the results to be reproduced on any conventional GPU like RTX 2080Ti.
Several pre-trained classifiers and detectors were used as the backbone layers to test the prediction accuracy for both classifiers and detectors: CSPResNeXt50-PANet-SPP, CSPResNeXt50-PANet-SPP-RFB, CSPResNeXt50-PANet-SPP-SAM, CSPDarknet53-PANet-SPP (Bochkovskiy, Wang and Liao, 2020).
CSPResNeXt-50 models had the best classification accuracy however CSPDarknet53 model was most accurate in terms of object detection. The YOLO v4 model proved to be better in terms of both speed (Frames per second) and accuracy to the best of the state-of-the-art object detectors till April 2020 (Bochkovskiy, Wang and Liao, 2020).
Here we build our own custom object detector for which the following is required:
• Gathering the required OSM images
• A custom dataset of labelled OSM images
• custom .cfg file
• obj.data and obj. names file
• train.txt file
• test.txt file is optional for model validation on test images.

OSM Symbol/Object Mapping

The OSM Wikipedia symbols tab page provides us with the complete list of categories of objects grouped together into individual tabs. Additionally, various objects have been depicted with separate colour coding and lines/patterns so that they are easily identifiable. It is a comprehensive list of 19 different categories, each category consisting of several classes. These classes could be used as tags to retrieve their data from the OSM API’s by providing the tag name as parameter to the code (Open Street Map, 2021).
Below is the complete list of the broad categories along with few tags which could serve as economic indicators for our project (Open Street Map, 2021):
• Gastronomy: Cafes, Bars, Pubs, Restaurants.
• Culture, Arts and entertainment: Theatre, Cinema, Nightclub, Library, Museum
• Historical Objects: Memorial, Castle, Monument, Statues.
• Leisure, recreation and sports: Playground, Gym, swimming pool, golf course.
• Waste Management: Disposal bin, Waste basket, public toilets
• Outdoor: Weather Shelter, drinking water taps, bench
• Tourism and accommodation: Hotel, Apartment, Point of information
• Finance: Bank, ATM, Currency exchanges
• Healthcare: Hospital, Pharmacy, clinic, Dentist
• Communication: Post office, Post Box, public telephone
• Transportation: Bus stop, Gas/ petrol station, railway station, Airport, subways
• Road features: traffic lights, railroad crossing, one-way
• Nature: Tree, topographic saddle
• Administrative facilities: Police station, Fire station, Embassy, social facility, prison
• Religious places: temple, mosque, church, gurudwara
• Shops and services: Supermarket, coffee shop, tea shop, nursing home, travel agency, car wash, barber’s
• Landmarks, man-made infrastructure, masts and towers: Windmill, towers for lighting, bell tower
• Electricity: electric pole, big electricity pylon, carrying high voltage electricity cables
• Places: City, capital

Step 1: Image Collection

To build a custom object detector, the very first step is to collect a good dataset of images to train the object detector efficiently to detect the required objects. Open Street Maps are available to be downloaded and saved on its website (OpenStreetMap Foundation, 2020).To download the maps for any particular region, we just need to provide the latitude-longitude values as input to the map and the maps then provides us with the image containing all possible objects/landmarks within that region. These objects could be used as economic indicators to determine the performance/ wellbeing of the local economies. Keeping future objectives of scalability in mind, a custom code has been written using python programming language along with selenium webdriver, which automates the downloading of open street map images passing just the latitude-longitude as input parameters to the code. From the selenium webdriver packages different libraries have been imported for usage such as WebDriverWait, expected_conditions, options, time, and By. The list of latitude-Longitude could be fed into this program using a csv file, which would be read by the code sequentially and download the corresponding image on the local system by automatically opening the default web browser. This code also verifies the presence of objects in the region of the input latitude-longitude, and only downloads the map images if there are objects detected in that region, as otherwise the image would not be useful for our project.
This could be used by Pepper-global in future on a global scale to get the map images all around the world wherever their customers reside. It also reduces the time, effort and manpower needed in manual download of the images and could be extremely useful to train a production grade model in the future which requires training thousands of images for better accuracy, generalizability and to avoid overfitting where the model performs well on the training images, but the accuracy drops drastically on the unseen testing images/data (Code in Annexure 1).

Step 2: Data Annotation/ Labeling

There are various open-source tools available for labeling the images which are fed to a computer vision model. Label Me, Label Img, Label Studio, Rubrix, Ybat are amongst the most widely used annotation tools. Depending on the computer vision task at hand, we use the tool which are most effective for the specific task. For instance, Label Img is effective in drawing the bounding boxes over the regions containing the objects. Label Me is effective in segmentation purpose where the exact co-ordinates of the objects need to be labeled in a pixel wise manner. For our data labeling purpose, the LabelImg open- source graphical image annotation tool was first downloaded from their GitHub website (darrenl, 2018). LabelImg uses the python programming language at the backend, using Qt graphics view as its graphical interface. Qt uses a binary space partitioning tree for visualizing and detecting millions of objects within images in a real time. It is a cross platform language which works on windows, Linux, Mac and android systems by installing it in the native system. The annotations created by LabelImg were stored as XML files in the PASCAL VOC format which also supports the YOLO format. Python, PyQt5 and lxml libraries need to be installed through certain commands from the command prompt in windows. The python library lxml helps in conveniently handling the XML and Html files (Bochkovskiy, Wang and Liao, 2020).
The below annotated OSM image presents an example labeled image used for training our YOLO v4 model. The image was first imported to the labelImg tool, and then using the OSM key-value symbols tag mapping on the OSM Wikipedia page, the create RectBox option was selected to create rectangular boxes over the different classes of objects found in the image and each box was then labeled using the class name used to specify that category of object in the OSM symbols tag. For instance, classes such as outdoor, religious places, places of art and history could be seen tagged in the below annotated image(Open Street Map, 2021).
This then saves a txt file in YOLO format for each .jpg file in the same folder where the training images are stored. This file has the same name as the original image file with just a .txt extension. Also, each new line would contain the object number along with their geographical coordinates for each object.
Also, a file called classes.txt would be created simultaneously, which would have the list of all class names that we have labelled in our images and this list keeps on getting updated automatically on labeling new images. Each new line would consist of the following parameters:
<Object-class> <x- mean> <y-mean> <width> <height>
These parameters can be described as under (Bochkovskiy, Wang and Liao, 2020):
• <Object-class> - integer numbers from 0 to 8(classes-1)
• <width> <height> - floating point values containing the width and height of each image, it may
range from (0.0 to 1.0] where <width> = <absolute width> / <image width> or <height> =
<absolute height> / <image height>
• <x-mean> <y-mean> - are the mean/center of the rectangle
For example: for img4.jpg we will be having img4.txt containing: 0 0.716797 0.595833 0.916406 0.147222
0 0.287109 0.579167 0.258469 0.198333

Step 3: To train the YOLO v4 model the following steps needs to be followed (Bochkovskiy, Wang and Liao, 2020):
• First, we need to install the required frameworks for training our yolo v4 model over General Processing Unit (GPU). The below frameworks have been installed:
• The darknet model on which yolo v4 framework is trained is cloned from its repository.
• We use Tesla T4 GPU for model training of the images and make some changes in the makefile
which contains all the driver dependencies and make GPU and OpenCV by setting their values to
1.
• The GPU, CUDA and driver versions installed for training this model are shown in the below
picture (Bochkovskiy, Wang and Liao, 2020):
o CUDAversion>=10.2 o OpenCV >= 2.4
o cuDNN >=8.0.2
o GPU
o TensorFlow2.0

The darknet model on which yolo v4 framework is trained is cloned from its repository.
• We use Tesla T4 GPU for model training of the images and make some changes in the makefile
which contains all the driver dependencies and make GPU and OpenCV by setting their values to
1.
• The GPU, CUDA and driver versions installed for training this model are shown in the below
picture (Bochkovskiy, Wang and Liao, 2020):
  
Then we download the pre-trained weights of YOLO v4 (yolov4.conv.137) available on google drive to train the configuration file of yolo v4(cfg/yolov4-custom.cfg)
• A new file called yolo-obj.cfg was then created by copying the contents of cfg/yolov4- custom.cfg.
• The following parameters are then changed in the cfg/yolov4-custom.cfg files:
• The batch parameter has been updated to the value 64
• The subdivisions parameter was set to the value sixteen, if we receive out of memory error then
this parameter value can be increased to 32 or 64.
• The parameter max_batches was set to the value 18,000 (number of classes *2000) since we
have trained the model for 9 classes. The minimum value for this parameter should be 6000 and
this number should also be greater than the number of images used for training.
• The steps parameter was updated and set to eighty and ninety percent of the valuer for
max_batches (14,400, 16200)
• The network size parameters width and height parameters were both set to the values of 416
• Parameter classes value was set to 80
• The Filters/ kernel used in the last convolution layers just before each of the feed forward YOLO
layers was updated from 255 to (classes + 5) * 3 that is (8+5) *3 = 39
• A new file called obj.names was created in the directory build\darknet\x64\data\ with each line
containing the different objects/ class labels
• Another new file called obj.data was created under the same directory build\darknet\x64\data\
which contained classes = 9 (number of objects)
• The files containing the labeled images were then put under the directory
build\darknet\x64\data\obj\
• A train.txt file was made in the directory build\darknet\x64\data\ containing the names of the
images used for training for example: build\darknet\x64\data\img1.jpg
• Next, we downloaded the pre-trained YOLO v4 weights files called yolov4.conv.137 from its
google drive site respective to the configuration files which we changed above for our custom
object detection.
• This file is then placed under the directory build\darknet\x64
Finally, the model training was initiated using the command;./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137
• The custom object detection is then started.
• Two directories called train and test are created under the darknet folder where the training
and test images reside.

Results and Analysis
  
the model was trained for 10,000 epochs and was able to achieve an average error/loss of 0.25. Any error below 0.05 is considered excellent, hence our model performance is quite accurate in terms of mean average precision which on an average range above ninety percent on the training dataset. It can also be noted that if the average loss value is displayed as nan during the model training, it is an indication that there is something wrong with the training. The other fields can however display nan values which is acceptable.
Finally, the testing images were passes to test the model on the test dataset where the threshold value was set to 0.3 to display all objects having an accuracy of greater than thirty percent for detection.
The imshow (‘predictions.jpg’) command displays the prediction image with the detected objects and their individual accuracies for detection:

he model is successfully able to detect the objects present in the image such as art, religious and transportation. The accuracy of detecting each object is measured by a metric called mean average precision (mAP).
It shows the mean value of average precision for each class where average precision is given by the mean value of eleven points on the precision-recall curve for each possible probability of detection for the same class. Precision is measured by the total number of true positives divided by the sum of the true positives and the false positives.
  
The files containing the model best weights (yolo-obj_best.weights) for our custom object detection are saved in the backup folder build\darknet\x64\backup\. These are the most optimized weights for the convolution neural network model with the lowest loss/ error. This weights in this file could be used by Pepper in future for scaling up to train on images across the globe without training the model to learn the features from scratch as it has been trained on the map images to detect the objects which could be useful in understanding a local economy.
  
When Should We Ideally Stop Training the Model
For each class/object we should at least train for two thousand epochs/iterations. Also, this should be greater than the number of training images considered and there should be a minimum of six thousand iterations overall for all the images. However, a more precise stopping point would be when the average loss (error) no longer reduces after many iterations and is below the value of 0.05 (the lower this value, the better our model will be). If the training dataset is complex and huge (in millions) then an average loss in the range of 0.05 – 3.0 is considered good. Additionally, if we train with the flag -map then we will see the mean average precision (mAP) values in the console. We must continue training till the time mAP increases (Bochkovskiy, Wang and Liao, 2020).
  
How to get Weights from Early Stopping
In order to avoid overfitting our model, it is important to regularize it through early stopping when we observe than the training error is reducing but the validation error starts increasing. After stopping the training early, we should choose the best weights from the file last.weights from the folder \build\darknet\x64\backup, which would contain the weights for the last few batches (as per the number of images passes in each batch) using the mini batch gradient descent optimization algorithm. For example, if we stopped training after 8000 iterations, the best weights could be obtained in 7000 or
6000 iterations. The below command could be used to obtain these weights:
 • darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights
• darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_6000.weights
This might be due to overfitting where the model accurately detects objects in the training dataset, but the performance accuracy goes down with test images. The figure below illustrates how to select an early stopping point with respect to the average error and number of training iterations (Bochkovskiy, Wang and Liao, 2020).
  
Conclusion:
  
OSM provides a valuable open-source map data globally and artificial intelligence provides us with the means of deriving many indicators from different disciplines. Machine Learning techniques can be successfully applied on the map images to perform different socio-economic analysis. In this paper, we discussed and accessed the feasibility of usage of satellite images and open-source map images to derive the local economic indicators for different regions in India. This work goes beyond the state of the art to develop a standardized process to compare different regions with each other, track the economic development at different spatial and temporal scales across the entire region. This data would be beneficial for Pepper-Global to be used across different projects within the organization. Observing the dataset derived and analyzing them to find useful insights from was beyond the scope of this project.
The neural network based deep learning frameworks is far more effective in working with image data than the tree based random forest or other machine learning algorithms. In future, the classification obtained through the computer vision method could be fed as input variables to a machine learning model for deducing the local economic indicators. The evaluation of objects obtained through YOLO V4 algorithm provides us with encouraging insights into the different socio-economic variables present in any given region of the OSM image.

Recommendations for Future:
  
There is potential for further research in future to make the model globally scalable and adaptable to any region of the world for which such map images are available and access the variations required in the model to account for varying levels of completeness of map images and differences in the object tagging guidelines in different places. While building a local economic profile we must put greater emphasis on the outlier value for each indicator and take them into proper consideration. Furthermore, domain/local economic expertise is required to understand the economic indicators and their influence/effect on the local economy and do the economic profiling of a region accordingly to answer the questions like whether the customer residing in that region would repay the loan and how long could it take for them to repay the same. Additional analysis of the images with respect to a time series of images captured in different years of the same region to optimize the model performance/accuracy and thereby increase the predictive power of the machine learning methods to derive the local economic indicators. We could perform clustering on the features derived from the images and provide them as input to a clustering algorithm such a K-Means or DB Scan clustering which would further group the features into similar categories to come up with the economic profiling of any region.


Annexure 1:
Code to automate the downloading of Open Street Map Images:
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.by import By
import csv
with open('latlong.csv', mode='r') as latlong: csv_reader = csv.reader(latlong)
for row in csv_reader:
url = "https://tyrasd.github.io/overpass-turbo/?lat={}&lon={}&zoom=16".format(row[0], row[1])
options = Options()
driver = webdriver.Chrome(executable_path=r'./chromedriver/chromedriver.exe', chrome_options=options) driver.maximize_window()
driver.get(url)
WebDriverWait(driver, 30).until(
EC.presence_of_element_located((By.XPATH, "/html/body/nav/div[1]/div[1]/div/button[4]/span[2]")))
driver.find_element_by_xpath("/html/body/nav/div[1]/div[1]/div/button[4]/span[2]").click()
WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "export-image"))) driver.find_element_by_id("export-image").click()
time.sleep(1)
driver.quit()
Annexure 2:
Object detection Code:
# Check the GPU Versions and configurations
! nvidia-smi
# Mount your Google Drive
from google. colab import drive drive. mount('/content/drive')

# Clone darknet repo
! git clone https://github.com/AlexeyAB/darknet
# Change make file to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile !sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
# Verify CUDA
! /usr/local/cuda/bin/nvcc --version ! make
# Define helper functions
def imShow(path):
import cv2
import matplotlib. pyplot as plt %Matplotlib inline
image = cv2.imread(path)
height, width = image. Shape [:2]
resized image = cv2.resize(image, (3*width, 3*height), interpolation = cv2.INTER_CUBIC)
fig = plt.gcf ()
fig.set_size_inches (18, 10)
plt. Axis("off")
plt. Imshow (cv2.cvtColor(resized image, cv2.COLOR_BGR2RGB)) plt. Show ()
# Use this to upload files
def upload ():
from google. colab import files uploaded = files. Upload ()
for name, data in uploaded. Items ():
with open (name, 'wb') as f:
f. write (data)
print ('saved file', name)
# Use this to download a file
def download(path):
from google. colab import files files. Download(path)
#Create train test directory
import os os.mkdir('/content/darknet/data/obj')
os.mkdir('/content/darknet/data/test')
#Copy all the images from the drive
! cp r /content/drive/MyDrive/economic_profiling/train/*.* /content/darknet/data/obj ! cp r /content/drive/MyDrive/economic_profiling/test/*.* /content/darknet/data/test
# Download cfg to google drive and change its name
!cp /content/drive/MyDrive/economic_profiling/yolov4-obj.cfg ./cfg
# Verify that the newly generated train.txt and test.txt can be seen in our darkn
et/data folder
! ls data/
data
! cp /content/drive/MyDrive/economic_profiling/obj.data ./data
# Generating train.txt and test.txt
! cp /content/drive/MyDrive/economic_profiling/generate_train.py ./ !cp /content/drive/MyDrive/economic_profiling/generate_test.py ./
!python generate_train.py
!python generate_test.py
# Download pre-trained weights for the convolutional layers.
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
# Train your custom detector! (uncomment %%capture below if you run into me mory issues or your Colab is crashing)
!. /darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
imShow('chart.png')
# Need to set our custom cfg to test mode
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-obj.cfg
# Run your custom detector with this command (upload an image to your googl e drive to test, thresh flag sets accuracy that detection must be in order to show
it)
!. /darknet detector test data/obj.data cfg/yolov4-
obj.cfg /content/drive/MyDrive/economic_profiling/backup/yolov4- obj_best.weights /content/drive/MyDrive/economic_profiling/test/8.png -thresh 0.3
imShow('predictions.jpg')
! git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
%cd tensorflow-yolov4-tflite
! cp /content/drive/MyDrive/economic_profiling/coco.names /content/tensorflow-yolov4-tflite/data/classes ! cp /content/drive/MyDrive/economic_profiling/detect.py /content/tensorflow-yolov4-tflite
! cp /content/drive/MyDrive/economic_profiling/utils.py /content/tensorflow-yolov4-tflite/core ! python save_model.py --weights /content/drive/MyDrive/economic_profiling/backup/yolov4-
obj_best.weights --output /content/tensorflow-yolov4-tflite/ec --input_size 416 --model yolov4
! python detect.py --weights /content/tensorflow-yolov4-tflite/ec --size 416 --model yolov4 -- image /content/drive/MyDrive/economic_profiling/test/8.png
