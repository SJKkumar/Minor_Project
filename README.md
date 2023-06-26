# Minor_Project

# BIOMEDICAL IMAGE SEGMENTATION USING DEEP LEARNING MODEL
Diagnosis of eye problems such as Retinal tear, Retinal detachment, Diabetic retinopathy, Epiretinal membrane done by automation’s which uses image segmentation algorithms to extract various
features from the eyes, lacks accuracy which leads to improper treatment and adverse consequences.
ACE-net is proposed with the intent of developing an image segmentation model that provides extracted features with higher accuracy in the field of bio-medicals than the previous models. ACE-net
is based on U-net algorithm. DRIVE dataset is imported which has train and test datasets for their
respective functions. The images in the dataset are first pre-processed. Both images and masks in
train and test dataset are converted to jpeg format. Then the ACE-net model is constructed it consist
of Four layers of ACB and AEB blocks. A detailed view into ACB and AEB blocks are explained
later in this paper. The model is trained with the train dataset so that in can be taught to extract only
the necessary features. After training the test dataset is used to obtain the extracted images. With this
result accuracy score of 95, F1 score of 48, jaccard score of 31, recall score of 88, and precision score
of 30 this project provided satisfactory results. An accuracy graph is plotted to represent the accuracy
of each test case

# Data importing and processing
Being able to import and export data is useful when you move data between databases that are used for
different purposes but need to share some data. between development and production databases. The
dataset we are going to use is the DRIVE (Digital Retinal Images for Vessel Extraction) dataset. The
classification of retinal vessels into arteries and veins is an important step for the analysis of retinal
vascular trees, for which the scientists have proposed several classification methods. An obvious concern regarding the strength of these methodologies is the closeness of the result of a particular method
to the gold standard. Unfortunately, the research community lacks benchmarks, resulting in increased
subjective error, biased opinion and an uncertain progress. This paper introduces a manually-labelled,
artery/vein categorized gold standard image database, as an extension of the most widely used image
set DRIVE. The labelling criterion is set after a careful analysis of the physiological facts about the
retinal vascular system. In addition, the labelling process also includes several versions of original
images to get certainty.
A two-step validation phase consists of verification from the trained computer vision observers and
a professional ophthalmologist, followed by a comparison with a gold standard set for the junction
locations introduced in V4-Like filters. We using OS module to open dataset the main purpose of the
OS module is to interact with your operating system. The primary use I find for it is to create folders,
remove folders, move folders, and sometimes change the working directory. You can also access the
names of files within a file path by doing listdir(). OS.walk() generate the file names in a directory
tree by walking the tree either top-down or bottom-up.For each directory in the tree rooted at directory
top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames) The DRIVE database has
been established to enable comparative studies on segmentation of blood vessels in retinal images.
Retinal vessel segmentation and delineation of morphological attributes of retinal blood vessels, such
18
as length, width, tortuosity, branching patterns and angles are utilized for the diagnosis, screening,
treatment, and evaluation of various cardiovascular and ophthalmologic diseases such as diabetes,
hypertension, arteriosclerosis and chorodial neovascularization. Automatic detection and analysis
of the vasculature can assist in the implementation of screening programs for diabetic retinopathy,
can aid research on the relationship between vessel tortuosity and hypertensive retinopathy, vessel
diameter measurement in relation with diagnosis of hypertension, and computer-assisted laser surgery.
Automatic generation of retinal maps and extraction of branch points have been used for temporal
or multimodal image registration and retinal image mosaic synthesis. Moreover, the retinal vascular
tree is found to be unique for each individual and can be used for biometric identification. The images
to train the deep-learning model is imported from the DRIVE (Digital Retinal Images for Vessel
Extraction) dataset and are resized. The DRIVE database has been established to enable comparative
studies on the segmentation of blood vessels in retinal images. Retinal vessel segmentation and
delineation of morphological attributes of retinal blood vessels, such as length, width, tortuosity,
branching patterns and angles are utilized for the diagnosis, screening, treatment, and evaluation of
various cardiovascular and ophthalmologic diseases such as diabetes, hypertension, arteriosclerosis
and chorodial neovascularization. This dataset is used for retinal vessel segmentation. It consists of
a total of JPEG 25 colour fundus images. The set of 25 images was divided into 20 images for the
training set and 5 images for the testing set. The images were obtained from a diabetic retinopathy
screening program in the Netherlands. The images were acquired using Canon CR5 non-mydriatic
3CCD camera with FOV equals to 45 degrees. Each image resolution is 584*565 pixels with eight
bits per colour channel (3 channels). Skimage. transform library is used to resize the images. Rescale
operation resizes an image by a given scaling factor. The scaling factor can either be a single floatingpoint value, or multiple values - one along each axis.
Resize serves the same purpose, but allows to specify an output image shape instead of a scaling
factor. The resized images are appended into an array which will be used to train the model. tqdm
is a Python library that allows you to output a smart progress bar by wrapping around any iterable.
A progress bar not only shows you how much time has elapsed but also shows the estimated time
remaining for the iterable. This library is used to show the progress (a progress bar) for the process
of resizing the images.
5.1.2 Training the model
Deep learning (DL) is a special type of machine learning that involves a deep neural network (DNN)
composed of many layers of interconnected artificial neurons. Training is the process of “teaching” a
DNN to perform the desired AI task (such as image classification or converting speech into text) by
19
feeding it data, resulting in a trained deep learning model. During the training process, known data
is fed to the DNN, and the DNN makes a prediction about what the data represents. Any error in
the prediction is used to update the strength of the connections between the artificial neurons. As the
training process continues, the connections are further adjusted until the DNN is making predictions
with sufficient accuracy.
This training process continues with the images being fed to the DNN and the weights being
updated to correct for errors, over and over again until the DNN is making predictions with the desired
accuracy. At this point, the DNN is considered “trained” and the resulting model is ready to be used
to make predictions against never-before-seen images. The model is trained with the DRIVE dataset,
to accurately extract the vessels from the image of a retina. Here TensorFlow is used to create the
deep-learning model. TensorFlow is an open-source artificial intelligence software library for using
data flow graphs to build models. It allows developers to create large-scale neural networks with
many layers. Nodes in the graph represent mathematical operations, while the graph edges represent
the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you
deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without
rewriting code. TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google’s Machine Intelligence research organization for the purposes of
conducting machine learning and deep neural networks research. The system is general enough to be
applicable in a wide variety of other domains, as well. TensorFlow is mainly used for: Classification,
Perception, Understanding, Discovering, Prediction and Creation. It has a comprehensive, flexible
ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art
in ML, and developers easily build and deploy ML-powered applications. The network was trained
on GPU provided by google colab with a mini-batch size of two
# Preprocessing
• INPUT: Retinal image in tif format
• OUTPUT: Retinal image in jpeg
• Step 1: The file name of the images are split into four arrays namely trainx, trainy, testx, testy
• Step 2: The images are sent through five filters namely horizontalflip, verticalflip, elastictransform, griddistortion, opticaldistortion. are saved in jepg format
• Step 3: The masks are converted format the mask are saved
