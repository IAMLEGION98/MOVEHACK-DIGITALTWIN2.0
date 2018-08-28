# VisionXRajasthanHack5.0
 To provide route optimization( connecting paths of freight carriers and suggesting rerouted paths). To minimize the FOB cost of manufacturing a company good, prior to export.
 TOOLS USED:  MATLAB , Jupyter Notebook, Online API – BigML, ThingSpeak, OpenCV, Spyder, Unity3d, Vuforia library, Raspberrypi3b+, Picam software, Blender.
 
 
 
 The datasets for the complete model have been provided in two files: ‘fin.csv’ and ‘eye_last.csv’.
1.	COST ESTIMATION - fin.csv : This file contains the dataset for cost estimation with three parameters : Item, Quantity and Region. The three parameters are used to predict the Price for the buyer.
•	We have chosen only raw materials as the items to facilitate in making a correct real-time estimation of the price for the buyer based on varying economic factors.
•	We upload the dataset in BigML and run different models to get a correct prediction of the price

•	For running the SVM model in MATLAB:
1.	Open MATLAB and change directory to the folder ‘svm_model’, which has been provided.
2.	Type and run ‘svm’ in the MATLAB command line

For AR and IOT Private.
Thingspeak server  GET https://api.thingspeak.com/channels/517882/feeds.json?api_key=QK46LGJTCUISK236&results=2
The Thingspeak Server, is a private online data collection point that feeds in on sensory values from individual freight containers, commuting trucks, all through gps enabled systems; prividing realtime and space monitoring of the commodity.
