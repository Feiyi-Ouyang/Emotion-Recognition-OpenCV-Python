Summary:
The files in the folder trained a KNN model to classify emotions using whole face images. The dataset and trained models are too large, so we only include the scripts training the models here:

- compute_dsift.py: extract the DSIFT features for the face images of three emotions (happy, sad, angry) and store them in the data.pkl

- KNN.py: load the data.pkl and train a KNN model to classify emotions based on these dense SIFT features. 

- KNNresult.rtf: testing results

Performance:
The KNN model achieved an accuracy of ~50%. 

Authors:
Mayar Arafa: arafa004@umn.edu
Feiyi Ouyang: ouyan103@umn.edu

