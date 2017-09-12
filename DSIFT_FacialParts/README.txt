Summary:
The files in the folder trained a KNN model to classify emotions using facial parts images. The dataset and trained models are too large, so we only include the scripts training the models here:

- facialParts.py: use the Haar cascade for mouth with constraints on the detected areas to find left eye, right eye, and mouth areas of each faces, write the facial part images to jpg files.

- main.py: extract the dense SIFT features of facial part images; train SVM models to classify emotions for using the three facial part image repositories, respectively.

Performance:
The 3 SVM models achieved similar accuracies of ~23%, which is even lower than the whole-face DSIFT algorithm. Reflection on this was written in the report.

Authors:
Feiyi Ouyang: ouyan103@umn.edu

