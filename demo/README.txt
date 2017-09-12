Summary:
The demo used the trained fisher face classifier (fisherFace.xml) to detect emotions (happy, neutral, sad, disgust, surprise) shown in the camera. 

To run the demo:
- Install OpenCV and OpenCV_contrib on your python
  - check if needed modules are installed by: 
    import cv2
    help(cv2.xfeatures2d)
- Run the emtDet_video.py
- Stop the program by pressing any key

Performance:
The model can distinguish positive emotions from negative emotions well. But you may need to exaggerate your facial expressions to make the model recognize your correct negative emotions (sad, disgust, surprise).  

Author:
Feiyi Ouyang: ouyan103@umn.edu
Mayar Arafa: arafa004@umn.edu