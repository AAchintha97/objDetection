# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 05:22:48 2023

Purpose: Extract and display information about the detected objects in an image using yoloV8, including its type (class), coordinates (bounding box), and confidence score (probability).

@author: Achintha Aththanayake
"""

from ultralytics import YOLO

def objDetectionY8(image):    
    #Initialize the YOLO model-------------------------------------------------
    model = YOLO("yolov8m.pt")

    #Perform object detection--------------------------------------------------
    results = model.predict(image)
    
    #Retrieve the results------------------------------------------------------
    result = results[0]
    
    #Access the # of detected objects, their coordinates and probabilities-----
    len(result.boxes)
    result.boxes
    
    #Initialize an empty list to store detection objects-----------------------
    detection_list = []
    
    for box in result.boxes:
        #image = cv2.imread(url) #image = np.array(url)
        #Extract bounding box coordinates of the object------------------------
        cords =box.xyxy[0].tolist()
        
        #Extract the class and convert it to an integer------------------------
        class_id =box.cls[0].item()
        
        #Extract the confidence score and convert it to a float----------------
        conf = box.conf[0].item()
        
        #Create a JSON object for the detected object--------------------------
        detection_object = {
            "Object type": result.names[class_id],
            "Coordinates": {
                "x0": round(cords[0], 2),
                "y0": round(cords[1], 2),
                "x1": round(cords[2], 2),
                "y1": round(cords[3], 2)
            },
            "Probability": round(conf, 2)
        }
        
        #Append the detection object to the list-------------------------------
        detection_list.append(detection_object)

    return detection_list