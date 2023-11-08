# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:31:28 2023

Purpose: Draw bounding boxes according to ensemble model results

@author: Achintha Aththanayake
"""

import cv2
import random

def boundingBoxDrawer(detection_list, image):
    for item in detection_list:
        #Get object type and coordinates---------------------------------------
        objectType = item.get('Object type')
        coordinates = item.get('Coordinates')

        x0 = int(coordinates['x0'])
        y0 = int(coordinates['y0'])
        x1 = int(coordinates['x1'])
        y1 = int(coordinates['y1'])

        #Generate a random RGB color-------------------------------------------
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        #Draw bounding box with the random color-------------------------------
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        #Display object type with the same color-------------------------------
        cv2.putText(image, objectType, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    #return the image----------------------------------------------------------
    return image