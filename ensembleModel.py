# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:55:44 2023

Purpose: Ensemble model with YoloV8 and FRCNN

@author: Achintha Aththanayake
"""

def ensembleModel(detection_list_Y8, detection_list_RCNN):
    #Store all objects---------------------------------------------------------
    allObjects = []
    
    #Comapre the object types of yoloV8 with RCNN and add it-------------------
    for itemY8 in detection_list_Y8:
        objectTypeY8 = itemY8.get('Object type')
        
        for itemRCNN in detection_list_RCNN:
            #compare with case insensitive since different models give same object type in different cases
            if (itemRCNN.get('Object type').casefold() == objectTypeY8.casefold()):
                if itemY8['Probability'] > itemRCNN['Probability']:
                    allObjects.append(itemY8)
                else:
                    allObjects.append(itemRCNN)
    
    
    #Get lowercase object types of each----------------------------------------
    itemY8_lower = [itemY8.get('Object type').lower() for itemY8 in detection_list_Y8]
    itemRCNN_lower = [itemRCNN.get('Object type').lower() for itemRCNN in detection_list_RCNN]                
    allObjects_lower = [allObject.get('Object type').lower() for allObject in allObjects] 
    
    
    #Add other elements in the detection_list_Y8 to allObjects-----------------
    for itemY8 in itemY8_lower:
        if itemY8 not in allObjects_lower:
            newElement = detection_list_Y8[itemY8_lower.index(itemY8)]
            allObjects.append(newElement)
    
            
    #Add other elements in the detection_list_RCNN to allObjects---------------
    for itemRCNN in itemRCNN_lower:
        if itemRCNN not in allObjects_lower:
            newElement = detection_list_RCNN[itemRCNN_lower.index(itemRCNN)]
            allObjects.append(newElement)
            
            
    return allObjects