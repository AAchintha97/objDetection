# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:55:36 2023

Purpose: Extract and display information about the detected objects in an image using RCNN, including its type (class), coordinates (bounding box), and confidence score (probability).

@author: Achintha Aththanayake
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from mods_folder import ops as utils_ops
import json

def objDetectionRCNN(image):
    #Read label information--------------------------------------------------------
    with open('mods_folder/old_oid_labels.json') as f:
        old_oid = json.load(f)
    
        
    #Set Model and Label Paths-----------------------------------------------------
    MODEL_NAME = 'r-faster-cnn-inception-v2'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    
    
    #Load the Frozen Graph---------------------------------------------------------
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
        
    #Convert Images to NumPy Arrays------------------------------------------------
    def load_image_into_numpy_array(image):
      image = np.asarray(image)
      #(im_width, im_height) = image.size
      return image #np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    
    def run_inference_for_single_image(image, graph):
      with graph.as_default():
        with tf.compat.v1.Session() as sess:
          #Get handles to input and output tensors---------------------------------
          ops = tf.compat.v1.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
              
          if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            
            #Reframe is required to translate mask from box coordinates to 
            #image coordinates and fit the image size.-----------------------------
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            
            #Follow the convention by adding back the batch dimension--------------
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            
          image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
    
          #Run inference-----------------------------------------------------------
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
    
          #Convert types as appropriate--------------------------------------------
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
    
    #Define the get_class function-------------------------------------------------
    def get_class(id_num):
        class_label = ''
        for i in old_oid:
            if i['id'] == id_num:
                class_label = i['name']
                break
        return class_label
    
    #Define the get_class_name function--------------------------------------------
    def get_class_name(id_num):
        class_name = ''
        for i in old_oid:
            if i['id'] == id_num:
                class_name = i['display_name']
                break
        return class_name

    
    #Convert the image to a NumPy array--------------------------------------------
    image_np = load_image_into_numpy_array(image)
 
    #Perform object detection and get the results----------------------------------
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    #Get detected object information-----------------------------------------------
    detection_boxes = np.around(output_dict['detection_boxes'], decimals=2)
    
    #Get indices of detected objects-----------------------------------------------
    ind = np.where(detection_boxes.any(axis=1))[0]
    ind = list(ind)
    
    #Initialize an empty list to store detection objects---------------------------
    detection_list = []

    #Loop through the detected objects---------------------------------------------
    for i in ind:
        l = output_dict['detection_boxes'][i]
        id_num = output_dict['detection_classes'][i]
        cls_name = get_class_name(id_num) 
        prob = round(output_dict["detection_scores"][i], 2)
        
        #Create a JSON object for the detected object------------------------------
        detection_object = {
            "Object type": cls_name,
            "Coordinates": {
                "x0": round(l[1], 2),
                "y0": round(l[0], 2),
                "x1": round(l[3], 2),
                "y1": round(l[2], 2)
            },
            "Probability": prob
        }
        
        #Append the detection object to the list-----------------------------------
        detection_list.append(detection_object)

    return detection_list
