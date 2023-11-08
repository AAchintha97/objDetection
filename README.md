# objDetection
An ensemble model that leverages the strengths of both YOLO and Faster R-CNN while mitigating their weaknesses, resulting in a balanced and robust object detection system

Problem to solve: Find the optimal balance between precision and accuracy in object detection.

Method used: Create ensemble model which grabs the most accurate probability comparing both model and draw the bounding box. Additionally it provide other detected objects of each models by reducing the probability to miss objects.

Advantages: The model achieves a harmonious blend of precision and reliability, offering dependable object detection results. Moreover, it proves highly versatile, finding utility across diverse domains, from autonomous driving to surveillance and medical diagnostics.

Files included ==>
>>>>Python files
*app.py: Flask app
*boundingBoxDrawer: Draws the bounding box of the detected object
*ensembleModel: Compares the detected objects of each model and output the best bounding box according to probability (ensemble model).
*objDetectionRCNN
*objDetectionY8

>>>>Other files
*Dockerfile: Includes docker commands
*requirements: Includes libraries required

>>>>Folders
*mods_folder: Includes labels of FRCNN
*r-faster-cnn-inception-v2: Includes frozen_inference_graph of FRCNN
*templates: Includes index.html, style.css and pet.png which are used to develop web interface.