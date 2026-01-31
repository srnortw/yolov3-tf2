

# import cv2
# #print(cv2.getBuildInformation())
# #cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # Works with GTK backend
# img=cv2.imread('detected_result_animals.jpg')
# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import subprocess
import numpy as np

import ai_edge_litert.interpreter as tflite
from yolov3_tf2.models_raspi3bp import (
    yolo_boxes_numpy as yolo_boxes,yolo_nms_numpy as yolo_nms,pick_anchors,yolo_anchor_masks,draw_outputs

)

import time
import random


import os 
import urllib.request


# -----------------------------
# Download Weights From Github Releases
# -----------------------------

with open("github_releases_file_urls.txt") as f:
    model_urls = [line.strip() for line in f if line.strip()]

model_paths = []

for url in model_urls:
  filename = url.split("/")[-1]
  path = os.path.join("checkpoints", filename)
  model_paths.append(path)

  if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)



model_file='yolov3_animals_uint8'
MODEL_PATH = f"checkpoints/{model_file}.tflite"


SCORE_THRESHOLD = 0.5
INPUT_SIZE = 416
# -----------------------------
# Load model and labels
# -----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if model_file.startswith("yolov3_animals"):
  CLASS_PATH = "data/animals_class_names.txt"
  yolo_anchors=pick_anchors(res=INPUT_SIZE)
elif model_file.startswith("yolov3_construction_safety"):
  CLASS_PATH = "data/_darknet.labels"
  yolo_anchors=pick_anchors('construction_safety',res=INPUT_SIZE)


class_names = [c.strip() for c in open(CLASS_PATH).readlines()]

num_classes=len(class_names)


# -----------------------------
# Assign a random but fixed color to each class
# -----------------------------
random.seed(42)  # ensures same colors every run
class_colors = {name: tuple([random.randint(0, 255) for _ in range(3)]) for name in class_names}

WIDTH = 640
HEIGHT = 480

cmd = [
    "rpicam-vid",
    "--timeout", "0",
    "--inline",
    "--framerate", "30",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--codec", "yuv420",   # IMPORTANT: raw YUV, no MJPEG
    "--nopreview",
    "-o", "-"              # pipe to stdout
]

# YUV420 frame size = width * height * 1.5
frame_size = int(WIDTH * HEIGHT * 1.5)

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_size*3)

while True:
    # Read raw frame from pipe
    raw = p.stdout.read(frame_size)

    if len(raw) != frame_size:
        print("Camera closed")
        break

    # Convert YUV420 ? BGR for OpenCV
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    
    
    image=frame
    
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE))
    input_data = np.expand_dims(input_image, 0).astype(np.uint8) #/ 255.0

    
    # -----------------------------
    # Inference
    # -----------------------------
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()    


    output_early=output_details[1]#13 grid

    output_mid=output_details[0]#26 grid #102

    output_deep=output_details[2]#52 grid

    inference_time = time.time() - start_time
    print(f"Inference + NMS + output parsing: {inference_time:.3f} s")

    outputs=[output_early,output_mid,output_deep]

    anchors,masks=yolo_anchors,yolo_anchor_masks
        
    boxes=[]

    for i,output in enumerate(outputs):
        output_data=interpreter.get_tensor(output['index'])

        #details = output_details[i]
        scale, zero_point = output['quantization']
        real_output = scale * (output_data.astype(np.float32) - zero_point)
        
        
        #print(scale,zero_point,i,real_output)

        # real_output=output_data

        # print(np.min(real_output), np.max(real_output))
        
        box=yolo_boxes(real_output, anchors[masks[i]],num_classes)[:3]
        
        #print(box[0][0][0][0],i)

        boxes.append(box)

    boxes=tuple(boxes)

    #print(boxes[0][0][0],'heeey')

    # import pdb
    # pdb.set_trace()
    #boxes,anchors,masks,num_classes



    boxes, scores, classes, nums,anch_nums =yolo_nms(boxes,num_classes)

    # import pdb
    # pdb.set_trace()

    # -----------------------------
    # Draw outputs
    # -----------------------------
    # def draw_outputs(img, boxes, scores, classes, nums, class_names,class_colors,anch_nums):
    #     img_h, img_w = img.shape[:2]
    #     wh = np.array([img_w, img_h, img_w, img_h])
        
    #     batch=len(nums)
    #     for b in range(batch):
    #         for i in range(nums[b]):
    #             if scores[b][i] >= SCORE_THRESHOLD:
                    
    #                 print(boxes[b][i],anch_nums[b][i])
    #                 box = boxes[b][i] * wh
    #                 x1, y1, x2, y2 = box.astype(np.int32)
    #                 label = class_names[int(classes[b][i])]
    #                 score = scores[b][i]
    #                 color = class_colors[label]  # use pre-assigned color
    #                 cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)#(255, 0, 0)
    #                 cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 5),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255-color[0],255-color[1],255-color[2]), 2)#(0, 0, 255)
    #     return img
        
    # import pdb
    # pdb.set_trace()
    image_out = draw_outputs(image, boxes, scores, classes, nums, class_names,class_colors,anch_nums,SCORE_THRESHOLD)    
    
    
    
    cv2.imshow("Camera", image_out)
    if cv2.waitKey(1) == ord('q'):
        break

p.terminate()
cv2.destroyAllWindows()

