import streamlit as st

import os


import cv2
import numpy as np

import ai_edge_litert.interpreter as tflite

import time
import random

# os.chdir('raspi3bp')
# print(os.getcwd())
from yolov3_tf2.models_raspi3bp import (
    yolo_boxes_numpy as yolo_boxes,yolo_nms_numpy as yolo_nms,pick_anchors,yolo_anchor_masks,pick_anchors
)


# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


@st.cache_resource
def prepare(model_path):

    # -----------------------------
    # Load model and labels
    # -----------------------------

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter

    # new_model = tf.keras.models.load_model(f'vision/models/{model_name}.keras')
    #
    # # Show the model architecture
    # new_model.summary()
    #
    # with open(f"vision/unique_labels_folder/{model_name.split("_")[-2]}_unique_labels.pkl", "rb") as f:
    #     unique_labels = pickle.load(f)
    #
    # return new_model,unique_labels


# model_urls=open('git_releases_file_urls.txt').readlines()
# model_paths=['checkpoints/'+ model_url.split('/')[-1] for model_url in model_urls]
# MODEL_URL = "https://github.com/srnortw/yolov3-tf2/releases/download/weights/yolov3_animals_uint8.tflite"
# MODEL_PATH = "checkpoints/yolov3_animals_uint8.tflite"

import urllib.request

@st.cache_resource
def prepare0():
  
  with open("github_releases_file_urls.txt") as f:
      model_urls = [line.strip() for line in f if line.strip()]

  model_paths = []

  for url in model_urls:
    filename = url.split("/")[-1]
    path = os.path.join("checkpoints", filename)
    model_paths.append(path)

    if not os.path.exists(path):
      urllib.request.urlretrieve(url, path)

  return model_paths

@st.cache_data
def prepare1(class_path):


    class_names = [c.strip() for c in open(class_path).readlines()]

    # -----------------------------
    # Assign a random but fixed color to each class
    # -----------------------------
    random.seed(42)  # ensures same colors every run
    class_colors = {name: tuple([random.randint(0, 255) for _ in range(3)]) for name in class_names}

    return class_names,class_colors


# import os
#
# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
#
# os.chdir(parent_dir)

z=prepare0()

print(os.getcwd())

# model_file = 'yolov3_animals_uint8'
# MODEL_PATH = f"tflite-models/yolov3-animals/{model_file}.tflite"

model_files = [f for f in os.listdir('checkpoints') if f.endswith(".tflite")]

model_name= st.sidebar.radio("Pick A Model",model_files)

model_path='checkpoints/'+model_name

interpreter=prepare(model_path)


INPUT_SIZE = 416

#if model_name=='yolov3_construction_safety_objdet_train.tfrecord.gz_70_fine_tune.tflite':
if model_name.startswith('yolov3_construction_safety'):
  CLASS_PATH = "data/_darknet.labels"
  yolo_anchors=pick_anchors(name='construstion_safety',res=INPUT_SIZE)
else:
  CLASS_PATH = "data/animals_class_names.txt"
  yolo_anchors=pick_anchors(res=INPUT_SIZE)

class_names,class_colors=prepare1(CLASS_PATH)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


num_classes = len(class_names)

# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
# os.chdir(parent_dir+'/data')

if uploaded_file is not None:
    # Display file name

    file_content = uploaded_file.read()

    img_array = np.frombuffer(file_content, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    input_image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

    input_image = np.expand_dims(input_image, 0)


    # -----------------------------
    # Inference
    # -----------------------------
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'],input_image)
    interpreter.invoke()

    output_early = output_details[1]  # 13 grid

    output_mid = output_details[0]  # 26 grid #102

    output_deep = output_details[2]  # 52 grid

    inference_time = time.time() - start_time
    print(f"Inference + NMS + output parsing: {inference_time:.3f} s")

    outputs = [output_early, output_mid, output_deep]

    anchors, masks = yolo_anchors, yolo_anchor_masks

    boxes = []

    for i, output in enumerate(outputs):
        output_data = interpreter.get_tensor(output['index'])

        scale, zero_point = output['quantization']
        real_output = scale * (output_data.astype(np.float32) - zero_point)

        # print(scale,zero_point,i,real_output)

        # print(np.min(real_output), np.max(real_output))

        box = yolo_boxes(real_output, anchors[masks[i]], num_classes)[:3]

        # print(box[0][0][0][0],i)

        boxes.append(box)

    boxes = tuple(boxes)

    SCORE_THRESHOLD = st.sidebar.slider(
        "Score threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )

    soft_nms_sigma = st.sidebar.slider(
        "Soft sigma from none maximum supression",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.25
    )

    boxes, scores, classes, nums, anch_nums = yolo_nms(boxes, num_classes,SCORE_THRESHOLD,soft_nms_sigma)


    # -----------------------------
    # Draw outputs
    # -----------------------------
    def draw_outputs(img, boxes, scores, classes, nums, class_names, class_colors, anch_nums):
        img_h, img_w = img.shape[:2]

        ref = min(img_h, img_w)

        thickness_box = max(1, int(ref / 200))

        font_scale = ref / 600

        thickness_text  = max(1, int(ref / 400))

        wh = np.array([img_w, img_h, img_w, img_h])

        batch = len(nums)
        for b in range(batch):
            for i in range(nums[b]):
                if scores[b][i] >= SCORE_THRESHOLD:
                    #print(boxes[b][i], anch_nums[b][i])
                    box = boxes[b][i] * wh
                    x1, y1, x2, y2 = box.astype(np.int32)
                    label = class_names[int(classes[b][i])]
                    score = scores[b][i]
                    anch_id=anch_nums[b][i]
                    print(score,box,label,anch_id)
                    color = class_colors[label]  # use pre-assigned color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color,thickness_box)  # (255, 0, 0)
                    cv2.putText(img, f"{label} {score:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255 - color[0], 255 - color[1], 255 - color[2]),
                                thickness_text)  # (0, 0, 255)

        #2 0.5 2

        return img


    # import pdb
    # pdb.set_trace()

    # SCORE_THRESHOLD = 0.5


    image_out = draw_outputs(image, boxes, scores, classes, nums, class_names, class_colors, anch_nums)

    right, left = st.columns(2)

    with right:

        st.write("Filename:", uploaded_file.name)
                # Optionally, read contents
        st.write("File size (bytes):", len(file_content))

        st.image(image_out, caption="Uploaded Image", width=500)

        class_names

    # with left:
    #     class_names


#     image_rgb_uint = tf.image.resize(image_rgb_uint, input_shape, method='nearest')




