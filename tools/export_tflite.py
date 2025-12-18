import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny,yolo_boxes,yolo_nms,yolo_anchors,yolo_anchor_masks
)
import glob
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', './checkpoints/yolov3.tflite',
                    'path to saved_model')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('size', 416, 'image size')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes,training=True)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes,training=True)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    yolo.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(yolo)


    #Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]


    # Representative dataset generator for calibration

    def representative_data_gen():
      for image_path in glob.glob(f"testx/images/traincv/*.jpg")[:120]:  # 100 sample images
        img = cv2.imread(image_path)
        img = cv2.resize(img, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)/255.0
        yield [img]

    
    converter.representative_dataset = representative_data_gen

    # Fix from https://stackoverflow.com/questions/64490203/tf-lite-non-max-suppression
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]#, tf.lite.OpsSet.SELECT_TF_OPS]


    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    open(FLAGS.output, 'wb').write(tflite_model)
    logging.info("model saved to: {}".format(FLAGS.output))

    interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for d in output_details:
      print(d['name'], d['dtype'], d['quantization'])

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    image_raw = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)

    img=tf.image.resize(image_raw, (416,416))

    img = tf.expand_dims(img, 0)

    img=tf.cast(img,tf.uint8)

    #img = transform_images(img, 416)

    t1 = time.time()
    outputs = interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    # #import pdb
    # #pdb.set_trace()


    output_early=output_details[1]#13 grid

    output_mid=output_details[0]#26 grid

    output_deep=output_details[2]#52 grid

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
    
      box=yolo_boxes(real_output, anchors[masks[i]], FLAGS.num_classes)[:3]

      boxes.append(box)

    boxes=tuple(boxes)


    # print(boxes[0][0][0],'heeey')

    # import pdb
    # pdb.set_trace()
    boxes, scores, classes, nums,anchors_ids =yolo_nms(boxes,anchors,masks,FLAGS.num_classes)

    #print(boxes)

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {},{}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i]),
                                          np.array(anchors_ids[0][i])))


    #img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    image = cv2.cvtColor(image_raw.numpy(), cv2.COLOR_RGB2BGR)

    image = draw_outputs(image, (boxes, scores, classes, nums), class_names)

    loc='data/test_lite.jpg'

    cv2.imwrite(loc, image)
    logging.info('output saved to: {}'.format(loc))


    # print(output_data)

if __name__ == '__main__':
    app.run(main)
