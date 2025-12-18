import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset#,transform_targets


from yolov3_tf2.models import (
    yolo_anchors, yolo_anchor_masks
)


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('folder_name', 'fiftyone_data', 'folder name for fiftyone')
flags.DEFINE_string('weights', './checkpoints/yolov3.weights.h5',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')

flags.DEFINE_string('train_tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('traincv_tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('cv_tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('test_tfrecord', None, 'tfrecord instead of image')


flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
import pdb



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.summary()

    yolo.load_weights(FLAGS.weights)  # .expect_partial()
    logging.info('weights loaded')

    def cx(a):

        # axy=(a[:,:,2:4]+a[:,:,0:2])/2

        amin = a[:, :, 0:2]
        awh = a[:, :, 2:4] - a[:, :, 0:2]

        a = tf.concat([amin, awh], axis=-1)

        return a


    full = [FLAGS.traincv_tfrecord, FLAGS.cv_tfrecord, FLAGS.test_tfrecord]

    full = [load_tfrecord_dataset(tfrec, FLAGS.classes, FLAGS.size) for tfrec in full]



    anchor_masks = yolo_anchor_masks

    anchors = yolo_anchors

    # full = [tfrec.map(lambda i, l: (transform_images(tf.expand_dims(i,0),FLAGS.size), tf.expand_dims(l, 0))) for
    #         tfrec in full]


    full = [tfrec.map(lambda i, l: (transform_images(i,FLAGS.size),l)).batch(8) for
            tfrec in full]

    def dsgf(i,l):

      b,s,c,n,a=yolo(i)

      return i,l,(b,s,c,n,a)

    #x=[tfrec.map(lambda i,l: yolo(i)) for tfrec in full]
    # x=[tfrec.map(dsgf) for tfrec in full]



    full = [tfrec.map(lambda i,l: dsgf(i,l)) for tfrec in full]
    #full = [tfrec.map(lambda i,l: (i, l, yolo(i))) for tfrec in full]


    # for i in full[0].take(1):
    #   print(i)

    # import pdb
    # pdb.set_trace()



    # pdb.set_trace()
#old
    full = [tfrec.map(
        lambda i, l, p: (i, tf.concat([cx(l[:, :, 0:4]), l[:, :, 4:5]], axis=-1), (cx(p[0]), p[1], p[2], p[3],p[4]))) for
        tfrec in full]

    os.makedirs('testx', exist_ok=True)
    os.makedirs('testx/predictions', exist_ok=True)
    os.makedirs('testx/labels', exist_ok=True)
    os.makedirs('testx/images', exist_ok=True)

    ds_names = ['traincv', 'cv', 'test']

    for n, ds in enumerate(full):

        os.makedirs(f'testx/labels/{ds_names[n]}', exist_ok=True)
        os.makedirs(f'testx/predictions/{ds_names[n]}', exist_ok=True)
        os.makedirs(f'testx/images/{ds_names[n]}', exist_ok=True)
        t=0

        for bi, bl, bp in ds:

          ab, bb, cb, numsb,anc_idsb=bp

          bi=tf.cast(bi * 255, tf.uint8)

          # print(start,bi.shape)


          for i,l,a, b, c, nums,anc_ids in zip(bi, bl,ab, bb, cb, numsb,anc_idsb):

            # print(i)

            #i = i[0]
            # tf.io.write_file(f"testx/images/{ds_names[n]}/example{t}.jpg",
            #                  i)

            
            t+=1
            
            tf.io.write_file(f"testx/images/{ds_names[n]}/example{t}.jpg",
                             tf.image.encode_jpeg(i))


            # tf.io.write_file(f"testx/images/{ds_names[n]}/example{t}.jpg",
            #                  tf.image.encode_jpeg(tf.cast(i * 255, tf.uint8)))

            #l = l[0]

            with open(f"testx/labels/{ds_names[n]}/example{t}.txt", "w") as f:
                for t1 in l:
                    if sum(t1[:4]) == 0:
                        continue
                    else:
                        xmin = max(t1[0].numpy(), 0.0)
                        ymin = max(t1[1].numpy(), 0.0)

                        width = t1[2].numpy()
                        height = t1[3].numpy()

                        class_id = int(t1[4].numpy())

                        annstr = f"{xmin} {ymin} {width} {height} {class_id}\n"

                        print(annstr)

                        f.write(annstr)

            #a, b, c, nums,anc_ids = p

            #a, b, c, nums,anc_ids = a[0], b[0], c[0], nums[0],anc_ids[0]

            

            # Create (or overwrite) a txt file
            with open(f"testx/predictions/{ds_names[n]}/example{t}.txt", "w") as f:
                for i in range(nums):
                    xmin = max(a[i][0].numpy(), 0.0)
                    ymin = max(a[i][1].numpy(), 0.0)

                    width = a[i][2].numpy()
                    height = a[i][3].numpy()
                    confidence = b[i].numpy()
                    class_id = int(c[i].numpy())


                    #print(anc_ids)

                    anc_id=int(anc_ids[i].numpy())

                    annstr = f'{xmin} {ymin} {width} {height} {confidence} {class_id} {anc_id}\n'

                    f.write(annstr)

                    print(annstr)
            


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass