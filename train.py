from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

import yolov3_tf2.models
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.weights.h5',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'autograd_graph_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'autograd_graph_tf: custom GradientTape in graph mode')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune','open'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('lambda_reg', 1.0, 'lambda regularization parameter')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_boolean('cosine_learning_decay_restarts',False, 'cosine learning decay restart function')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')


def setup_model():
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(None,training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
            
            # for i in range(3):
            #   model.get_layer('yolo_conv_'+str(i)).set_weights(
            #       model_pretrained.get_layer('yolo_conv_'+str(i)).get_weights())
            #   freeze_all(model.get_layer('yolo_conv_'+str(i)))
              
            
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                # l.set_weights(model_pretrained.get_layer(
                #         l.name).get_weights())
                # if not l.name.startswith('yolo_output'):
                #   freeze_all(l)

                if not l.name.startswith('yolo_output'):
                  l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                  
                  freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)
        else:
          pass


    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=FLAGS.learning_rate,
    # decay_steps=60,
    # decay_rate=0.95,
    # )

    # decay_steps = 3000  # depends on dataset size; e.g., num_batches_per_epoch * num_epochs
    # alpha = 0.1         # final learning rate = alpha * initial_lr

    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=7e-4,
    #     decay_steps=decay_steps,
    #     alpha=alpha
    # )

    if FLAGS.cosine_learning_decay_restarts:
      training_size=1204#997#1080#638*80/100#1504

      #training_size=448

      steps_per_epoch=training_size/FLAGS.batch_size

      from math import ceil


      first_decay_steps=int(10*ceil(steps_per_epoch))

      lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
      initial_learning_rate=FLAGS.learning_rate,#0.1,
      first_decay_steps=first_decay_steps,#376,
      t_mul=2.0,   # cycle durations: 1000, 2000, 4000...
      m_mul=0.95,   # max LRs: 0.1, 0.09, 0.081... #0.9
      alpha=0.03 # 0.05
      )
    else:
      lr_schedule=FLAGS.learning_rate

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)




    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)#,name=f"yolo_output_{i}")
            for i,mask in  enumerate(anchor_masks)]


    # z=[]
    # t=[]
    # for i in loss:
      
    #   def total(x,y):
    #     return i(x,y)[0]
    #   t.append(total)

    #   def xy_wh(x,y):
    #     return i(x,y)[1]

    #   def obj(x,y):
    #     return i(x,y)[2]

    #   def cls(x,y):
    #     return i(x,y)[3]

    #   z.append([xy_wh,obj,cls])

    # metrics=z
    # loss=t



    model.compile(optimizer=optimizer, loss=loss,#,metrics=metrics,#t,#metrics=z,
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    
    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=602)#540#498


    # for v in train_dataset.take(1):
    #   print(v)

    # base_seed=(52,0)
    
    train_dataset=train_dataset.enumerate().map(lambda i,inputs: dataset.augmentation(inputs[0],inputs[1],i))
    
    dataset_name=FLAGS.dataset.split('/')[-1]

    import matplotlib.pyplot as plt
    import cv2
    import os

    os.makedirs(f'data/{dataset_name}_augmented_image_samples', exist_ok=True)
    for i,(l,im) in enumerate(train_dataset.take(25)):
      # print(l,im)
      l=tf.cast(l,tf.uint8)
      limg=cv2.cvtColor(l.numpy(), cv2.COLOR_RGB2BGR)
      plt.imshow(limg)
      plt.axis('off')
      plt.show() # test1 test1
      #plt.imshow(l.numpy())

      cv2.imwrite(f'./data/{dataset_name}_augmented_image_samples/output{i}.jpg', limg)

    # import pdb
    # pdb.set_trace()



    train_dataset = train_dataset.batch(FLAGS.batch_size)
    #
    # for i in train_dataset.take(1):
    #     print(i[1])
    # train_dataset = train_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, FLAGS.size),
    #     dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))


    # train_dataset = train_dataset.enumerate().map(lambda i,data: (
    #     dataset.transform_images(data[0],352+64*((i//10)%3)),
    #     dataset.transform_targets(data[1], anchors, anchor_masks,352+64*((i//10)%3)))) #320-640

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))


    if FLAGS.tiny:
        model_name='yolov3-tiny'
    else:
        model_name='yolov3'

    if FLAGS.transfer=='no_output':
      limit=13
    elif FLAGS.transfer=='fine_tune':
      limit=69
    else:
      limit=0

    if FLAGS.mode == 'autograd_graph_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        @tf.function
        def train(res,images,labels,lambda_reg=FLAGS.lambda_reg):#1.25
          # for batch, (images, labels) in enumerate(train_dataset):

          #print(res)

          images=dataset.transform_images(images,res)
          labels=dataset.transform_targets(labels,anchors, anchor_masks,res)


          with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
              pred_loss.append(tf.reduce_mean(loss_fn(label, output)))
            # total_loss = (tf.reduce_sum(pred_loss)/FLAGS.batch_size) + regularization_loss xx

            # total_loss = tf.reduce_mean(tf.reduce_sum(pred_loss,0),1) + regularization_loss
            total_loss = tf.add_n(pred_loss) + lambda_reg*regularization_loss


          grads = tape.gradient(total_loss, model.trainable_variables)
          optimizer.apply_gradients(
            zip(grads, model.trainable_variables))
                
          avg_loss.update_state(total_loss)
        
        @tf.function
        def evaluate(val_dataset,lambda_reg=FLAGS.lambda_reg):#1.5
          for batch, (images, labels) in enumerate(val_dataset):
            outputs = model(images)
            regularization_loss =  tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
               pred_loss.append(tf.reduce_mean(loss_fn(label, output)))
            # total_loss = (tf.reduce_sum(pred_loss)/FLAGS.batch_size) + regularization_loss xx

            # total_loss = tf.reduce_mean(tf.reduce_sum(pred_loss,0),1) + regularization_loss
            total_loss = tf.add_n(pred_loss) + lambda_reg*regularization_loss

            avg_val_loss.update_state(total_loss)



        for epoch in range(1, FLAGS.epochs + 1):
            # for batch, (images, labels) in enumerate(train_dataset):
                # with tf.GradientTape() as tape:
                #     outputs = model(images, training=True)
                #     regularization_loss = tf.reduce_sum(model.losses)
                #     pred_loss = []
                #     for output, label, loss_fn in zip(outputs, labels, loss):
                #         pred_loss.append(loss_fn(label, output))
                #     total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                # grads = tape.gradient(total_loss, model.trainable_variables)
                # optimizer.apply_gradients(
                #     zip(grads, model.trainable_variables))

                # logging.info("{}_train_{}, {}, {}".format(
                #     epoch, batch, total_loss.numpy(),
                #     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                # avg_loss.update_state(total_loss)
                
            #train(train_dataset)
            for batch, (images, labels) in enumerate(train_dataset):
              res=352+64*((batch//10)%3)#every 10 batch resolution increases 
              #res 480-  352 416 480
              #res=416
              train(res,images,labels)
                # logging.info("{}_train_{}, {}, {}".format(
                #     epoch, batch, total_loss.numpy(),
                #     list(map(lambda x: np.sum(x.numpy()), pred_loss))))

                

            # for batch, (images, labels) in enumerate(val_dataset):
                # outputs = model(images)
                # regularization_loss = tf.reduce_sum(model.losses)
                # pred_loss = []
                # for output, label, loss_fn in zip(outputs, labels, loss):
                #     pred_loss.append(loss_fn(label, output))
                # total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                # logging.info("{}_val_{}, {}, {}".format(
                #     epoch, batch, total_loss.numpy(),
                #     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                # avg_val_loss.update_state(total_loss)

            evaluate(val_dataset)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_state()
            avg_val_loss.reset_state()

                      
            if  epoch>limit:# epoch % 25 == 0:
              model.save_weights('checkpoints/{}_{}_{}_{}.weights.h5'.format(model_name,dataset_name,epoch,FLAGS.transfer))
    else:                    
          
        # class fAvg(tf.keras.callbacks.Callback):
        #     def on_epoch_end(self, epoch, logs=None):
              
        #       names=['obj','xy_wh','cls']

        #       avg=[]
        #       for i in names:
        #         cls=logs.get(f'val_yolo_output_0_{i}', 0)+logs.get(f'val_yolo_output_1_{i}', 0)+logs.get('val_yolo_output_2_{i}', 0)

        #         avg.append(cls)

        #       print(f'avg_obj:{avg[0]},avg_xy_wh:{avg[1]},avg_cls:{avg[2]}')


        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=2,min_lr=1e-15),
            EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.weights.h5',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
            #fAvg()#YoloLossTracker(loss)
        ]

        # for i in train_dataset.take(10):
        #     print(i[1][0])
        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
