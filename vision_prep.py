# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 02:51:47 2025

@author: Serkan
"""

import c_sql_d

import pdb
import numpy as np
import cv2 as cv
# from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# from concurrent.futures import ThreadPoolExecutor

import argparse

from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser(
    description='preparation_and_training')

parser.add_argument(
    '-dn',
    '--dataset_name',
    default='Indoors_objdet')  # sea_animals_objdet,animals_multilabel,eurosat,satelimgslocs,sports

parser.add_argument(
    '-res', '--resolution',
    type=int,
    default=256)

# parser.add_argument("--mode", default='client') when you run directly python console,uncomment this


inputs = parser.parse_args()

dataset_name = inputs.dataset_name
res = inputs.resolution

load_dotenv()
sql_d_o = c_sql_d.c_sql_d_c(os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME'), os.getenv('DB_USER'),os.getenv('DB_PASSWORD'))


#
sql = f"""
SELECT json_build_object(
    'img_loc', img_loc,
    'height', img_resh,
    'width', img_resw,
    'channel', img_resc,
    'objects', json_agg(
        json_build_object(
            'class_id', class_id,
            'class_name', class_name,
            'x1', x1,
            'y1', y1,
            'x2', x2,
            'y2', y2
        )
    )
) AS image_data
FROM {dataset_name}_view
GROUP BY img_loc,img_resh,img_resw,img_resc;
"""



query_gen=sql_d_o.query_generator(sql)

# for i in query_gen():
#     print(i)

def tform_query_generator(gen):
    def generator():
        for row in gen():
            rec = row[0]
            objs = rec["objects"]

            class_ids,class_names, x1s, y1s, x2s, y2s = map(list, zip(*[
                (o["class_id"],o["class_name"], o["x1"], o["y1"], o["x2"], o["y2"]) for o in objs
            ]))

            yield {
                "img_loc": rec["img_loc"],
                "height": rec["height"],
                "width": rec["width"],
                "channel": rec["channel"],

                "objects": {
                    "class_id": class_ids,
                    "class_name": class_names,
                    "x1": x1s,
                    "y1": y1s,
                    "x2": x2s,
                    "y2": y2s,
                }}
    return generator

query_gen=tform_query_generator(query_gen)

output_signature = {'img_loc':tf.TensorSpec(shape=(), dtype=tf.string),
                    'height':tf.TensorSpec(shape=(), dtype=tf.int32),
                    'width':tf.TensorSpec(shape=(), dtype=tf.int32),
                    'channel':tf.TensorSpec(shape=(), dtype=tf.int32),
                    'objects':{'class_id':tf.TensorSpec(shape=(None,), dtype=tf.int32),
                               'class_name':tf.TensorSpec(shape=(None,), dtype=tf.string),
                               'x1':tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'y1':tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'x2':tf.TensorSpec(shape=(None,), dtype=tf.float32),
                               'y2':tf.TensorSpec(shape=(None,), dtype=tf.float32)}}# img_loc)

ds = tf.data.Dataset.from_generator(
    lambda: query_gen(),
    output_signature=output_signature
    # output_signature = (
    #     tf.TensorSpec(shape=()),  # img_loc
    #     tf.TensorSpec(shape=(None, 5))  # objects list
    # )
)
# ds=sql_d_o.query_to_tfdata1(genf,output_signature)
#
# ds = sql_d_o.query_to_tfdata(sql, output_signature)
# #


# for i in ds.take(1):
#     print(i)
ds = ds.map(lambda dic: {**dic,#'img_loc':dic['img_loc'],
                               "objects":{"class_id": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['class_id'],tf.int32),axis=1), ragged_rank=1),
                                          "class_name": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['class_name'],tf.string),axis=1), ragged_rank=1),
                                         "x1": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['x1'],tf.float32),axis=1), ragged_rank=1),
                                         "y1": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['y1'],tf.float32),axis=1), ragged_rank=1),
                                         "x2": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['x2'],tf.float32), axis=1), ragged_rank=1),
                                         "y2": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(dic['objects']['y2'],tf.float32), axis=1), ragged_rank=1)}
                                         #"bbox": tf.RaggedTensor.from_tensor(objs[:,1:], ragged_rank=1)}
                               }
            )





#
# ds = ds.map(lambda img, objs: (img,{"filename":{img},
#                                "object":{"class_id": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(objs[:,0],tf.int32),axis=1), ragged_rank=1),
#                                          "bndbox":{
#                                          "xmin": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(objs[:,1],tf.float32),axis=1), ragged_rank=1),
#                                          "ymin": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(objs[:,2],tf.float32),axis=1), ragged_rank=1),
#                                          "xmax": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(objs[:,3],tf.float32), axis=1), ragged_rank=1),
#                                          "ymax": tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(objs[:,4],tf.float32), axis=1), ragged_rank=1)}},
#                                 "size":{"width":tf.shape(img)[0],
#                                         "height":tf.shape(img)[1],
#                                         }
#                                          #"bbox": tf.RaggedTensor.from_tensor(objs[:,1:], ragged_rank=1)}
#                                }
#             ))




sdaq_com = f'''

SELECT source,COUNT(*) FROM {dataset_name}_images

GROUP BY source

'''

output_signature = (
    tf.TensorSpec(shape=(), dtype=tf.string),  # img_loc
    tf.TensorSpec(shape=(), dtype=tf.int32))

# sdaq_ds = sql_d_o.query_to_tfdata(sdaq_com,output_signature)

sdaq_ds=sql_d_o.query_generator(sdaq_com)

sdaq_ds_tfd = tf.data.Dataset.from_generator(
    lambda: sdaq_ds(),
    output_signature=output_signature
)
import zipfile

for zf,q in sdaq_ds_tfd:
  if os.path.isdir(f'data/{dataset_name}'):
    pass
  else:
    zf_d=zf.numpy().decode('utf-8')
    print(zf_d,q)
    os.makedirs(f'data/{dataset_name}', exist_ok=True)
    with zipfile.ZipFile('data/'+zf_d, "r") as z:
      z.extractall(f'data/{dataset_name}')



#sdaq_df = sql_d_o.query(sdaq_com)

# all_meta_d_com = f'''
# SELECT image_id,source,img_loc FROM {dataset_name}
# '''
#
# all_metad_df = sql_d_o.query(all_meta_d_com)


#vvvvvv
# sql_d_o.close_connection()


resh = res
resw = res



def fix(f, reh, rew):
    xmin_norm = f['x1'] - f['x2'] / 2
    ymin_norm = f['y1'] - f['y2'] / 2
    xmax_norm = f['x1'] + f['x2'] / 2
    ymax_norm = f['y1'] + f['y2'] / 2

    xmin = xmin_norm  # * rew
    ymin = ymin_norm  # * reh
    xmax = xmax_norm  # * rew
    ymax = ymax_norm  # * reh

    return {**f,
            'x1': xmin,
            'y1': ymin,
            'x2': xmax,
            'y2': ymax}


# zipped = ds.map(lambda dic: {**dic, 'img': tf.image.resize(
#     tf.image.decode_jpeg(tf.io.read_file('tmp/' + dic['img_loc']), channels=3), [resh, resw], method='nearest'),
#                          'height': resh, 'width': resw,
#                          'objects': fix(dic['objects'], resh, resw)})



zipped = ds.map(lambda dic: {**dic, 'img_raw':tf.io.read_file(f'data/{dataset_name}/' + dic['img_loc']),
                         'objects': fix(dic['objects'], resh, resw)})


import input_preparation
# from input_preparation import inp_prep_f

batch_sz=200

zipped,stat2_m=input_preparation.inp_prep_f(zipped,batch_sz)

#zipped=zipped.cache()

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#
#
#
# zipped=zipped.map(lambda d,y,k:(preprocess_input(d),y,k),num_parallel_calls=tf.data.AUTOTUNE)#x
#

#
#
#
#
#initial_state=tf.zeros(len(unique_labels),dtype=tf.int32)
#
# # Reduce function
# def count_fn(state,inp):
#     #indx=md['label_id'].numpy().decode('utf-8')
#     md=inp[2]
#     return tf.tensor_scatter_nd_add(state, [[md['label_id']]], [1])
#
# # Apply take and reduce
# class_counts = zipped.reduce(initial_state=initial_state, reduce_func=count_fn)
#
# ratio=tf.cast(class_counts/tf.reduce_sum(class_counts),tf.float32)
# print("Before resampling:", ratio.numpy())
#
# # eq=1/len(unique_labels)
# #
# # eq=np.zeros(len(unique_labels))+eq
#
# eq = tf.constant([1.0 / len(unique_labels)] * len(unique_labels), dtype=tf.float32)
#
# def class_func(a,b,md,tr):
#     return tf.argmax(b,axis=-1)
#
# zipped = (
#     zipped
#     .rejection_resample(class_func, target_dist=eq,initial_dist=ratio)
#     .map(lambda extra_label, features_and_label: features_and_label)).cache()
#
# # for i,j in enumerate(resample_ds.take(50)):
# #     print(i)
#
# # Apply take and reduce
# resample_counts = zipped.reduce(initial_state=initial_state, reduce_func=count_fn)
# resample_counts=tf.cast(resample_counts,tf.int64)
#
# counts=tf.reduce_sum(resample_counts)
#
# ratio1=tf.cast(resample_counts/counts,tf.float32)
#
# print("After resampling :", ratio1.numpy())
#
# zipped=zipped.apply(tf.data.experimental.assert_cardinality(counts))
#
#
#
# zipped_cv_test=zipped.filter(lambda a0,b0,c0,d0:d0).map(lambda a,b,c,d:(a,b,c))
#
# zipped_train_traincv=zipped.filter(lambda a1,b1,c1,d1:~d1).map(lambda a,b,c,d:(a,b,c))


bboxes_com = f"""
SELECT x1,y1,x2,y2 FROM {dataset_name}_view;
"""
#-- WHERE img_loc ILIKE '%/train/%';
bboxes_df=sql_d_o.query(bboxes_com)




import numpy as np

def wh_iou(box, clusters):
    # box: (2,) = [w,h]
    # clusters: (k,2) = [[w,h],...]
    w, h = box
    inter_w = np.minimum(w, clusters[:,0])
    inter_h = np.minimum(h, clusters[:,1])
    inter = inter_w * inter_h
    union = (w*h) + (clusters[:,0]*clusters[:,1]) - inter
    return inter / (union + 1e-10)  # returns (k,)

def avg_iou(boxes, clusters):
    # boxes: (n,2), clusters: (k,2)
    return np.mean([np.max(wh_iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def iou_kmeans(boxes, k=9, max_iter=100, seed=42, use_median=True):
    """
    boxes: (n,2) array of widths and heights (in pixels at target input size)
    returns: clusters (k,2)
    """
    rng = np.random.RandomState(seed)
    # initialize: pick k boxes randomly as initial centroids
    centroids = boxes[rng.choice(boxes.shape[0], k, replace=False)].astype(float)

    for it in range(max_iter):
        # assign each box to the centroid with max IoU (min distance = 1 - IoU)
        distances = np.array([1 - wh_iou(box, centroids) for box in boxes])  # (n,k)
        nearest = np.argmin(distances, axis=1)  # (n,)

        new_centroids = []
        for i in range(k):
            assigned = boxes[nearest == i]
            if assigned.shape[0] == 0:
                # reinitialize this centroid randomly if empty
                new_centroids.append(centroids[i])
            else:
                if use_median:
                    new_centroids.append(np.median(assigned, axis=0))
                else:
                    new_centroids.append(np.mean(assigned, axis=0))
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids, nearest

def anchor_recall(boxes, anchors, iou_threshold=0.5):
    # fraction of boxes that have at least one anchor with IoU >= threshold
    n = boxes.shape[0]
    ok = 0
    for i in range(n):
        if np.max(wh_iou(boxes[i], anchors)) >= iou_threshold:
            ok += 1
    return ok / n

# -------------------------
# Helper to prepare boxes
# -------------------------
def prepare_boxes_from_yolo_df(df, target_size=640):
    """
    df: pandas DataFrame with columns [cx, cy, w_norm, h_norm] in YOLO normalized format (0..1)
    image_sizes: array-like Nx2 with [img_w, img_h] for each row (must match df rows)
    target_size: desired model input size (square, e.g., 640)
    returns: (n,2) boxes resized to target_size in PIXELS [w_px, h_px]
    """
    arr = df.to_numpy()
    w_norm = arr[:,2].astype(float)
    h_norm = arr[:,3].astype(float)
    #image_sizes = np.array(image_sizes, dtype=float)
    #img_w = image_sizes[:,0]
    #img_h = image_sizes[:,1]

    # # absolute pixels in source image
    # w_px = w_norm * img_w
    # h_px = h_norm * img_h
    #
    # # scale to target square input (maintain aspect scale for width/height separately)
    # scale_w = target_size / img_w
    # scale_h = target_size / img_h
    #
    # w_resized = w_px * scale_w
    # h_resized = h_px * scale_h

    w_resized = target_size * w_norm
    h_resized = target_size * h_norm


    boxes = np.stack([w_resized, h_resized], axis=1)
    # remove degenerate boxes (very small)
    boxes = boxes[(boxes[:,0] > 1) & (boxes[:,1] > 1)]
    return boxes



#image_sizes=np.array([256,256])

model_input_res=416
boxes = prepare_boxes_from_yolo_df(bboxes_df, target_size=model_input_res)

K = 9
anchors, assignment = iou_kmeans(boxes, k=K, max_iter=200, use_median=True)
# sort by area
areas = anchors[:,0] * anchors[:,1]
order = np.argsort(areas)
anchors_sorted = anchors[order]
print(f"Anchors (pixels) for input {model_input_res}:")
print(np.round(anchors_sorted).astype(int))
print("Anchors (normalized w,h):")
print(anchors_sorted / model_input_res)
print("Average IoU:", avg_iou(boxes, anchors_sorted))


centroids=anchors_sorted #/ resw

centroids_r=np.round(centroids).astype(int)


print("Average IoU:", avg_iou(boxes,centroids_r))

for thr in [0.5, 0.6, 0.7]:
    print(f"Recall@IoU{thr}:", anchor_recall(boxes, centroids_r, iou_threshold=thr))


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32)

print("Average IoU:", avg_iou(boxes,yolo_anchors))
for thr in [0.5, 0.6, 0.7]:
    print(f"Recall@IoU{thr}:", anchor_recall(boxes, yolo_anchors, iou_threshold=thr))

np.save(f"yolov3_tf2/anchor_{dataset_name}.npy", centroids_r)







#
# bboxes_array_wh=bboxes_df.to_numpy()[:,2:4]
#
# #
# # boxes_sizes=bboxes_array[:,2:4]-bboxes_array[:,:2]
#
# from prep import images_relationships
#
# iro = images_relationships.images_relationships_c()
# iro.cluster_quantity_test(bboxes_array_wh)
#
# clusters,centroids,wcss=iro.cluster_images_kmeans(bboxes_array_wh,9)
#
# area=np.prod(centroids,axis=-1)
#
# order=np.argsort(area,axis=0)
#
# centroids=centroids[order]

# np.save("anchor_african_animals.npy", centroids)





# zipped = zipped.map(lambda dic: {**dic,
#                                  'k_mean': tuple(tf.numpy_function(iro.cluster_images_kmeans,
#                                                                    [dic['img_histograms_correlation_normalized'],
#                                                                     num_clusters],
#                                                                    Tout=[tf.int32, tf.float32, tf.float64]))
#                                  }
#                     )



sample_sc = f'''

SELECT COUNT(*) FROM {dataset_name}_images
'''



sample_size =next(sql_d_o.query_generator(sample_sc)())[0]


def manual_split(dataset):
    dataset = dataset.map(lambda dic: {'img_loc': dic['img_loc'],
                                      'img_raw': dic['img_raw'],
                                      'height': dic['height'],
                                      'width': dic['width'],
                                      'channel': dic['channel'],
                                      'objects': dic['objects'],
                                      'img_histograms': dic['img_histograms'],
                                      'img_histograms_moments_normalized': dic['img_histograms_moments_normalized'],
                                      'img_histograms_correlation_normalized': dic['img_histograms_moments_normalized'],
                                      }
                         ).unbatch()

    dataset_train = dataset.skip(107).take(1012)

    dataset_traincv = dataset.skip(1119).take(230)

    dataset_cv_test = dataset.take(107).shuffle(buffer_size=107)

    dataset_cv =dataset_cv_test.take(54)

    dataset_test = dataset_cv_test.skip(54).take(53)

    return dataset_train, dataset_traincv, dataset_cv, dataset_test

def random_shuffle_split(dataset, batch_sz,perc=0.1,sh_size=1504,seed=42):


    test_size=tf.cast(sh_size*perc, tf.int64)
    cv_size=test_size
    traincv_size = test_size *2


    dataset = dataset.map(lambda dic: {'img_loc':dic['img_loc'],
                                     'img_raw':dic['img_raw'],
                                    'height': dic['height'],
                                       'width': dic['width'],
                                       'channel': dic['channel'],
                                     'objects':dic['objects'],
                                     'img_histograms':dic['img_histograms'],
                                     'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'],
                                     'img_histograms_correlation_normalized':dic['img_histograms_moments_normalized'],
                                     }
                ).unbatch()

    dataset=dataset.shuffle(sh_size,seed=seed)

    dataset_traincv=dataset.take(traincv_size)
    dataset_cv=dataset.skip(traincv_size).take(cv_size)
    dataset_test=dataset.skip(traincv_size+cv_size).take(test_size)
    dataset_train=dataset.skip(traincv_size+cv_size+test_size)

    return dataset_train, dataset_traincv, dataset_cv, dataset_test


# Slice a group
def slice_group1(a,start, end):

    return a[start:end]
    #
    # return {'img_loc':dic['img_loc'][start:end],
    #         'img': dic['img'][start:end],
    #         'objects':{k:v[start:end] for k,v in dic['objects'].items()},
    #         'img_histograms': dic['img_histograms'][start:end],
    #         'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'][start:end],
    #         'img_histograms_correlation_normalized':dic['img_histograms_correlation_normalized'][start:end]
    #         }


def clustered_split_pipeline1(dataset, batch_sz, perc=0.02, seed=42):

    # Step 1: compute closest cluster center indices



    dataset = dataset.map(lambda dic: {**dic,
                                     'order':tf.argsort(tf.norm(dic['img_histograms_correlation_normalized']-dic['k_mean'][1], axis=1))
                                     }
                )


    # Step 2: gather according to cluster order


    dataset = dataset.map(lambda dic: {'img_loc':tf.gather(dic['img_loc'],dic['order']),
                                     'img':tf.gather(dic['img'],dic['order']),
                                    'height': tf.gather(dic['height'], dic['order']),
                                       'width': tf.gather(dic['width'], dic['order']),
                                     'objects':{k:tf.gather(v,tf.reshape(dic['order'],[-1]),axis=0) for k,v in dic['objects'].items()},
                                     'img_histograms':tf.gather(dic['img_histograms'],dic['order']),
                                     'img_histograms_moments_normalized':tf.gather(dic['img_histograms_moments_normalized'],dic['order']),
                                     'img_histograms_correlation_normalized':tf.gather(dic['img_histograms_correlation_normalized'],dic['order']),
                                     }
                )




    # # Step 3: split into train_traincv and cv_test

    dataset_train_traincv =dataset.map(lambda dic:{**dic,
                                                   'start':tf.cast(tf.cast(tf.shape(dic['img'])[0],tf.float32) *2* perc,tf.int32),
                                                   'end':tf.shape(dic['img'])[0]
                                                   })



    dataset_train_traincv =dataset_train_traincv.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                                                 'height': slice_group1(dic['height'], dic['start'],
                                                                                     dic['end']),
                                                                 'width': slice_group1(dic['width'], dic['start'],
                                                                                     dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end']),
                                                   'traincv_size':dic['start']//2
                                                   })


    dataset_cv_test =dataset.map(lambda dic:{**dic,
                                                   'start':0,
                                                   'end':tf.cast(tf.cast(tf.shape(dic['img'])[0], tf.float32) *2* perc, tf.int32)
                                                   })


    dataset_cv_test =dataset_cv_test.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                                     'height': slice_group1(dic['height'], dic['start'],
                                                                            dic['end']),
                                                     'width': slice_group1(dic['width'], dic['start'],
                                                                           dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end'])
                                                   })


    #Step 4 : Shuffle train_traincv and cv_test

    dataset_train_traincv =dataset_train_traincv.map(lambda dic:{**dic,
                                                    'idx':tf.random.shuffle(tf.range(tf.shape(dic['img'])[0]),seed=seed)
                                                   })


    dataset_train_traincv =dataset_train_traincv.map(lambda dic:{'img_loc':tf.gather(dic['img_loc'],dic['idx']),
                                                   'img': tf.gather(dic['img'],dic['idx']),
                                                                 'height': tf.gather(dic['height'], dic['idx']),
                                                                 'width': tf.gather(dic['width'], dic['idx']),
                                                   'objects': {k: tf.gather(v,dic['idx']) for k, v in dic['objects'].items()},
                                                   'img_histograms': tf.gather(dic['img_histograms'],dic['idx']),
                                                   'img_histograms_moments_normalized':tf.gather(dic['img_histograms_moments_normalized'],dic['idx']),
                                                   'traincv_size': dic['traincv_size'],
                                                   'img_histograms_correlation_normalized':tf.gather(dic['img_histograms_correlation_normalized'],dic['idx'])
                                                   })



    dataset_cv_test =dataset_cv_test.map(lambda dic:{**dic,
                                                    'idx':tf.random.shuffle(tf.range(tf.shape(dic['img'])[0]),seed=seed)
                                                   })


    dataset_cv_test =dataset_cv_test.map(lambda dic:{'img_loc':tf.gather(dic['img_loc'],dic['idx']),
                                                   'img': tf.gather(dic['img'],dic['idx']),
                                                     'height': tf.gather(dic['height'], dic['idx']),
                                                     'width': tf.gather(dic['width'], dic['idx']),
                                                   'objects': {k: tf.gather(v,dic['idx']) for k, v in dic['objects'].items()},
                                                   'img_histograms': tf.gather(dic['img_histograms'],dic['idx']),
                                                   'img_histograms_moments_normalized':tf.gather(dic['img_histograms_moments_normalized'],dic['idx']),
                                                   'img_histograms_correlation_normalized':tf.gather(dic['img_histograms_correlation_normalized'],dic['idx'])
                                                   })







    # # Step 5: final internal splits

    dataset_train =dataset_train_traincv.map(lambda dic:{**dic,
                                                   'start': dic['traincv_size'],#tf.cast(tf.cast(tf.shape(dic['img'])[0],tf.float32) * perc,tf.int32)//2,
                                                   'end':tf.shape(dic['img'])[0]
                                                   })


    dataset_train =dataset_train.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                                 'height': slice_group1(dic['height'], dic['start'],
                                                                        dic['end']),
                                                 'width': slice_group1(dic['width'], dic['start'],
                                                                       dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end'])
                                                   })




    dataset_traincv =dataset_train_traincv.map(lambda dic:{**dic,
                                                   'start':0,
                                                   'end': dic['traincv_size']#tf.cast(tf.cast(tf.shape(dic['img'])[0],tf.float32) * perc,tf.int32)//2
                                                   })

    dataset_traincv =dataset_traincv.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                                     'height': slice_group1(dic['height'], dic['start'],
                                                                            dic['end']),
                                                     'width': slice_group1(dic['width'], dic['start'],
                                                                           dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end'])
                                                   })




    dataset_cv =dataset_cv_test.map(lambda dic:{**dic,
                                                   'start':tf.shape(dic['img'])[0]//2,#tf.cast(tf.cast(tf.shape(dic['img'])[0],tf.float32) * perc,tf.int32)//2,
                                                   'end':tf.shape(dic['img'])[0]
                                                   })
    dataset_cv =dataset_cv.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                           'height': slice_group1(dic['height'], dic['start'],
                                                                  dic['end']),
                                           'width': slice_group1(dic['width'], dic['start'],
                                                                 dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end'])
                                                   })



    dataset_test =dataset_cv_test.map(lambda dic:{**dic,
                                                   'start':0,
                                                   'end':tf.shape(dic['img'])[0]//2#tf.cast(tf.cast(tf.shape(dic['img'])[0],tf.float32) * perc,tf.int32)//2
                                                   })

    dataset_test =dataset_test.map(lambda dic:{'img_loc':slice_group1(dic['img_loc'],dic['start'],dic['end']),
                                                   'img': slice_group1(dic['img'],dic['start'],dic['end']),
                                               'height': slice_group1(dic['height'], dic['start'],
                                                                      dic['end']),
                                               'width': slice_group1(dic['width'], dic['start'],
                                                                     dic['end']),
                                                   'objects': {k: slice_group1(v,dic['start'],dic['end']) for k, v in dic['objects'].items()},
                                                   'img_histograms': slice_group1(dic['img_histograms'],dic['start'],dic['end']),
                                                   'img_histograms_moments_normalized':slice_group1(dic['img_histograms_moments_normalized'],dic['start'],dic['end']),
                                                   'img_histograms_correlation_normalized':slice_group1(dic['img_histograms_correlation_normalized'],dic['start'],dic['end'])
                                                   })


    return dataset_train.unbatch(), dataset_traincv.unbatch(), dataset_cv.unbatch(), dataset_test.unbatch()




perc = 0.05

# zipped_train, zipped_traincv, zipped_cv, zipped_test = clustered_split_pipeline1(zipped, batch_sz, perc)


zipped_train, zipped_traincv, zipped_cv, zipped_test = random_shuffle_split(zipped, batch_sz, perc,sample_size)


#zipped_train, zipped_traincv, zipped_cv, zipped_test = manual_split(zipped)


zipped_train = input_preparation.img_proc(zipped_train, stat2_m, batch_sz)

# zipped_train = zipped_train.map(lambda dic: {**dic,
#                                              'img_raw': tf.image.encode_png(dic['img']),
#                                              })
#
# zipped_traincv = zipped_traincv.map(lambda dic: {**dic,
#                                                            'img_raw': tf.image.encode_png(dic['img'])
#                                                            })
#
# zipped_cv = zipped_cv.map(lambda dic: {**dic,
#                                                  'img_raw': tf.image.encode_png(dic['img'])
#                                                  })
#
# zipped_test = zipped_test.map(lambda dic: {**dic,
#                                                      'img_raw': tf.image.encode_png(dic['img'])
#                                                      })


def build_example(dic):
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[dic['height']])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[dic['width']])),
        # 'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     annotation['filename'].encode('utf8')])),
        # 'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dic['img_raw'].numpy()])),
        # 'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(
            float_list=tf.train.FloatList(value=dic['objects']['x1'].flat_values.numpy().tolist())),
        'image/object/bbox/xmax': tf.train.Feature(
            float_list=tf.train.FloatList(value=dic['objects']['x2'].flat_values.numpy().tolist())),
        'image/object/bbox/ymin': tf.train.Feature(
            float_list=tf.train.FloatList(value=dic['objects']['y1'].flat_values.numpy().tolist())),
        'image/object/bbox/ymax': tf.train.Feature(
            float_list=tf.train.FloatList(value=dic['objects']['y2'].flat_values.numpy().tolist())),
        'image/object/class/text': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=dic['objects']['class_name'].flat_values.numpy().tolist())),
        'image/object/class/label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=dic['objects']['class_id'].flat_values.numpy().tolist())),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


# writer = tf.io.TFRecordWriter('test.tfrecord')
# # q=0


# class_map = {idx:name for idx, name in enumerate(
#     open('tmp/class_names.txt').read().splitlines())}
#
#
# keys_tensor = tf.constant(list(class_map.keys()), dtype=tf.int64)  # [0,1,2,3]
# vals_tensor = tf.constant(list(class_map.values()), dtype=tf.string)  # ['buffalo','elephant','rhino','zebra']
#
# initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
# id_to_name_table = tf.lookup.StaticHashTable(initializer, default_value="unknown")
#
#
# zipped_testq =zipped_test.map(lambda dic:{'img_loc':dic['img_loc'],
#                                                'img': dic['img'],
#                                            'img_raw':dic['img_raw'],
#                                                'objects': {'class_id':dic['objects']['class_id'],
#                                                            'class_name': id_to_name_table.lookup(tf.cast(dic['objects']['class_id'], tf.int64)),
#                                                            'x1': dic['objects']['x1'],
#                                                            'y1': dic['objects']['y1'],
#                                                            'x2': dic['objects']['x2'],
#                                                            'y2': dic['objects']['y2'],
#                                                            },
#                                                'img_histograms': dic['img_histograms'],
#                                                'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'],
#                                                'img_histograms_correlation_normalized':dic['img_histograms_correlation_normalized']
#                                                })
#
# for i in zipped_testq.take(2):
#     print(i)

options1 = tf.io.TFRecordOptions(compression_type="GZIP")


# q=0
# for i in zipped_train.take(1):
#     print(i)


# Write TFRecord
dnames = ['train','traincv', 'cv', 'test']
dsets = [zipped_train,zipped_traincv, zipped_cv, zipped_test]

for idd, dataset in enumerate(dsets):

    # if dnames[idd] != 'train':
    #     os.makedirs(f'testx/images/{dnames[idd]}', exist_ok=True)

    with tf.io.TFRecordWriter(f"data/{dataset_name}_{dnames[idd]}.tfrecord.gz", options=options1) as writer:
        for indx,dic in enumerate(dataset):
            # print(dic['objects']['y1'].flat_values)
            # print(dic['objects']['x1'])
            # print(tf.reshape(dic['objects']['x1'].to_tensor(),[-1]).numpy().tolist())
            # print(dic['objects']['x1'].flat_values.numpy().tolist())
            # print(i['img_raw'].numpy())
            #
            # print('tf',i['img_raw'])


            serialized_example = build_example(dic)
            writer.write(serialized_example)


            # if dnames[idd]!='train':
            #     tf.io.write_file(f"testx/images/{dnames[idd]}/example{indx}.jpg",dic['img_raw'])

    # for indx, dic in enumerate(dataset):
    #     if dnames[idd]!='train':
    #         tf.io.write_file(f"testx/images/{dnames[idd]}/example{indx}.jpg",dic['img_raw'])



        # writer.close()

    # q+=1
    # print(i['img_loc'].numpy().decode('utf-8'),i['objects']['x1'].to_tensor())

# for i in zipped_train.skip(10).take(5):
#     print(i[1]['object']['x1']).to_tensor()
#return zipped_train, zipped_traincv, zipped_cv, zipped_test











# print('we are starting to create tfrecord file.')
# import json
#
#
# def convert_md_to_json_serializable(md):
#     md_serializable = {}
#     for key, value in md.items():
#         if isinstance(value, tf.Tensor):
#             # Convert scalar tensor to native Python type
#             value = value.numpy()
#             if isinstance(value, bytes):
#                 value = value.decode('utf-8')  # For string tensors
#             elif hasattr(value, 'item'):
#                 value = value.item()  # For int64, float32 scalars
#         md_serializable[key] = value
#     return md_serializable
#
#
# def serialize_example(x, y, md, dataset_name):
#     md_serializable = convert_md_to_json_serializable(md)
#
#     feature = {
#         'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
#         'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
#         'md': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(md_serializable).encode()])),
#         # md is dictionary
#         'dataset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataset_name.encode()]))  # Store dataset name
#     }
#
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example.SerializeToString()
#
#
# options1 = tf.io.TFRecordOptions(compression_type="GZIP")
#
# os.makedirs('processed_datasets', exist_ok=True)
# # Write TFRecord
# with tf.io.TFRecordWriter(f"processed_datasets/{dataset_name}{res}_images_and_labels.tfrecord",
#                           options=options1) as writer:
#     # Process train dataset
#     for x, y, md in train_dataset_d:
#         serialized = serialize_example(x, y, md, "train")
#         writer.write(serialized)
#
#     # Process traincv dataset
#     for x, y, md in traincv_dataset_d:
#         serialized = serialize_example(x, y, md, "traincv")
#         writer.write(serialized)
#
#     # Process cv dataset
#     for x, y, md in cv_dataset_d:
#         serialized = serialize_example(x, y, md, "cv")
#         writer.write(serialized)
#
#     # Process test dataset
#     for x, y, md in test_dataset_d:
#         serialized = serialize_example(x, y, md, "test")
#         writer.write(serialized)