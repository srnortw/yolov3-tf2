import tensorflow as tf

from prep import get_all_image_samples
from prep import images_properties
from prep import filtering_images
from prep import images_relationships

import matplotlib.pyplot as plt
import matplotlib


def shp(f, shape):
    f.set_shape(shape)
    return f

def z_score_norm(x):
    u = tf.reduce_mean(x, axis=0)

    s = tf.math.reduce_std(x, axis=0)

    imgs_histograms_moments = (x - u) / s

    return imgs_histograms_moments, u, s

def z_score_norm_trans(x, y):
    u, s = y
    return (x - u) / s


# Shuffle all tensors consistently
def shuffle_all(a, b, c, d, seed=42):
    idx = tf.random.shuffle(tf.range(tf.shape(a)[0]), seed=seed)
    return (
        tf.gather(a, idx),
        {obj: {k: tf.gather(v, idx) for k,v in clss_bbox.items()} for obj, clss_bbox in b.items()},
        tf.gather(c, idx),
        tf.gather(d, idx),
    )


# Slice a group
def slice_group(a, b, c, d, start, end):
    return (
        a[start:end],
        {obj: {k:v[start:end] for k,v in clss_bbox.items()} for obj, clss_bbox in b.items()},
        c[start:end],
        d[start:end]
    )

# def update_dic(dic, **updates):
#     """Return a new dictionary with updated keys."""
#     new_dic = dict(dic)
#     new_dic.update(updates)
#     return new_dic


def img_proc(zipped,stat2_m,batch_sz):

    # zipped = zipped.apply(tf.data.experimental.assert_cardinality(q))

    fio = filtering_images.filtering_images_c()

    # zipped = zipped.map(lambda a, b, c: (a,
    #                                         b,
    #                                         c,
    #                                         tf.numpy_function(fio.ImagePreprocessor, [a, c, batch_sz],
    #                                                               # zipped.cardinality()
    #                                                               Tout=tf.uint8)  # ,[tf.uint8,tf.uint8]
    #                                         )
    #                     )  # ,num_parallel_calls=tf.data.AUTOTUNE)


    # zipped =zipped.map(lambda dic:{'img_loc':dic['img_loc'],
    #                                                'img': dic['img'],
    #                                                'objects': dic['objects'],
    #                                                'img_histograms': dic['img_histograms'],
    #                                                'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'],
    #                                                'img_histograms_correlation_normalized':dic['img_histograms_correlation_normalized'],
    #                                'new_img':tf.numpy_function(fio.ImagePreprocessor, [dic['img'], dic['img_histograms_moments_normalized'], batch_sz],Tout=tf.uint8)
    #                                                })
    #

    # zipped =zipped.map(lambda dic:{**dic,
    #                                'new_img_raw':tf.image.encode_png(
    #                                    tf.numpy_function(fio.ImagePreprocessor,
    #                                                      [tf.image.decode_jpeg(dic['img_raw'],channels=3), dic['img_histograms_moments_normalized'], batch_sz],
    #                                                      Tout=tf.uint8)
    #                                    )})
    #
    #
    #
    #
    # for dict in zipped.take(3):
    #     print(dict['objects'],dict['img_histograms_moments_normalized'])
    #
    #     new_img=tf.image.decode_jpeg(dict['new_img_raw'],channels=3)
    #     img = tf.image.decode_jpeg(dict['img_raw'], channels=3)
    #
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(new_img)#plt.imshow(dict['new_img'])
    #     d_input_shape = img.shape#dict['img'].shape
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(img)#plt.imshow(dict['img'])
    #     plt.show()
    #     # plt.show(block=True)

    # import matplotlib.pyplot as plt
    #
    # matplotlib.use("TkAgg")

    # for org, o, hm, image in zipped.skip(25).take(10):
    #
    #     print(o,hm)
    #
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(image)
    #     d_input_shape = image.shape
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(org)
    #     plt.show()
    #     # plt.show(block=True)

    # for i in zipped.map(lambda a,b,c,d:d).take(2):
    #     print(i.shape)
    #


    #
    # zipped = zipped.map(lambda a, b, c, d: (d,
    #                                            b,
    #                                            c
    #                                            )
    #                     )


    # zipped =zipped.map(lambda dic:{'img_loc':dic['img_loc'],
    #                                                'img': dic['new_img'],
    #                                                'objects': dic['objects'],
    #                                                'img_histograms': dic['img_histograms'],
    #                                                'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'],
    #                                                'img_histograms_correlation_normalized':dic['img_histograms_correlation_normalized']
    #                                                })

    zipped =zipped.map(lambda dic:{**dic,
                                   'img_raw': dic['img_raw']#'img': dic['img']
                                   })

    return zipped
    #
    # zipped = zipped.map(lambda a, b, c, d: (a,
    #                                         b,
    #                                         c,
    #                                         tf.numpy_function(ipo.compute_histogram, [a], Tout=tf.float32)
    #                                         )
    #                     , num_parallel_calls=tf.data.AUTOTUNE)
    #
    # zipped = zipped.map(lambda a, b, c, d: (a,
    #                                         b,
    #                                         c,
    #                                         d,
    #                                         ipo.compute_histograms_moments(d)
    #                                         )
    #                     , num_parallel_calls=tf.data.AUTOTUNE).batch(batch_sz)  # all_metad_df.shape[0]
    #
    # # def z_score_norm_trans(x, y):
    # #     u, s = y
    # #
    # #     return (x - u) / s
    #
    # zipped = tf.data.Dataset.zip((zipped, stat2_m)).map(lambda a, b: (a[0],
    #                                                                   a[1],
    #                                                                   a[2],
    #                                                                   a[3],
    #                                                                   z_score_norm_trans(a[4], b)
    #                                                                   )
    #                                                     )  # .unbatch()
def inp_prep_f(zipped,batch_sz):

    #
    # zipped = zipped.shuffle(zipped.cardinality(), seed=42)
    #

    num_bins = 256
    channelq = 3
    ipo = images_properties.images_properties_c(channelq, num_bins)


    zipped = zipped.map(lambda dic:{**dic,'img_histograms':tf.numpy_function(ipo.compute_histogram, [tf.image.decode_jpeg(dic['img_raw'],channels=3)], Tout=tf.float32)},
                        num_parallel_calls=tf.data.AUTOTUNE)


    zipped = zipped.map(lambda dic:{**dic,'img_histograms_moments':ipo.compute_histograms_moments(dic['img_histograms'])},
                        num_parallel_calls=tf.data.AUTOTUNE).batch(batch_sz)


    zipped = zipped.map(lambda dic:{**dic,
                                    'img_histograms': tf.reshape(dic['img_histograms'], [tf.shape(dic['img_histograms'])[0], -1]),
                                    'img_histograms_moments_normalized':z_score_norm(dic['img_histograms_moments'])})





    # stat2_m = zipped.map(lambda a, b, c,d: (d[1], d[2]))

    stat2_m = zipped.map(lambda dic: {
                                     'img_histograms_moments_normalized':
                                         {'mean':dic['img_histograms_moments_normalized'][1],'std':dic['img_histograms_moments_normalized'][2]}
                                     }
                )

    zipped = zipped.map(lambda dic: {**dic,
                                     'img_histograms_moments_normalized':dic['img_histograms_moments_normalized'][0]
                                     }

                )

    iro = images_relationships.images_relationships_c()


    zipped = zipped.map(lambda dic: {**dic,
                                     'img_histograms_correlation':tf.cast(tf.numpy_function(iro.compute_histogram_correlation, [dic['img_histograms']],Tout=tf.float64),tf.float32)
                                     }
                )



    zipped = zipped.map(lambda dic: {**dic,
                                     'img_histograms_correlation_normalized':tf.cast(z_score_norm(dic['img_histograms_correlation'])[0], tf.float32)
                                     }
                )

    # iro.cluster_quantity_test(correlation_matrix)

    num_clusters = tf.constant(1, dtype=tf.int32)


    zipped = zipped.map(lambda dic: {**dic,
                                     'k_mean':tuple(tf.numpy_function(iro.cluster_images_kmeans,
                                                       [dic['img_histograms_correlation_normalized'], num_clusters],
                                                       Tout=[tf.int32, tf.float32, tf.float64]))
                                     }
                )



    zipped_pca = zipped.map(lambda dic: tf.numpy_function(iro.pca,
                                                          [dic['img_histograms_correlation_normalized'],dic['k_mean'][1], dic['k_mean'][2]],
                                                          Tout=[tf.float32,tf.float32,tf.float64]))


    # for i in zipped_pca.take(1):
    #     print(i[0])

    print('pca')
    t = ['Image Samples Correlations', f'Its {num_clusters} Clusters Centroids']

    import pandas as pd

    # pca_i=0

    for datas,centroid,wcss in zipped_pca:  # .map(lambda a,b,c,d,e,f,g,h,i,j:a) :

        # pc_df1 = pd.DataFrame(data=datas, columns=['PC1', 'PC2'])
        #
        # pc_df2 = pd.DataFrame(data=centroid, columns=['PC1', 'PC2'])

        fig, ax = plt.subplots(figsize=(10, 8))  # plt.figure(figsize=(10, 8))

        ax.scatter(datas[:,0], datas[:,1], c='blue', label=f'{t[0]}')  # s

        ax.scatter(centroid[:,0], centroid[:,1], c='red', label=f'{t[1]}')  # s

        ax.set_title(f'2D PCA of Images Correlations also Wcss is {wcss}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        # plt.savefig(f"plot{pca_i}.png")
        # pca_i+=1
        plt.show()

        print('2D pca has created')  # samq=i[0].shape[0]

    return zipped,stat2_m