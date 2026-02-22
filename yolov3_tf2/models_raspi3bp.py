import numpy as np
import cv2
# yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                          (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416
def pick_anchors(name='default',res=416):
  if name=='construction_safety':
    npy=np.load("yolov3_tf2/anchor_construction_safety_objdet.npy")
    yolo_anchors=npy/res
    return yolo_anchors
  else:
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
    return yolo_anchors


yolo_anchors = pick_anchors(res=416)


yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


max_boxes=100
 
iou_thresh=0.5

score_thresh=0.5

sigma=0.5


# ------------------------
# utilities
# ------------------------
def _meshgrid_n(n_a, n_b):
    # matches the TF version used in your code:
    # returns [gx, gy] shaped (n_b, n_a)
    gx = np.tile(np.arange(n_a), (n_b, 1))        # shape (n_b, n_a)
    gy = np.reshape(np.repeat(np.arange(n_b), n_a), (n_b, n_a))
    return [gx, gy]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Reusable NMS (hard or gaussian soft-nms)
def _nms_indices(boxes_xyxy, scores, max_boxes=100, iou_threshold=0.45,
                 score_threshold=0.0, soft_nms_sigma=0.0):
    """
    boxes_xyxy: (N,4) [x1,y1,x2,y2] (floats)
    scores: (N,) floats
    Returns indices of kept boxes (sorted by final score desc)
    Implements:
      - if soft_nms_sigma == 0.0 -> standard hard NMS (suppress boxes with IoU > threshold)
      - else -> gaussian Soft-NMS (reduces scores of overlapping boxes)
    """
    
    # boxes_xyxy = boxes_xyxy.astype(np.float32)
    # scores = scores.astype(np.float32)
    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    inds = np.arange(boxes_xyxy.shape[0], dtype=int)
    scores = scores.copy().astype(float)

    keep = []
    kept_scores = []

    sigma = float(soft_nms_sigma)
    while inds.size > 0 and len(keep) < max_boxes:
        # pick highest score
        i = inds[np.argmax(scores[inds])]
        sc_i = scores[i]
        if sc_i < score_threshold:
            break
        keep.append(i)
        kept_scores.append(sc_i)

        if inds.size == 1:
            break

        # compute IoU with the rest
        rest = inds[inds != i]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]

        if sigma <= 0.0:
            # hard NMS: remove indices with IoU > threshold
            remaining_mask = iou <= iou_threshold
            inds = rest[remaining_mask]
        else:
            # gaussian Soft-NMS: decay the scores of rest by gaussian factor
            # scores[j] *= exp(-(iou^2) / (2 * sigma^2))
            #decay = np.exp(-(iou * iou) / (2 * sigma * sigma))
            decay = np.exp(-(iou * iou) / sigma)
            scores[rest] = scores[rest] * decay
            # filter out boxes that drop below score_threshold
            inds = rest[scores[rest] >= score_threshold]

    # sort keep by kept_scores descending
    if len(keep) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    kept_order = np.argsort(kept_scores)[::-1]
    keep_arr = np.array(keep, dtype=int)[kept_order]
    kept_scores_arr = np.array(kept_scores, dtype=float)[kept_order]
    return keep_arr, kept_scores_arr


# ------------------------
# yolo_boxes (numpy)
# ------------------------
def yolo_boxes_numpy(pred, anchors, num_classes):
    """
    pred : numpy array with shape (batch, grid_h, grid_w, anchors, (x,y,w,h,obj, ...classes))
           values are raw model outputs (floats).
    anchors: numpy array shape (num_anchors, 2) in same scale as network (width, height).
             Example: np.array([[10,13],[16,30],[33,23]], dtype=float)
    num_classes : int

    Returns: (bbox, objectness, class_probs, pred_box)
      bbox: (batch, grid_h, grid_w, anchors, 4) -> [x1,y1,x2,y2] normalized (0..1)
      objectness: (batch, grid_h, grid_w, anchors, 1)
      class_probs: (batch, grid_h, grid_w, anchors, num_classes)
      pred_box: (batch, grid_h, grid_w, anchors, 4) original xywh (center, w,h) normalized
    """
    pred = np.asarray(pred)
    batch = pred.shape[0]
    grid_h = pred.shape[1]
    grid_w = pred.shape[2]
    n_anchors = pred.shape[3]
    depth = pred.shape[4]
    assert depth == 4 + 1 + num_classes, "pred last dim must be 4(xywh) + 1(obj) + num_classes"

    # split
    box_xy = pred[..., 0:2]    # raw x,y
    box_wh = pred[..., 2:4]    # raw w,h
    objectness = pred[..., 4:5]
    class_probs = pred[..., 5:5+num_classes]

    # apply sigmoid where used in TF model
    box_xy = sigmoid(box_xy)
    objectness = sigmoid(objectness)
    class_probs = sigmoid(class_probs)
    pred_box = np.concatenate([box_xy, box_wh], axis=-1)

    # build grid and normalize
    gx, gy = _meshgrid_n(grid_w, grid_h)  # gx,gy shapes (grid_h, grid_w)
    grid = np.stack([gx, gy], axis=-1)            # (grid_h, grid_w, 2)
    grid = np.expand_dims(grid, axis=2)           # (grid_h, grid_w, 1, 2)
    grid = np.expand_dims(grid, axis=0)           # (1, grid_h, grid_w, 1, 2) to broadcast

    # anchors -> shape (1,1,1,n_anchors,2)
    #anchors = np.asarray(anchors, dtype=float)
    anchors = anchors.reshape((1, 1, 1, anchors.shape[0], 2))

    # convert to normalized coordinates
    # box_xy relative to grid cell: (sigmoid + grid) / [grid_w, grid_h]
    grid_size = np.array([grid_h, grid_w], dtype=float)
    box_xy_abs = (box_xy + grid) / grid_size.reshape((1, 1, 1, 1, 2))
    # box_wh predicted as log-space: exp(raw) * anchors ; anchors are in pixels -> normalize by input size
    box_wh_abs = np.exp(box_wh) * anchors
    # Normalize wh by (input width, input height) assuming anchors are given in same pixel scale as network input.
    # If anchors are already normalized to input size, skip dividing.
    # We'll normalize by (grid_w, grid_h) respectively *BUT* anchors are usually defined in pixels relative to INPUT_SIZE.
    # To keep behavior identical to TF training setup, user should pass anchors normalized to the input size.
    # If anchors are in pixels, you must divide by [input_w, input_h] before calling this function.
    # (common approach: anchors already scaled for model input).
    # Here we assume anchors already normalized -> box_wh_abs is normalized.
    # Now compute x1,y1,x2,y2
    box_x1y1 = box_xy_abs - box_wh_abs / 2.0
    box_x2y2 = box_xy_abs + box_wh_abs / 2.0
    bbox = np.concatenate([box_x1y1, box_x2y2], axis=-1)  # (batch, gh, gw, anchors, 4)
    
    bbox = np.clip(bbox, 0.0, 1.0)

    return bbox, objectness, class_probs, pred_box

# ------------------------
# yolo_nms (numpy)
# ------------------------
def yolo_nms_numpy(outputs, classes,score_thresh=0.5,sigma=0.5):
	
    """
    outputs: list of tuples as returned by yolo_boxes_numpy (bbox, objectness, class_probs, pred_box)
    classes: number of classes (int)
    Returns: boxes (1, max_boxes, 4), scores (1, max_boxes), classes (1, max_boxes), valid_detections (1,)
    Notes:
      - Assumes batch size 1 (like your TFLite inference). If batch>1, you can adapt similarly per batch.
      - anchors and masks are not needed here since boxes are already converted to absolute/normalized coords.
    """
    # concatenate per-output tensors into one big list
    b_list, c_list, t_list = [], [], []
    
    an_n=[]
    

    for i,o in enumerate(outputs):
        # each o is (bbox, objectness, class_probs, pred_box)
        bbox = o[0]
        obj = o[1]
        cls = o[2]
        
        
        obj_shape = o[1].shape
        
        anchor_ids = np.arange(3, dtype=np.int32) + (3 * i)#np.reshape(np.arange(3, dtype=np.int32) + (3 * i), (1, 1, 1, 3, 1))
        anchor_ids =anchor_ids.reshape(1, 1, 1, 3, 1)
        
        anchor_num = np.zeros(obj_shape, dtype=np.int32)+anchor_ids


        # reshape to (batch, -1, ...)
        b_list.append(bbox.reshape(bbox.shape[0], -1, 4))
        c_list.append(obj.reshape(obj.shape[0], -1, 1))
        t_list.append(cls.reshape(cls.shape[0], -1, classes))
        
        
        an_n.append(anchor_num.reshape(anchor_num.shape[0],-1,anchor_num.shape[-1]))
        
        #an.append(tf.reshape(anchor_num, (tf.shape(anchor_num)[0], -1, tf.shape(anchor_num)[-1])))
        

    bbox = np.concatenate(b_list, axis=1)        # (batch, N, 4)
    confidence = np.concatenate(c_list, axis=1)  # (batch, N, 1)
    class_probs = np.concatenate(t_list, axis=1) # (batch, N, classes)
    
    anchors_nums = np.concatenate(an_n, axis=1) # (batch, N, classes)
    
    
    
    # import pdb
    # pdb.set_trace()

    # For typical TFLite usage you have batch=1
    batch = bbox.shape[0]
    assert batch == 1, "This helper assumes batch size 1. For batch>1 adapt per-batch."

    if classes == 1:
        scores_all = confidence[..., 0]  # shape (1, N)
    else:
        scores_all = (confidence[..., 0:1] * class_probs).reshape(batch, -1, classes)  # (1, N, classes)


    # squeeze batch dimension
    if classes == 1:
        #dscores = scores_all[0]      # (N,)
        dscores=scores_all
        scores_for_nms = dscores
        class_for_each = np.zeros_like(dscores, dtype=int)
    else:
        #dscores = scores_all[0]      # (N, classes)
        dscores=scores_all
        scores_for_nms = np.max(dscores, axis=-1)  # (N,) 1
        class_for_each = np.argmax(dscores, axis=-1)  # (N,)1

    # flatten bboxes and convert to (N,4)
    #bboxes_flat = bbox.reshape(-1, 4)  # (N,4)
    

    # Filter by score_threshold first
    keep_mask = scores_for_nms >= score_thresh
    
    anchors_nums=np.squeeze(anchors_nums,axis=-1)
    
    d=[]
    
    boxes_out = np.zeros((batch,max_boxes,4),dtype=np.float32)
    scores_out = np.zeros((batch,max_boxes), dtype=np.float32)
    classes_out = np.zeros((batch,max_boxes), dtype=np.int32)
    anchors_nums_out = np.zeros((batch,max_boxes), dtype=np.int32)
    
    
    bb,bs,bc,bans,bkms=bbox,scores_for_nms,class_for_each,anchors_nums,keep_mask
    
    for i in range(batch):
        
        bboxes_flat,scores_for_nms,class_for_each,anchors_nums,keep_mask=bb[i],bs[i],bc[i],bans[i],bkms[i]

        if not np.any(keep_mask):
	        # no detections, return zero-padded arrays
            boxes_out = np.zeros((1, max_boxes, 4), dtype=np.float32)
            scores_out = np.zeros((1, max_boxes), dtype=np.float32)
            classes_out = np.zeros((1, max_boxes), dtype=np.int32)
            valid_det = np.array([0], dtype=np.int32)

        filtered_boxes = bboxes_flat[keep_mask]
        filtered_scores = scores_for_nms[keep_mask]
        filtered_classes = class_for_each[keep_mask]
        filtered_anchors_nums=anchors_nums[keep_mask]#batch=1
		
		
		
        # run NMS (soft or hard)
        keep_inds, kept_scores = _nms_indices(filtered_boxes, filtered_scores,
											 max_boxes=max_boxes,
											 iou_threshold=iou_thresh,
											 score_threshold=score_thresh,
											 soft_nms_sigma=sigma)

		# gather results and pad
        kept_boxes = filtered_boxes[keep_inds] if keep_inds.size > 0 else np.zeros((0,4))
        kept_score_vals = kept_scores if kept_scores.size > 0 else np.zeros((0,))
        kept_classes = filtered_classes[keep_inds] if keep_inds.size > 0 else np.zeros((0,), dtype=int)
        valid_count = int(kept_boxes.shape[0])
		
		
        kept_anchors_nums = filtered_anchors_nums[keep_inds] if keep_inds.size > 0 else np.zeros((0,))
		
		

		# pad to max_boxes
        pad = max_boxes - valid_count
        if pad < 0:
            kept_boxes = kept_boxes[:max_boxes]
            kept_score_vals = kept_score_vals[:max_boxes]
            kept_classes = kept_classes[:max_boxes]
            valid_count = max_boxes
			
            kept_anchors_nums=kept_anchors_nums[:max_boxes]
            pad = 0
        
        
        if valid_count > 0:
            boxes_out[i,:valid_count, :] = kept_boxes
            scores_out[i,:valid_count] = kept_score_vals
            classes_out[i,:valid_count] = kept_classes
            anchors_nums_out[i,:valid_count]=kept_anchors_nums
        
        d.append(valid_count)
    
    #import pdb
    #pdb.set_trace()


    return boxes_out, scores_out, classes_out,d,anchors_nums_out

  # # -----------------------------
    # # Draw outputs
    # # -----------------------------
def draw_outputs(img, boxes, scores, classes, nums, class_names, class_colors, anch_nums,SCORE_THRESHOLD):
  img_h, img_w = img.shape[:2]
  ref = min(img_h, img_w)

  thickness_box  = max(1, int(ref / 200))
  thickness_text = max(1, int(ref / 400))
  font_scale     = max(0.4, ref / 600)

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

        brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]

        if brightness >150:
          color= (0,0,0)
        else:
          color= (255,255,255)
          y1=max(10,y1-5)

          cv2.putText(img, f"{label} {score:.2f}", (x1,y1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255 - color[0], 255 - color[1], 255 - color[2]),
                      thickness_text,cv2.LINE_AA)  # (0, 0, 255)

        #2 0.5 2
  return img


