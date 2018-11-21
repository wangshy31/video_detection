import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
import sys
#sys.path.append('/ssd/wangshiyao/workspace/cvpr2019/video_detection/fgfa_rfcn/core/build/lib.linux-x86_64-2.7')
sys.path.append('/home/wangshiyao/Documents/workspace/VID/cvpr2019/video_detection/fgfa_rfcn/core/build/lib.linux-x86_64-2.7')
from coviar import load
from visualize_flow import visualize_flow
import scipy.misc
import math


# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_test_seg_image(roidb, cur_frame, end_frame, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_mv = []
    processed_residual = []
    processed_nearby_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]

        video_name = roi_rec['image'].split('/')
        prefix = '/'.join(video_name[0:5])+'/'+'-'.join(video_name[5:7])+'-'+str(int(video_name[-1].split('.')[0]))
        pos_target = end_frame - cur_frame
        mv, residual = read_mv_res(prefix, im.shape, im_scale, config.TRAIN.KEY_FRAME_INTERVAL, pos_target)

        if roidb[i]['flipped']:
            mv = mv[:, :, ::-1, :]
            residual = residual[:, :, ::-1, :]
        #read nearby roi_recs
        nearby_roidb = get_nearby_roi(roi_rec['image'], cur_frame, end_frame,
                                    roi_rec['frame_seg_len'], roi_rec['flipped'], im_info)
        for j in range(end_frame - cur_frame, config.TRAIN.KEY_FRAME_INTERVAL):
            nearby_roidb.append(nearby_roidb[-1])
        assert (len(nearby_roidb)-1) == mv.shape[0] == residual.shape[0], 'len(nearby_roidb) == mv.shape[0] == residual.shape[0]'

        processed_ims.append(im_tensor)
        processed_mv.append(mv)
        processed_residual.append(residual)
        processed_nearby_roidb.append(nearby_roidb)

    return processed_ims, processed_mv, processed_residual, processed_nearby_roidb


def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb

def get_triple_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    #processed_bef_ims = []
    #processed_aft_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        #if roi_rec.has_key('pattern'):
            # get two different frames from the interval [frame_id + MIN_OFFSET, frame_id + MAX_OFFSET]
            #offsets = np.random.choice(config.TRAIN.MAX_OFFSET - config.TRAIN.MIN_OFFSET + 1, 2, replace=False) + config.TRAIN.MIN_OFFSET
            #bef_id = min(max(roi_rec['frame_seg_id'] + offsets[0], 0), roi_rec['frame_seg_len']-1)
            #aft_id = min(max(roi_rec['frame_seg_id'] + offsets[1], 0), roi_rec['frame_seg_len']-1)
            #bef_image = roi_rec['pattern'] % bef_id
            #aft_image = roi_rec['pattern'] % aft_id

            #assert os.path.exists(bef_image), '%s does not exist'.format(bef_image)
            #assert os.path.exists(aft_image), '%s does not exist'.format(aft_image)
            #bef_im = cv2.imread(bef_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            #aft_im = cv2.imread(aft_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        #else:
            #bef_im = im.copy()
            #aft_im = im.copy()

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            #bef_im = bef_im[:, ::-1, :]
            #aft_im = aft_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        #bef_im, im_scale = resize(bef_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        #aft_im, im_scale = resize(aft_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        #bef_im_tensor = transform(bef_im, config.network.PIXEL_MEANS)
        #aft_im_tensor = transform(aft_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        #processed_bef_ims.append(bef_im_tensor)
        #processed_aft_ims.append(aft_im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    #return processed_ims, processed_bef_ims, processed_aft_ims, processed_roidb
    return processed_ims, processed_roidb


def load_vid_nearby_annotation(addr, cur_id, seg_len, flipped, im_info):
    """
    for a given index, load image and bounding boxes info from XML file
    :param index: index of a specific image
    :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """
    import xml.etree.ElementTree as ET
    roi_rec = dict()
    roi_rec['image'] = addr
    roi_rec['frame_id'] = 1
    roi_rec['pattern'] = '/'.join(addr.split('/')[5:8])+'/%06d'
    roi_rec['frame_seg_id'] = cur_id
    roi_rec['frame_seg_len'] = seg_len
    classes_map = ['__background__',  # always index 0
                   'n02691156', 'n02419796', 'n02131653', 'n02834778',
                   'n01503061', 'n02924116', 'n02958343', 'n02402425',
                   'n02084071', 'n02121808', 'n02503517', 'n02118333',
                   'n02510455', 'n02342885', 'n02374451', 'n02129165',
                   'n01674464', 'n02484322', 'n03790512', 'n02324045',
                   'n02509815', 'n02411705', 'n01726692', 'n02355227',
                   'n02129604', 'n04468005', 'n01662784', 'n04530566',
                   'n02062744', 'n02391049']

    filename = addr.replace('Data', 'Annotations').replace('.JPEG', '.xml')
    assert os.path.exists(filename), '%s does not exist'.format(filename)
    tree = ET.parse(filename)
    size = tree.find('size')
    roi_rec['height'] = float(size.find('height').text)
    roi_rec['width'] = float(size.find('width').text)

    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, len(classes_map)), dtype=np.float32)
    valid_objs = np.zeros((num_objs), dtype=np.bool)

    class_to_index = dict(zip(classes_map, range(len(classes_map))))
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = np.maximum(float(bbox.find('xmin').text), 0)
        y1 = np.maximum(float(bbox.find('ymin').text), 0)
        x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)
        y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)
        if not class_to_index.has_key(obj.find('name').text):
            continue
        valid_objs[ix] = True
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        if flipped:
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1

    boxes = boxes[valid_objs, :]
    gt_classes = gt_classes[valid_objs]
    overlaps = overlaps[valid_objs, :]

    assert (boxes[:, 2] >= boxes[:, 0]).all()
    roi_rec.update({'boxes': boxes.copy()*im_info[-1],
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'max_classes': overlaps.argmax(axis=1),
                    'max_overlaps': overlaps.max(axis=1),
                    'flipped': flipped,
                    'im_info': im_info})
    return roi_rec

def read_train_mv_res(prefix, im_shape, im_scale, num_interval, pos_target):
    mv_addr = prefix.replace('/VID/', '/MV/')+'.mv'
    res_addr = prefix.replace('/VID/', '/RES/')+'.res'
    #h = math.ceil(im_shape[0]*im_scale) if (im_shape[0]*im_scale)>int((im_shape[0]*im_scale))+0.5 else math.floor(im_shape[0]*im_scale)
    #w = math.ceil(im_shape[1]*im_scale) if (im_shape[1]*im_scale)>int((im_shape[1]*im_scale))+0.5 else math.floor(im_shape[1]*im_scale)
    h = im_shape[0]
    w = im_shape[1]
    for i in range(4):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    h, w = int(h), int(w)
    if pos_target == 0:
        mv = np.zeros((num_interval, 2, h, w), dtype=np.float16)
        res = np.zeros((num_interval, 3, h, w), dtype=np.float16)
        return mv, res
    mv = np.fromfile(mv_addr, dtype=np.float16)
    res = np.fromfile(res_addr, dtype=np.float16)
    assert mv.shape[0]%(2*h*w)==0, 'mv.shape[0]%(2*h*w)==0'
    assert res.shape[0]%(3*h*w)==0, 'res.shape[0]%(3*h*w)==0'
    mv = mv.reshape((num_interval, 2, h, w))
    res = res.reshape((num_interval, 3, h, w))
    return mv, res

def read_train_mv(prefix, im_info, num_interval, actual_num_interval):
    mv_addr = prefix.replace('/VID/', '/MV/')+'.mv'
    h = im_info[0]
    w = im_info[1]
    for i in range(4):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    h, w = int(h), int(w)
    if actual_num_interval == 0:
        mv = np.zeros((num_interval, 2, h, w), dtype=np.float16)
        return mv
    mv = np.fromfile(mv_addr, dtype=np.float16)
    assert mv.shape[0]%(2*h*w)==0, 'mv.shape[0]%(2*h*w)==0'
    mv = mv.reshape((-1, 2, h, w))[0:num_interval]
    return mv


def read_mv_res(prefix, im_shape, im_scale, num_interval, pos_target):
    mv_addr = prefix.replace('/VID/', '/MV/')+'.mv'
    res_addr = prefix.replace('/VID/', '/RES/')+'.res'
    #h = math.ceil(im_shape[0]*im_scale) if (im_shape[0]*im_scale)>int((im_shape[0]*im_scale))+0.5 else math.floor(im_shape[0]*im_scale)
    #w = math.ceil(im_shape[1]*im_scale) if (im_shape[1]*im_scale)>int((im_shape[1]*im_scale))+0.5 else math.floor(im_shape[1]*im_scale)
    h = im_shape[0]
    w = im_shape[1]
    for i in range(4):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    h, w = int(h), int(w)
    if pos_target == 0:
        mv = np.zeros((num_interval, 2, h, w), dtype=np.float16)
        res = np.zeros((num_interval, 3, h, w), dtype=np.float16)
        return mv, res
    mv = np.fromfile(mv_addr, dtype=np.float16)
    res = np.fromfile(res_addr, dtype=np.float16)
    assert mv.shape[0]%(2*h*w)==0, 'mv.shape[0]%(2*h*w)==0'
    assert res.shape[0]%(3*h*w)==0, 'res.shape[0]%(3*h*w)==0'
    if mv.shape[0] > num_interval*2*h*w:
        mv = mv[:num_interval*2*h*w]
        res = res[:num_interval*3*h*w]
    mv = mv.reshape((pos_target, 2, h, w))
    res = res.reshape((pos_target, 3, h, w))
    mv = np.pad(mv, ((0, num_interval - pos_target), (0,0), (0,0), (0,0)), 'constant')
    res = np.pad(res, ((0, num_interval - pos_target), (0,0), (0,0), (0,0)), 'constant')
    return mv, res
    #mv = np.fromfile()
def parse_mv(video_addr, gop_target, pos_target, im_scale):
    mv = load(video_addr, gop_target, pos_target, 1, False)
    shape = mv.shape
    h = math.ceil(shape[1]*im_scale) if (shape[1]*im_scale)>int((shape[1]*im_scale))+0.5 else math.floor(shape[1]*im_scale)
    w = math.ceil(shape[2]*im_scale) if (shape[2]*im_scale)>int((shape[2]*im_scale))+0.5 else math.floor(shape[2]*im_scale)
    for i in range(4):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    resize_mv = []
    for i in range(shape[0]):
        tmp_mv = cv2.resize(mv[i,:,:,:].astype(np.float32), None, None,
                   fx=w*1.0/shape[2],
                   fy=h*1.0/shape[1],
                   interpolation=cv2.INTER_LINEAR)
        tmp_mv = tmp_mv * (h*1.0/shape[1])
        resize_mv.append(tmp_mv.transpose(2, 0, 1))
    #for i in range(mv.shape[0]):
        #visualize_flow(mv[i].squeeze(), 'images/mv_'+str(i)+'.jpg')
    return resize_mv
def parse_residual(video_addr, gop_target, pos_target, im_scale):
    residual = load(video_addr, gop_target, pos_target, 2, False)
    shape = residual.shape
    h = math.ceil(shape[1]*im_scale) if (shape[1]*im_scale)>int((shape[1]*im_scale))+0.5 else math.floor(shape[1]*im_scale)
    w = math.ceil(shape[2]*im_scale) if (shape[2]*im_scale)>int((shape[2]*im_scale))+0.5 else math.floor(shape[2]*im_scale)
    for i in range(4):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    resize_residual = []
    for i in range(shape[0]):
        tmp_res = cv2.resize(residual[i,:,:,:].astype(np.float32), None, None,
                   fx=w*1.0/shape[2],
                   fy=h*1.0/shape[1],
                   interpolation=cv2.INTER_LINEAR)
        resize_residual.append(tmp_res.transpose(2, 0, 1))
    #for i in range(residual.shape[0]):
        #scipy.misc.imsave('images/res_'+str(i)+'.jpg', residual[i].squeeze())
    return resize_residual


def get_nearby_images(addr, begin_pos, end_pos, flipped, config):
    scale_ind = random.randrange(len(config.SCALES))
    target_size = config.SCALES[scale_ind][0]
    max_size = config.SCALES[scale_ind][1]
    cur_addr = '/'.join(addr.split('/')[:-1]) + '/%06d.JPEG'%begin_pos
    im = cv2.imread(cur_addr, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    if flipped:
        im = im[:, ::-1, :]
    im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
    cur_image = im_tensor.copy()
    #scipy.misc.imsave('images/org.jpg', cur_image[0].transpose(1,2,0))

    nearby_images = []
    h = im_info[0]
    w = im_info[1]
    for i in range(2):# num of pooling layers
        h = math.floor(0.5*(h - 1)) +1
        w = math.floor(0.5*(w - 1)) +1
    h, w = int(h), int(w)
    for i in range(begin_pos+1, end_pos+1):
        cur_addr = '/'.join(addr.split('/')[:-1]) + '/%06d.JPEG'%i
        im = cv2.imread(cur_addr, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        shape = im.shape
        im = cv2.resize(im, None, None, fx=w*1.0/shape[1], fy=h*1.0/shape[0], interpolation=cv2.INTER_LINEAR)
        if flipped:
            im = im[:, ::-1, :]
        im_tensor = transform_3d(im, config.network.PIXEL_MEANS)
        #scipy.misc.imsave('images/'+str(i)+'.jpg', im_tensor.transpose(1,2,0))
        nearby_images.append(im_tensor)
    return cur_image, nearby_images, im_info

def get_nearby_roi(addr, begin_pos, end_pos, seg_len, flipped, im_info):
    nearby_roi = []
    for i in range(begin_pos, end_pos+1):
        cur_addr = '/'.join(addr.split('/')[:-1]) + '/%06d.JPEG'%i
        nearby_roi.append(load_vid_nearby_annotation(cur_addr, i, seg_len, flipped, im_info))
    return nearby_roi

def get_seg_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_nearby_ims = []
    processed_mv = []
    processed_nearby_roidb = []
    num_interval = config.TRAIN.KEY_FRAME_INTERVAL
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        image_name = roi_rec['image'].split('/')
        prefix = '/'.join(image_name[0:5])+'/'+'-'.join(image_name[5:8])+'-'+str(int(image_name[-1].split('.')[0]))
        cur_pos = int(image_name[-1].split('.')[0])
        actual_num_interval = min(num_interval, roi_rec['frame_seg_len'] - cur_pos -1)
        cur_im, nearby_ims, im_info = get_nearby_images(roi_rec['image'], cur_pos, cur_pos+actual_num_interval,
                          roi_rec['flipped'], config)
        nearby_roidb = get_nearby_roi(roi_rec['image'], cur_pos, cur_pos+actual_num_interval,
                                    roi_rec['frame_seg_len'], roi_rec['flipped'], im_info)
        for j in range(actual_num_interval, num_interval):
            nearby_ims.append(nearby_ims[-1])
            nearby_roidb.append(nearby_roidb[-1])

        mv = read_train_mv(prefix, im_info, num_interval, actual_num_interval)

        if roidb[i]['flipped']:
            mv = mv[:, :, ::-1, :]
            mv[:, 0, :, :] = -mv[:, 0, :, :]
        #read nearby roi_recs
        #print len(nearby_roidb), mv.shape, residual.shape
        assert (len(nearby_roidb)-1) == mv.shape[0] == (len(nearby_ims)), 'len(nearby_roidb) == mv.shape[0] == len(ims)-1'

        processed_ims.append(cur_im)
        processed_nearby_ims.append(nearby_ims)
        processed_mv.append(mv)
        processed_nearby_roidb.append(nearby_roidb)

    return processed_ims, processed_nearby_ims, processed_mv, processed_nearby_roidb

def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    #print im.shape, im_scale, im.shape[0]*im_scale, im.shape[1]*im_scale
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    #print im.shape

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor
def transform_3d(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
