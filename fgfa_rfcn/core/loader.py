# --------------------------------------------------------
# Flow-Guided Feature Aggregation

# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------

import numpy as np
import mxnet as mx
import cv2
from mxnet.executor_manager import _split_input_slice

from config.config import config
from utils.image import tensor_vstack
from rpn.rpn import get_rpn_testbatch, get_rpn_seg_batch, assign_anchor
from rcnn import get_rcnn_testbatch, get_rcnn_batch
import collections
import math
import sys

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = np.sum([x['frame_seg_len'] for x in self.roidb])
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'mv', 'residual', 'im_info']#, 'data_cache', 'feat_cache']
        self.label_name = None

        #
        self.cur_roidb_index = 0
        self.cur_frameid = 0
        self.cur_seg_len = 0
        self.key_frame_flag = -1
        self.cur_interval = 1
        self.cur_index = 0

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_init_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.cur_interval
            self.cur_frameid += self.cur_interval
            self.cur_index += 1
            if self.cur_frameid >= self.cur_seg_len:
                self.cur_roidb_index += 1
                self.cur_frameid = 0
                #self.key_frame_flag = 1
            return self.im_info, self.key_frame_flag, self.cur_interval, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur_index
        #return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        end_frame = cur_roidb['key_frames'][self.cur_frameid]
        self.cur_seg_len = cur_roidb['frame_seg_len']
        data, label, im_info = get_rpn_testbatch([cur_roidb], self.cur_frameid, end_frame, self.cfg)
        if self.cur_frameid == 0: # new video
                self.key_frame_flag = 0
        else:       # normal frame
            self.key_frame_flag = 2

        extend_data = [{'data': data[0]['data'] ,
                        'mv': data[0]['mv'],
                        'residual': data[0]['residual'],
                        'im_info': data[0]['im_info']}]
        self.data = [[mx.nd.array(extend_data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info
        self.cur_interval = end_frame-self.cur_frameid+1

    def get_init_batch(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        end_frame = cur_roidb['key_frames'][int(self.cur_frameid)]
        self.cur_seg_len = cur_roidb['frame_seg_len']
        data, label, im_info = get_rpn_testbatch([cur_roidb], self.cur_frameid, end_frame, self.cfg)
        if self.cur_frameid == 0: # new frame
                self.key_frame_flag = 0
        else:       # normal frame
            self.key_frame_flag = 2

        extend_data = [{'data': data[0]['data'],
                        'mv': data[0]['mv'],
                        'residual': data[0]['residual'],
                        'im_info': data[0]['im_info'],
                        'num_interval': end_frame-self.cur_frameid+1}]
        self.data = [[mx.nd.array(extend_data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info
        self.cur_interval = end_frame-self.cur_frameid+1

class AnchorLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, normalize_target=False, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.4, 0.4)):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.normalize_target = normalize_target
        self.bbox_mean = bbox_mean
        self.bbox_std = bbox_std

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data','im_info', 'gt_boxes', 'mv', 'nearby_data']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []

        if 'mv' not in max_data_shape:
            h = max_data_shape[0][1][2]
            w = max_data_shape[0][1][3]
            for i in range(4):# num of pooling layers
                h = math.floor(0.5*(h - 1)) +1
                w = math.floor(0.5*(w - 1)) +1
            max_data_shape.append(('mv', (self.cfg.TRAIN.KEY_FRAME_INTERVAL, 2, int(h), int(w))))

        if 'nearby_data' not in max_data_shape:
            h = max_data_shape[0][1][2]
            w = max_data_shape[0][1][3]
            for i in range(2):# num of pooling layers
                h = math.floor(0.5*(h - 1)) +1
                w = math.floor(0.5*(w - 1)) +1
            max_data_shape.append(('nearby_data', (self.cfg.TRAIN.KEY_FRAME_INTERVAL, 3, int(h), int(w))))

        #if 'residual' not in max_data_shape:
            #h = max_data_shape[0][1][2]
            #w = max_data_shape[0][1][3]
            #for i in range(4):# num of pooling layers
                #h = math.floor(0.5*(h - 1)) +1
                #w = math.floor(0.5*(w - 1)) +1
            #max_data_shape.append(('residual', (self.cfg.TRAIN.KEY_FRAME_INTERVAL, 3, int(h), int(w))))


        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([self.cfg.TRAIN.KEY_FRAME_INTERVAL+1] +
                                 list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_triple_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]

        new_label_list = []
        for data, label in zip(data_list, label_list):
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

            # assign anchor for label
            label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border,
                                  self.normalize_target, self.bbox_mean, self.bbox_std)
            new_label_list.append(nearby_label)
        #print 'new_label_list!!!', new_label_list

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu
        #data, label = get_rpn_triple_batch(iroidb, self.cfg)
        data, label = get_rpn_seg_batch(iroidb, self.cfg)

        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes']


        # assign anchor for label
        #label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              #self.feat_stride, self.anchor_scales,
                              #self.anchor_ratios, self.allowed_border,
                              #self.normalize_target, self.bbox_mean, self.bbox_std)

        nearby_label = collections.defaultdict(list)
        for i in range(label['gt_boxes'].shape[0]):
            tmp_label = assign_anchor(feat_shape, label['gt_boxes'][i], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
            if i == 0:
                for item in tmp_label.keys():
                    nearby_label[item].append(tmp_label[item][0])
            else:
                for item in tmp_label.keys():
                    nearby_label[item] = np.vstack((np.array(nearby_label[item]), np.array(tmp_label[item])))

        return {'data': data, 'label': nearby_label}

