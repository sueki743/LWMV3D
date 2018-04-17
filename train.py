import tensorflow as tf
import numpy as np
import cv2
import threading
from time import time, sleep, strftime
import glob
import os
import ctypes
from numba import jit
from easydict import EasyDict
import argparse

from blocks import *
from lib.nms.cpu_nms import cpu_nms as nms
from lib.roi_pooling_layer.roi_pooling_op import roi_pool as tf_roipooling
from lib.utils.bbox import bbox_overlaps


DIR_TRAIN = {
    'lidar': '/home/katou01/MV3D/data/raw/kitti/data_object/train/velodyne_points/data',
    'rgb'  : '/home/katou01/MV3D/data/raw/kitti/data_object/train/image_02/data',
    'calib': '/home/katou01/MV3D/data/raw/kitti/data_object/train/calib',
    'label': '/home/katou01/MV3D/data/raw/kitti/data_object/train/label_2',
}
DIR_VAL = {
    'lidar': '/home/katou01/MV3D/data/raw/kitti/data_object/val/velodyne_points/data',
    'rgb'  : '/home/katou01/MV3D/data/raw/kitti/data_object/val/image_02/data',
    'calib': '/home/katou01/MV3D/data/raw/kitti/data_object/val/calib',
    'label': '/home/katou01/MV3D/data/raw/kitti/data_object/val/label_2',
}
SUFFIX = {
    'lidar': '.bin',
    'rgb'  : '.png',
    'calib': '.txt',
    'label': '.txt',
}
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6
TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.3
SharedLib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'lib/top/LidarTopPreprocess.so'))


class Calib:
    def __init__(self, path):
        with open(path) as file:
            for line in file:
                if line.startswith('P2'):
                    values = line.split()[1:]
                    assert len(values) == 12
                    P2 = np.array(values, dtype=float).reshape((3, 4))
                    self.R_P2 = P2[:, 0:3]
                    self.T_P2 = P2[:, 3]
                elif line.startswith('R0_rect'):
                    values = line.split()[1:]
                    assert len(values) == 9
                    self.R0_rect = np.array(values, dtype=float).reshape((3, 3))
                elif line.startswith('Tr_velo_to_cam'):
                    values = line.split()[1:]
                    assert len(values) == 12
                    Tr_velo_to_cam = np.array(values, dtype=float).reshape((3, 4))
                    self.R_velo_to_cam = Tr_velo_to_cam[:, 0:3]
                    self.T_velo_to_cam = Tr_velo_to_cam[:, 3]

    def cam_to_velo(self, cam):
        return self.R_velo_to_cam.T @ (self.R0_rect.T @ cam - self.T_velo_to_cam)

    def velo_to_cam(self, velo):
        # return self.R0_rect @ (self.R_velo_to_cam @ velo + self.T_velo_to_cam)
        return (velo @ self.R_velo_to_cam.T + self.T_velo_to_cam) @ self.R0_rect.T

    def cam_to_im(self, cam):
        homo = cam @ self.R_P2.T + self.T_P2
        return homo[..., :2] / homo[..., [2]]

    def velo_to_im(self, velo):
        return self.cam_to_im(self.velo_to_cam(velo))


class Label:
    def __init__(self, line=None):
        if line is not None:
            values = line.split()
            assert len(values) == 15
            self.type = values[0]
            self.truncated = float(values[1])
            # self.occluded = int(values[2])
            # self.alpha = float(values[3])
            # self.bbox = np.array(values[4:8], dtype=float)
            self.dimensions = np.array(values[8:11], dtype=float)
            self.location = np.array(values[11:14], dtype=float)
            self.rotation_y = float(values[14])

    def write(self, file):
        file.write('{} '.format(self.type))
        file.write('-1 ')
        file.write('-1 ')
        file.write('-10 ')
        for f in self.bbox:
            file.write('{:.16f} '.format(f))
        for f in self.dimensions:
            file.write('{:.16f} '.format(f))
        for f in self.location:
            file.write('{:.16f} '.format(f))
        file.write('{:.16f} '.format(self.rotation_y))
        file.write('{:.16f}\n'.format(self.score))


def read_label(path, calib):
    objects = []
    with open(path) as file:
        for line in file:
            label = Label(line)
            if not 0. <= label.truncated < 1. or label.type == 'DontCare':
                continue
            o = type('', (), {})()
            o.type = label.type
            o.translation = calib.cam_to_velo(label.location)
            o.rotation = np.array([0., 0., - np.pi/2 - label.rotation_y])
            o.size = label.dimensions
            objects.append(o)
    return objects


def clidar_to_top(lidar):
    Xn = int((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION)
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION)
    Zn = int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)

    top_flip = np.ones((Xn, Yn, Zn + 2), dtype=np.float32)

    num = lidar.shape[0]

    SharedLib.createTopMaps(ctypes.c_void_p(lidar.ctypes.data),
                            ctypes.c_int(num),
                            ctypes.c_void_p(top_flip.ctypes.data),
                            ctypes.c_float(TOP_X_MIN), ctypes.c_float(TOP_X_MAX),
                            ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Y_MAX),
                            ctypes.c_float(TOP_Z_MIN), ctypes.c_float(TOP_Z_MAX),
                            ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION),
                            ctypes.c_float(TOP_Z_DIVISION),
                            ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn)
                            )
    top = np.flipud(np.fliplr(top_flip))
    return top


@jit
def box3d_compose(obj):
    h, w, l = obj.size
    trackletBox = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0.0, 0.0, 0.0, 0.0, h, h, h, h]
    ])

    yaw = obj.rotation[2]
    rotMat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ])
    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(obj.translation, (8, 1)).T

    box3d = cornerPosInVelo.transpose()
    return box3d


class Loader:
    def __init__(self, dirs=DIR_TRAIN, queue_size=20, shuffle=False, is_testset=False):
        def make_tags(kind):
            def make_tag(abspath):
                basename = os.path.basename(abspath)
                assert basename.endswith(SUFFIX[kind])
                return basename[:-len(SUFFIX[kind])]
            return frozenset(make_tag(abspath) for abspath in glob.iglob(dirs[kind] + '/*'))
        self.dirs = dirs
        tagsetset = set(make_tags(kind) for kind in SUFFIX.keys())
        assert len(tagsetset) == 1, 'file names of different kinds of data must be the same except their extensions'
        self.tags = list(tagsetset.pop())
        self.tags.sort()
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.tags)
        self.tag_index = 0
        self.is_testset = is_testset

        self.queue_size = queue_size
        self.queue = []
        threading.Thread(target=self.loader, daemon=True).start()

    def __iter__(self):
        for _ in range(len(self.tags)):
            yield self.load()

    def _load(self):
        def make_path(kind, tag):
            return self.dirs[kind] + '/' + tag + SUFFIX[kind]

        def keep_gt_inside_range(labels, boxes3d):
            def box3d_in_top_view(box3d):
                for i in range(8):
                    if TOP_X_MIN <= box3d[i, 0] <= TOP_X_MAX and TOP_Y_MIN <= box3d[i, 1] <= TOP_Y_MAX:
                        continue
                    else:
                        return False
                return True

            if labels.shape[0] == 0:
                return False, None, None
            assert labels.shape[0] == boxes3d.shape[0]

            keep = np.array([box3d_in_top_view(box3d) for box3d in boxes3d], dtype=bool)

            if not np.any(keep):
                return False, None, None
            return True, labels[keep], boxes3d[keep]

        while True:
            tag = self.tags[self.tag_index]

            # load
            lidar = np.fromfile(make_path('lidar', tag), np.float32).reshape((-1, 4))
            rgb = cv2.imread(make_path('rgb', tag))
            resize_coef = np.array([
                (IMAGE_WIDTH - 1) / (rgb.shape[1] - 1),
                (IMAGE_HEIGHT - 1) / (rgb.shape[0] - 1),
            ])
            rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
            calib = Calib(make_path('calib', tag))
            obstacles = read_label(make_path('label', tag), calib)

            # preprocess
            top = clidar_to_top(lidar)
            boxes3d = np.array([box3d_compose(obs) for obs in obstacles], dtype=np.float32)
            labels = np.array([1 if obs.type in ('Van', 'Truck', 'Car', 'Tram') else 0 for obs in obstacles], dtype=np.int32)

            self.tag_index += 1
            if self.tag_index >= len(self.tags):
                self.tag_index = 0
                if self.shuffle:
                    np.random.shuffle(self.tags)

            if self.is_testset:
                break
            is_gt_inside_range, labels, boxes3d = keep_gt_inside_range(labels, boxes3d)
            if is_gt_inside_range:
                break

        return np.array([rgb]), np.array([top]), np.array([labels]), np.array([boxes3d]), tag, calib, resize_coef

    def loader(self):
        while True:
            if len(self.queue) >= self.queue_size:
                sleep(1)
            else:
                self.queue.append(self._load())

    def load(self, peek=False):
        while len(self.queue) == 0:
            sleep(1)
        if peek:
            return self.queue[0]
        return self.queue.pop(0)

    def get_shape(self):
        rgbs, tops, _, _, _, _, _ = self.load(peek=True)
        return tops[0].shape, rgbs[0].shape


def rpn_nms_generator(stride, img_width, img_height, img_scale, nms_thresh, min_size, nms_pre_topn, nms_post_topn):
    def rpn_nms(scores, deltas, anchors, inside_inds):
        def box_transform_inv(et_boxes, deltas):
            num = len(et_boxes)
            boxes = np.zeros((num, 4), dtype=np.float32)
            if num == 0: return boxes

            et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
            et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
            et_cxs = et_boxes[:, 0] + 0.5 * et_ws
            et_cys = et_boxes[:, 1] + 0.5 * et_hs

            et_ws  = et_ws [:, np.newaxis]
            et_hs  = et_hs [:, np.newaxis]
            et_cxs = et_cxs[:, np.newaxis]
            et_cys = et_cys[:, np.newaxis]

            dxs = deltas[:, 0::4]
            dys = deltas[:, 1::4]
            dws = deltas[:, 2::4]
            dhs = deltas[:, 3::4]

            cxs = dxs * et_ws + et_cxs
            cys = dys * et_hs + et_cys
            ws = np.exp(dws) * et_ws
            hs = np.exp(dhs) * et_hs

            boxes[:, 0::4] = cxs - 0.5 * ws
            boxes[:, 1::4] = cys - 0.5 * hs
            boxes[:, 2::4] = cxs + 0.5 * ws
            boxes[:, 3::4] = cys + 0.5 * hs
            return boxes

        def clip_boxes(boxes, width, height):
            boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], width - 1), 0)
            boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], height - 1), 0)
            boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], width - 1), 0)
            boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], height - 1), 0)
            return boxes

        def filter_boxes(boxes, min_size):
            ws = boxes[:, 2] - boxes[:, 0] + 1
            hs = boxes[:, 3] - boxes[:, 1] + 1
            keep = np.where((ws >= min_size) & (hs >= min_size))[0]
            return keep

        scores = scores.reshape((-1, 2,1))
        scores = scores[:,1,:]
        deltas = deltas.reshape((-1, 4))

        scores = scores[inside_inds]
        deltas = deltas[inside_inds]
        anchors = anchors[inside_inds]

        proposals = box_transform_inv(anchors, deltas)

        proposals = clip_boxes(proposals, img_width, img_height)

        keep      = filter_boxes(proposals, min_size*img_scale)
        proposals = proposals[keep, :]
        scores    = scores[keep]

        order = scores.ravel().argsort()[::-1]
        if nms_pre_topn > 0:
            order = order[:nms_pre_topn]
            proposals = proposals[order, :]
            scores = scores[order]

        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if nms_post_topn > 0:
            keep = keep[:nms_post_topn]
            proposals = proposals[keep, :]
            scores = scores[keep]

        roi_scores = scores.squeeze()

        num_proposals = len(proposals)
        batch_inds = np.zeros((num_proposals, 1), dtype=np.float32)
        rois = np.hstack((batch_inds, proposals))

        return rois, roi_scores
    return rpn_nms


def tf_rpn_nms(
    scores, deltas, anchors, inside_inds,
    stride, img_width, img_height, img_scale,
    nms_thresh, min_size, nms_pre_topn, nms_post_topn,
    name='rpn_mns'):

    rpn_nms = rpn_nms_generator(stride, img_width, img_height, img_scale, nms_thresh, min_size, nms_pre_topn, nms_post_topn)
    return tf.py_func(
        rpn_nms,
        [scores, deltas, anchors, inside_inds],
        [tf.float32, tf.float32],
        name=name)


def _net_top(input, anchors, inds_inside, num_bases):
    block = mobilenet(input, down4x=True)
    feature = block
    with tf.variable_scope('upsampling'):
        block = upsample2d(block, factor=2, has_bias=True, trainable=True)
    with tf.variable_scope('RPN'):
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
        scores = conv2d(block, num_kernels=2 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='score')
        probs = tf.nn.softmax(tf.reshape(scores, [-1, 2]), name='prob')
        deltas = conv2d(block, num_kernels=4 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='delta')
    with tf.variable_scope('NMS'):
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        stride = 1.
        img_scale = 1
        rois, roi_scores = tf_rpn_nms(probs, deltas, anchors, inds_inside,
                                      stride, img_width, img_height, img_scale,
                                      nms_thresh=0.3, min_size=stride, nms_pre_topn=6000, nms_post_topn=100,
                                      name='nms')
    return feature, scores, probs, deltas, rois, roi_scores
_net_top.feature_stride = 4
_net_top.anchors_stride = 2


def _net_rgb(input):
    feature = mobilenet(input)
    return feature
_net_rgb.stride = 8


def _net_fusion(input):
    with tf.variable_scope('fuse-net'):
        roi_features_list = []
        for feature_name, feature, roi, pool_height, pool_width, pool_scale in input:
            with tf.variable_scope(feature_name + '-roi-pooling'):
                block, _ = tf_roipooling(feature, roi, pool_height, pool_width, pool_scale, name=feature_name+'-roi_pooling')
            with tf.variable_scope(feature_name + '-feature-conv'):
                for scope_name, n_kernel in ('block1', 128), ('block2', 256), ('block3', 512):
                    with tf.variable_scope(scope_name):
                        block = conv2d_bn_relu(block, num_kernels=n_kernel, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_name+'_conv_1')
                        residual = block
                        block = conv2d_bn_relu(block, num_kernels=n_kernel, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                               name=feature_name+'_conv_2') + residual
                        block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME', name=feature_name+'_avg_pool')
                roi_features = flatten(block)
                roi_features_list.append(roi_features)
        with tf.variable_scope('rois-feature-concat'):
            block = concat(roi_features_list, axis=1, name='concat')
        with tf.variable_scope('fusion-feature-fc'):
            block = linear_bn_relu(block, num_hiddens=512, name='1')
            block = linear_bn_relu(block, num_hiddens=512, name='2')
    return block


def _loss_rpn(scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):
    def modified_smooth_l1(box_preds, box_targets, sigma=3.0):
        sigma2 = sigma * sigma
        diffs = tf.subtract(box_preds, box_targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1

    scores1 = tf.reshape(scores, [-1, 2])
    rpn_scores = tf.gather(scores1, inds)  # remove ignore label
    rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))

    deltas1 = tf.reshape(deltas, [-1, 4])
    rpn_deltas = tf.gather(deltas1, pos_inds)  # remove ignore label

    with tf.variable_scope('modified_smooth_l1'):
        rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)

    rpn_reg_loss = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
    return rpn_cls_loss, rpn_reg_loss


def _loss_fuse(scores, deltas, rcnn_labels, rcnn_targets):
    def modified_smooth_l1(deltas, targets, sigma=3.0):
        sigma2 = sigma * sigma
        diffs = tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1

    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:]) // num_class

    with tf.variable_scope('get_scores'):
        rcnn_scores = tf.reshape(scores, [-1, num_class], name='rcnn_scores')
        rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=rcnn_scores, labels=rcnn_labels))

    with tf.variable_scope('get_detals'):
        num = tf.identity(tf.shape(deltas)[0], 'num')
        idx = tf.identity(tf.range(num)*num_class + rcnn_labels, name='idx')
        deltas1 = tf.reshape(deltas, [-1, dim], name='deltas1')
        rcnn_deltas_with_fp = tf.gather(deltas1, idx, name='rcnn_deltas_with_fp')  # remove ignore label
        rcnn_targets_with_fp = tf.reshape(rcnn_targets, [-1, dim], name='rcnn_targets_with_fp')

        # remove false positive
        fp_idxs = tf.where(tf.not_equal(rcnn_labels, 0), name='fp_idxs')
        rcnn_deltas_no_fp = tf.gather(rcnn_deltas_with_fp, fp_idxs, name='rcnn_deltas_no_fp')
        rcnn_targets_no_fp = tf.gather(rcnn_targets_with_fp, fp_idxs, name='rcnn_targets_no_fp')

    with tf.variable_scope('modified_smooth_l1'):
        rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_no_fp, rcnn_targets_no_fp, sigma=3.0)

    rcnn_reg_loss = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))
    return rcnn_cls_loss, rcnn_reg_loss


def make_net(top_shape, rgb_shape, num_class=2):
    def convert_w_h_cx_cy(base):
        """ Return width, height, x center, and y center for a base (box). """
        w = base[2] - base[0] + 1
        h = base[3] - base[1] + 1
        cx = base[0] + 0.5 * (w - 1)
        cy = base[1] + 0.5 * (h - 1)
        return w, h, cx, cy

    def make_bases_given_ws_hs(ws, hs, cx, cy):
        """ Given a vector of widths (ws) and heights (hs) around a center(cx, cy), output a set of bases. """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        bases = np.hstack((cx - 0.5 * (ws - 1),
                           cy - 0.5 * (hs - 1),
                           cx + 0.5 * (ws - 1),
                           cy + 0.5 * (hs - 1)))
        return bases

    def make_bases_given_ratios(base, ratios):
        """  Enumerate a set of bases for each aspect ratio wrt a base.  """
        w, h, cx, cy = convert_w_h_cx_cy(base)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        bases = make_bases_given_ws_hs(ws, hs, cx, cy)
        return bases

    def make_bases_given_scales(base, scales):
        """ Enumerate a set of bases for each scale wrt a base. """
        w, h, cx, cy = convert_w_h_cx_cy(base)
        ws = w * scales
        hs = h * scales
        bases = make_bases_given_ws_hs(ws, hs, cx, cy)
        return bases

    def make_bases(base_size, ratios, scales):
        """ Generate bases by enumerating aspect ratios * scales, wrt a reference (0, 0, 15, 15) base (box). """
        base = np.array([1, 1, base_size, base_size]) - 1
        ratio_bases = make_bases_given_ratios(base, ratios)
        bases = np.vstack(
            [make_bases_given_scales(ratio_bases[i, :], scales) for i in range(ratio_bases.shape[0])]
        )
        return bases

    def make_anchors(bases, stride, feature_shape):
        """ Reference "Faster R-CNN: Towards Real-Time ObjectDetection with Region Proposal Networks"  Figure 3:Left """
        H, W = feature_shape

        shift_x = np.arange(0, W) * stride
        shift_y = np.arange(0, H) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        B = len(bases)
        HW = len(shifts)
        anchors = bases.reshape((1, B, 4)) + shifts.reshape((1, HW, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((HW * B, 4)).astype(np.int32)
        return anchors

    # constant
    bases = make_bases(16, np.array([0.5, 1, 2], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32))
    stride = _net_top.anchors_stride
    top_view_anchors = make_anchors(bases, stride, (top_shape[0]//stride, top_shape[1]//stride))
    anchors_inside_inds = np.arange(len(top_view_anchors), dtype=np.int32)
    top_anchors = tf.constant(top_view_anchors, dtype=tf.int32, name='anchors')
    top_inside_inds = tf.constant(anchors_inside_inds, dtype=tf.int32, name='inside_inds')

    # placeholder
    top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
    rgb_images = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
    top_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='top_rois')
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rgb_rois')

    with tf.variable_scope('top_view_rpn'):
        top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
            _net_top(top_view, top_anchors, top_inside_inds, len(bases))
        with tf.variable_scope('loss'):
            top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
            top_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
            top_cls_loss, top_reg_loss = _loss_rpn(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    with tf.variable_scope('image_feature'):
        rgb_features = _net_rgb(rgb_images)

    with tf.variable_scope('fusion'):
        fuse_output = _net_fusion(
                (('top', top_features, top_rois, 6, 6, 1. / _net_top.feature_stride),
                 ('rgb', rgb_features, rgb_rois, 6, 6, 1. / _net_rgb.stride),))
        out_shape = (8, 3)
        with tf.variable_scope('predict'):
            dim = np.prod(out_shape)
            fuse_scores = linear(fuse_output, num_hiddens=num_class, name='score')
            fuse_probs = tf.nn.softmax(fuse_scores, name='prob')
            fuse_deltas = linear(fuse_output, num_hiddens=dim * num_class, name='box')
            fuse_deltas = tf.reshape(fuse_deltas, (-1, num_class, *out_shape))
        with tf.variable_scope('loss'):
            fuse_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='fuse_label')
            fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
            fuse_cls_loss, fuse_reg_loss = _loss_fuse(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)

    with tf.variable_scope('minimize_loss'):
        solver = tf.train.AdamOptimizer()
        targets_loss = top_cls_loss + 0.05 * top_reg_loss + fuse_cls_loss + fuse_reg_loss
        solver_op = solver.minimize(loss=targets_loss)

    return {
        # input
        'top_view': top_view,
        'rgb_images': rgb_images,
        'top_rois': top_rois,
        'rgb_rois': rgb_rois,

        'top_inds': top_inds,
        'top_pos_inds': top_pos_inds,
        'top_labels': top_labels,
        'top_targets': top_targets,

        'fuse_labels': fuse_labels,
        'fuse_targets': fuse_targets,

        # output
        'proposals': proposals,
        'proposal_scores': proposal_scores,

        'fuse_probs': fuse_probs,
        'fuse_deltas': fuse_deltas,

        'top_cls_loss': top_cls_loss,
        'top_reg_loss': top_reg_loss,
        'fuse_cls_loss': fuse_cls_loss,
        'fuse_reg_loss': fuse_reg_loss,
        'solver_op': solver_op,
    }, top_view_anchors, anchors_inside_inds


@jit
def lidar_to_top_coords(x, y):
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
    yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)
    return xx, yy


@jit
def box3d_to_top_box(boxes3d):
    num = len(boxes3d)
    boxes = np.zeros((num, 4), dtype=np.float32)

    for n in range(num):
        b = boxes3d[n]

        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)

        umin = min(u0, u1, u2, u3)
        umax = max(u0, u1, u2, u3)
        vmin = min(v0, v1, v2, v3)
        vmax = max(v0, v1, v2, v3)

        boxes[n] = np.array([umin, vmin, umax, vmax])

    return boxes


def rpn_target(anchors, inside_inds, gt_labels, gt_boxes):
    def box_transform(et_boxes, gt_boxes):
        et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
        et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
        et_cxs = et_boxes[:, 0] + 0.5 * et_ws
        et_cys = et_boxes[:, 1] + 0.5 * et_hs

        gt_ws  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
        gt_hs  = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
        gt_cxs = gt_boxes[:, 0] + 0.5 * gt_ws
        gt_cys = gt_boxes[:, 1] + 0.5 * gt_hs

        dxs = (gt_cxs - et_cxs) / et_ws
        dys = (gt_cys - et_cys) / et_hs
        dws = np.log(gt_ws / et_ws)
        dhs = np.log(gt_hs / et_hs)

        deltas = np.vstack((dxs, dys, dws, dhs)).transpose()
        return deltas

    CFG = EasyDict()
    CFG.TRAIN = EasyDict()
    CFG.TRAIN.RPN_BATCHSIZE    = 100
    CFG.TRAIN.RPN_FG_FRACTION  = 0.5
    CFG.TRAIN.RPN_FG_THRESH_LO = 0.7
    CFG.TRAIN.RPN_BG_THRESH_HI = 0.3

    inside_anchors = anchors[inside_inds, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inside_inds), ), dtype=np.int32)
    labels.fill(-1)

    # overlaps between the anchors and the gt process
    overlaps = bbox_overlaps(
        np.ascontiguousarray(inside_anchors,  dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    argmax_overlaps    = overlaps.argmax(axis=1)
    max_overlaps       = overlaps[np.arange(len(inside_inds)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps    = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    labels[max_overlaps <  CFG.TRAIN.RPN_BG_THRESH_HI] = 0   # bg label
    labels[gt_argmax_overlaps] = 1                           # fg label: for each gt, anchor with highest overlap
    labels[max_overlaps >= CFG.TRAIN.RPN_FG_THRESH_LO] = 1   # fg label: above threshold IOU

    # subsample positive labels
    num_fg = int(CFG.TRAIN.RPN_FG_FRACTION * CFG.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels
    num_bg = CFG.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    idx_label  = np.where(labels != -1)[0]
    idx_target = np.where(labels ==  1)[0]

    pos_neg_inds = inside_inds[idx_label]
    labels = labels[idx_label]

    pos_inds = inside_inds[idx_target]
    pos_anchors = inside_anchors[idx_target]
    pos_gt_boxes = gt_boxes[argmax_overlaps][idx_target]
    targets = box_transform(pos_anchors, pos_gt_boxes)

    return pos_neg_inds, pos_inds, labels, targets


@jit
def top_to_lidar_coords(xx, yy):
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    y = Yn * TOP_Y_DIVISION - (xx + 0.5) * TOP_Y_DIVISION + TOP_Y_MIN
    x = Xn * TOP_X_DIVISION - (yy + 0.5) * TOP_X_DIVISION + TOP_X_MIN
    return x, y


@jit
def top_box_to_box3d(boxes):
    num = len(boxes)
    boxes3d = np.zeros((num, 8, 3), dtype=np.float32)
    for n in range(num):
        x1, y1, x2, y2 = boxes[n]
        points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        for k in range(4):
            xx, yy = points[k]
            x, y = top_to_lidar_coords(xx, yy)
            boxes3d[n,   k, :] = x, y,  -2
            boxes3d[n, 4+k, :] = x, y, 0.4
    return boxes3d


@jit
def box3d_transform(et_boxes3d, gt_boxes3d):
    num = len(et_boxes3d)
    deltas = np.zeros((num, 8, 3), dtype=np.float32)
    for n in range(num):
        e = et_boxes3d[n]
        center = np.sum(e, axis=0, keepdims=True) / 8
        scale = (np.sum((e - center) ** 2) / 8) ** 0.5
        g = gt_boxes3d[n]
        deltas[n] = (g - e) / scale
    return deltas


def fusion_target(rois, gt_labels, gt_boxes, gt_boxes3d):
    CFG = EasyDict()
    CFG.TRAIN = EasyDict()
    CFG.TRAIN.RCNN_BATCH_SIZE   = 128
    CFG.TRAIN.RCNN_FG_FRACTION  = 0.25
    CFG.TRAIN.RCNN_FG_THRESH_LO = 0.5

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1, 5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'

    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(extended_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]

    # Select false positive
    fp_inds = np.where((max_overlaps < 0.01))[0]

    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, fp_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]
    labels[fg_inds.size:] = 0

    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes = rois[:, 1:5]

    et_boxes3d = top_box_to_box3d(et_boxes)
    targets = box3d_transform(et_boxes3d, gt_boxes3d)
    targets[np.where(labels == 0), :, :] = 0

    return rois, labels, targets


def project_to_roi3d(top_rois):
    rois3d = top_box_to_box3d(top_rois[:, 1:5])
    return rois3d


def project_to_rgb_roi(rois3d, calib, resize_coef):
    def box3d_to_rgb_box(boxes3d, calib, resize_coef):
        return (calib.velo_to_im(boxes3d) * resize_coef).astype(np.int32)

    num = len(rois3d)
    rois = np.zeros((num, 5), dtype=np.int32)
    projections = box3d_to_rgb_box(rois3d, calib, resize_coef)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:, 0])
        maxx = np.max(qs[:, 0])
        miny = np.min(qs[:, 1])
        maxy = np.max(qs[:, 1])
        rois[n, 1:5] = minx, miny, maxx, maxy
    return rois


def train(n_iter, save_path, load_path):
    loader = Loader(shuffle=True)
    net, top_view_anchors, anchors_inside_inds = make_net(*loader.get_shape())
    with tf.Session() as sess:
        if load_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(sess, load_path)
        print()
        print(' '*(18+len(str(n_iter))) + 'top_cls  top_reg   fuse_cls fuse_reg')
        time_start = time_prev = time()
        for i in range(n_iter):
            batch_rgb_images, batch_top_view, batch_gt_labels, batch_gt_boxes3d, frame_id, calib, resize_coef = loader.load()

            fd1 = {
                IS_TRAIN_PHASE: True,
                net['top_view']: batch_top_view,
            }
            batch_proposals, batch_proposal_scores = sess.run((net['proposals'], net['proposal_scores']), fd1)

            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d[0])
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets = \
                rpn_target(top_view_anchors, anchors_inside_inds, batch_gt_labels[0], batch_gt_top_boxes)
            batch_top_rois, batch_fuse_labels, batch_fuse_targets = \
                fusion_target(batch_proposals, batch_gt_labels[0], batch_gt_top_boxes, batch_gt_boxes3d[0])
            batch_rois3d = project_to_roi3d(batch_top_rois)
            batch_rgb_rois = project_to_rgb_roi(batch_rois3d, calib, resize_coef)

            fd2 = {
                IS_TRAIN_PHASE: True,
                net['top_view']: batch_top_view,
                net['rgb_images']: batch_rgb_images,
                net['top_rois']: batch_top_rois,
                net['rgb_rois']: batch_rgb_rois,

                net['top_inds']: batch_top_inds,
                net['top_pos_inds']: batch_top_pos_inds,
                net['top_labels']: batch_top_labels,
                net['top_targets']: batch_top_targets,

                net['fuse_labels']: batch_fuse_labels,
                net['fuse_targets']: batch_fuse_targets,
            }
            _, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss = \
                sess.run((net['solver_op'], net['top_cls_loss'], net['top_reg_loss'], net['fuse_cls_loss'], net['fuse_reg_loss']), fd2)
            if i % 40 == 0:
                print('train: | %6d/%d %8.5f %8.5f | %8.5f %8.5f' %
                      (i, n_iter, t_cls_loss, t_reg_loss, f_cls_loss, f_reg_loss), flush=True)
            if i % 200 == 199:
                time_now = time()
                print('200 iterations took %.2f seconds.' % (time_now - time_prev), flush=True)
                time_prev = time_now
        print('%d iterations took %.2f seconds.' % (n_iter, time() - time_start))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tf.train.Saver().save(sess, save_path)


if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int,
                        help='number of iterations')
    parser.add_argument('-s', '--save-path', default='checkpoint/'+strftime('%Y%m%d_%H%M'),
                        help='Where to save model after training. Defaults to checkpoint/yyyymmdd_hhmm.')
    parser.add_argument('-l', '--load', nargs='?', const='', dest='load_path',
                        help='Load pretrained model before training. If LOAD_PATH is not specified, it defaults to SAVE_PATH.')
    args = parser.parse_args()
    assert os.path.basename(args.save_path) != 'checkpoint', 'save_path basename must not be "checkpoint"'
    if args.load_path == '':
        args.load_path = args.save_path

    train(args.n_iter, args.save_path, args.load_path)
    print('train.py took %.2f seconds.' % (time() - time_start))
