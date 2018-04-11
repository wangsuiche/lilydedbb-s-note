# Faster R-CNN 源码 —— ProposalTargetLayer

`Proposal Target Layer` 只会在 `train` 阶段用到，将从 `RPN` 和 `Proposal Layer` 传过来的 `bbox` 和 `ground truth` 比较，判定正负样本，生成 `ground truth delta` 和 `classfication labels`  用来计算最后的 `loss_delta` 和 `loss_cls` ，将 `rois` 传给 `ROIPooling Layer` 用来做第二阶段的 `class-specific` 分类

`Faster R-CNN` 中关于 `ProposalTargetLayer` 的 `prototxt`:

```
layer {
  name: 'roi-data'
  type: 'Python'
  bottom: 'rpn_rois'
  bottom: 'gt_boxes'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': 61"
  }
}
```

源码注释中对该层的解释：

> Assign object detection proposals to ground-truth targets. Produces proposal classification labels and bounding-box regression targets.

**Setup**:

```python
def setup(self, bottom, top):
    # layer_params = yaml.load(self.param_str_)
    layer_params = yaml.load(self.param_str)
    self._num_classes = layer_params['num_classes']

    # sampled rois (0, x1, y1, x2, y2)
    top[0].reshape(1, 5)
    # labels
    top[1].reshape(1, 1)
    # bbox_targets
    top[2].reshape(1, self._num_classes * 4)
    # bbox_inside_weights
    top[3].reshape(1, self._num_classes * 4)
    # bbox_outside_weights
    top[4].reshape(1, self._num_classes * 4)
```

**Forward**:

首先初始化一些变量：

- `bottom[0]`: `proposal` 层传过来的 `all_rois`，格式为 `(0, x1, y1, x2, y2)`
- `bottom[1]`: `data` 层传过来的 `gt_boxes`，格式为 `(x1, y1, x2, y2, label)`

```python
# Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
# (i.e., rpn.proposal_layer.ProposalLayer), or any other source
all_rois = bottom[0].data
# GT boxes (x1, y1, x2, y2, label)
gt_boxes = bottom[1].data
```

```python
# Include ground-truth boxes in the set of candidate rois
zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
all_rois = np.vstack(
    (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
)
```

`Faster R-CNN` 默认只支持 `single batch`，即一张一张图片 `Forward`；`cfg.TRAIN.BATCH_SIZE` 指 `Minibatch size (number of regions of interets [ROIs])`，即在一张图上对多少个 `ROIs` 进行分类和坐标回归；`cfg.TRAIN.FG_FRACTION` 指 `Fraction of minibath that is labeled foreground (i.e. class > 0)`，即一个 `minibatch` 里正样本的占比

```python
num_images = 1
rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
```

根据分类标签和 `bbox` 回归结果，对 `ROIs` 进行采样

```python
# Sample rois with classification labels and bounding box regression
# targets
labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, gt_boxes, fg_rois_per_image,
    rois_per_image, self._num_classes)
```

**`_sample_rois` 函数**：

计算所有的 `ROIs` 和 所有的 `gt_bboxes` 之间的 `IOU`:

```python
# overlaps: (rois x gt_boxes)
overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
gt_assignment = overlaps.argmax(axis=1)
max_overlaps = overlaps.max(axis=1)
labels = gt_boxes[gt_assignment, 4]
```

根据 `cfg.TRAIN.FG_THRESH` 选取正样本，规则如论文中所说：

> We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box.

```python
# Select foreground RoIs as those with >= FG_THRESH overlap
fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
# Guard against the case when an image has fewer than fg_rois_per_image
# foreground RoIs
fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
# Sample foreground regions without replacement
fg_rois_per_this_image = int(fg_rois_per_this_image)
if fg_inds.size > 0:
    fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
```

根据 `cfg.TRAIN.BG_THRESH_LO` 和 `cfg.TRAIN.BG_THRESH_HI` 选取负样本，规则如论文中所说：

> We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes.

```python
# Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                   (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
# Compute number of background RoIs to take from this image (guarding
# against there being fewer than desired)
bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
# Sample background regions without replacement
if bg_inds.size > 0:
    bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
```

为选取的 `ROIs`，产生 `labels`（背景为0）和 `rois`:

```python
# The indices that we're selecting (both fg and bg)
keep_inds = np.append(fg_inds, bg_inds)
# Select sampled values from various arrays:
labels = labels[keep_inds]
# Clamp labels for the background RoIs to 0
labels[fg_rois_per_this_image:] = 0
rois = all_rois[keep_inds]
```

计算 `Targets`:

```python
bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
```

**`_compute_targets` 函数**:

```python
def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
```

**`bbox_transform` 函数**:

```python
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
```

获得 `bbox` 回归标签：

```python
bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
```

```python
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights
```

这其中：

```python
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
```

`reshape` 后传给 `top`:

```python
# sampled rois
top[0].reshape(*rois.shape)
top[0].data[...] = rois

# classification labels
top[1].reshape(*labels.shape)
top[1].data[...] = labels

# bbox_targets
top[2].reshape(*bbox_targets.shape)
top[2].data[...] = bbox_targets

# bbox_inside_weights
top[3].reshape(*bbox_inside_weights.shape)
top[3].data[...] = bbox_inside_weights

# bbox_outside_weights
top[4].reshape(*bbox_inside_weights.shape)
top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
```

------

完整源码：

```python
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # layer_params = yaml.load(self.param_str_)
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    fg_rois_per_this_image = int(fg_rois_per_this_image)
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
```

