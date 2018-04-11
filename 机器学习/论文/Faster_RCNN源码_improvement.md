# Faster R-CNN 源码 —— Improvement

### bbox_vote

参考自：[bbox_vote](https://github.com/sanghoon/pva-faster-rcnn/blob/master/lib/fast_rcnn/test.py)

```python
def bbox_vote(dets_NMS, dets_all, thresh=0.5):
    dets_voted = np.zeros_like(dets_NMS)   # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
			np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
			np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert(len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)        # Weighted bounding boxes
        dets_voted[i][4] = det[4]                         # Keep the original score

        # Weighted scores (if enabled)
        if cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])

    return dets_voted
```

**Input**:

- `dets_all`: 网络输出的所有 `detections`
- `dets_NMS`: `dets_all` 经过 `NMS` 过滤之后的剩下的 `dets` (`survived box`)

**Process**:

1. 首先计算 `dets_all` 和 `dets_NMS` 之间的 `IOU` 值

```python
_overlaps = bbox_overlaps(
			np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
			np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))
```

2. 遍历每一个 `dets_NMS`：

    ```
    for det in dets_NMS:
        将 dets_all 中和该 det 的 IOU 值大于阈值的 dets 筛选出来
        将筛选过后的 dets 按照其 scores 求加权均值坐标
        将求出的加权均值坐标赋给该 det （该 det 的 score 不变）
        if 需要求加权平均 score：
            则将筛选过后的 dets 的 scores 从大到小排序
            取前 n_agreement 个 scores 求均值，赋给该 det
    ```

    ```python
    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert(len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)        # Weighted bounding boxes
        dets_voted[i][4] = det[4]                         # Keep the original score

        # Weighted scores (if enabled)
        if cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])
    ```
