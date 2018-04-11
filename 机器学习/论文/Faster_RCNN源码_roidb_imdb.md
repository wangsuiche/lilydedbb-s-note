# Faster R-CNN —— roidb & imdb

首先在 `tools/train_net.py` 中根据 `argparser` 传过来的 `imdb_name` 解析 `imdb`:

```python
# tools/train_net.py
imdb, roidb = combined_roidb(args.imdb_name)
```

其中 `combined_roidb` 函数允许在 `imdb_name` 通过 `+` 连接多个数据集:

```python
def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)  # from datasets.factory import get_imdb
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)  # from fast_rcnn.train import get_training_roidb
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]  # 允许在 `imdb_name` 通过 `+` 连接多个数据集
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb
```

其中 `get_imdb` 为 `lib/datasets/factory.py` 中的工厂模式函数：

```python
__sets = {}

# ...

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# ...

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# ...

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
```

而这其中 `pascal_voc` 和 `coco` 等对象，都继承自 `lib/datasets/imdb.py` 中的 `imdb` 对象

`get_roidb` 中用到的 `imdb.set_proposal_method` 源自：

```python
# lib/datasets/imdb.py
def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method
```

`Faster R-CNN` 中 `PROPOSAL_METHOD` 用到的一般为 `gt`（默认为 `selective_search`）；而 `self.gt_roidb` 在 `pascal_voc.py` 或者 `coco.py` 这类文件中定义：

```python
def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    print(cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        return roidb

    gt_roidb = [self._load_zhuying_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb to {}'.format(cache_file)

    return gt_roidb
```

`get_roidb` 中用到的 `get_training_roidb` 函数在 `lib/fast_rcnn/train.py` 中；其主要作用是如果设置了翻转，则增加翻转，然后直接调用 `prepare_roidb` 方法

```python
# lib/fast_rcnn/train.py
def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)  # import roi_data_layer.roidb as rdl_roidb
    print 'done'

    return imdb.roidb
```

`prepare_roidb`:

```python
# lib/roi_data_layer/roidb.py
def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
```

`imdb` 的 `roidb` 属性：

```python
@property
def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
        return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

@property
def roidb_handler(self):
    return self._roidb_handler
```
