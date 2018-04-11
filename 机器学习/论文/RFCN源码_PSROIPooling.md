# R-FCN 源码 —— PSROIPooling & PSROIAlign

### PSROIPooling

RFCN 中官方代码中 （以 `/models/rfcn_prototxts/ResNet-101L_res3a/train_val.prototxt` 为例），关于 `PSROIPoolingLayer` 的 `prototxt` 如下：

```
#--------------position sensitive RoI pooling--------------
layer {
    bottom: "rfcn_cls"
    bottom: "rois"
    top: "psroipooled_cls_rois"
    name: "psroipooled_cls_rois"
    type: "PSROIPooling"
    psroi_pooling_param {
        spatial_scale: 0.0625
        output_dim: 21
        group_size: 7
    }
}

layer {
    bottom: "psroipooled_cls_rois"
    top: "cls_score"
    name: "ave_cls_score_rois"
    type: "Pooling"
    pooling_param {
        pool: AVE
        kernel_size: 7
        stride: 7
    }
}


layer {
    bottom: "rfcn_bbox"
    bottom: "rois"
    top: "psroipooled_loc_rois"
    name: "psroipooled_loc_rois"
    type: "PSROIPooling"
    psroi_pooling_param {
        spatial_scale: 0.0625
        output_dim: 8
        group_size: 7
    }
}

layer {
    bottom: "psroipooled_loc_rois"
    top: "bbox_pred"
    name: "ave_bbox_pred_rois"
    type: "Pooling"
    pooling_param {
        pool: AVE
        kernel_size: 7
        stride: 7
    }
}
```

[官方源码](https://github.com/daijifeng001/caffe-rfcn)（未实现 cpu 的 forward 和 backward）

**SetUp**:

- `spatial_scale_`: 输入的特征图相对于原始输入的尺寸比例，`1 / stride`
- `output_dim_`: 输出的 position-sensitive score maps 的维度，即 `C + 1`
- `group_size_`: pooling 的宽、高（pooled_height_ = pooled_width_ = group_size_）

```c
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    PSROIPoolingParameter psroi_pooling_param = this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }
```

**Reshape**

这里对 `input` 的维度进行检查，是否符合 `psroi_pooling` 的要求

`top[0]` 的维度为 (N, C + 1, K, K)；N 为 ROI 数量，C 为类别数，K 为 pooled_height（pooled_width)

```c
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }
```

**Forward**:

（官方代码未实现 cpu 版，这里只分析 gpu 版）

output 即 `top[0]` 的维度为 (n, ctop, ph, pw)

```c
int pw = index % pooled_width;
int ph = (index / pooled_width) % pooled_height;
int ctop = (index / pooled_width / pooled_height) % output_dim;
int n = index / pooled_width / pooled_height / output_dim;
```

根据 `index` 取对应的 `ROI`，并把 `ROI` 缩放为对应输入特征图的尺寸

```c
// [start, end) interval for spatial sampling
bottom_rois += n * 5;
int roi_batch_ind = bottom_rois[0];
Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;
```

避免 0 大小的 `ROI`

```c
Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
```

计算每个 bin 的大小，并由此计算此步 pooling 的始末位置 （hstart, hend, wstart, wend）

```c
Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                  + roi_start_h);
int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                  + roi_start_w);
int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                + roi_start_h);
int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                + roi_start_w);
```

避免越界

```c
hstart = min(max(hstart, 0), height);
hend = min(max(hend, 0), height);
wstart = min(max(wstart, 0),width);
wend = min(max(wend, 0), width);
bool is_empty = (hend <= hstart) || (wend <= wstart);
```

Pooling:

这里不同于 `ROIPooling` 的地方在于对于变量 c 的计算，即取 bottom_data 的哪一维度 feature map

```c
int gw = pw;
int gh = ph;
int c = (ctop*group_size + gh)*group_size + gw;

bottom_data += (roi_batch_ind * channels + c) * height * width;
Dtype out_sum = 0;
for (int h = hstart; h < hend; ++h){
  for (int w = wstart; w < wend; ++w){
    int bottom_index = h*width + w;
    out_sum += bottom_data[bottom_index];
  }
}

Dtype bin_area = (hend - hstart)*(wend - wstart);
top_data[index] = is_empty? 0. : out_sum/bin_area;
```

最后记录 `backword` 中用到的 `mapping_channel`:

```c
mapping_channel[index] = c;
```

**Backward**:

`backward` 中有很多和 `forward` 中重复的计算步骤，不再重复，只分析关键部分：

```c
int c = mapping_channel[index];
Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
Dtype bin_area = (hend - hstart)*(wend - wstart);
Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
for (int h = hstart; h < hend; ++h){
  for (int w = wstart; w < wend; ++w){
    int bottom_index = h*width + w;
    caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
  }
}
```

------

官方源码：

**`psroi_pooling_layer.cpp`**

```c
// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/rfcn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    PSROIPoolingParameter psroi_pooling_param = this->layer_param_.psroi_pooling_param();
    spatial_scale_ = psroi_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(psroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(psroi_pooling_param.group_size(), 0)
      << "group_size must be > 0";

    output_dim_ = psroi_pooling_param.output_dim();
    group_size_ = psroi_pooling_param.group_size();
    pooled_height_ = group_size_;
    pooled_width_ = group_size_;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(PSROIPoolingLayer);
#endif

  INSTANTIATE_CLASS(PSROIPoolingLayer);
  REGISTER_LAYER_CLASS(PSROIPooling);

}
```

**`psroi_pooling_layer.cu`**

```c
// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/rfcn_layers.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      Dtype bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? 0. : out_sum/bin_area;
      mapping_channel[index] = c;
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_, top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, top_diff, mapping_channel_ptr, top[0]->num(), spatial_scale_,
      channels_, height_, width_, pooled_height_, pooled_width_, output_dim_,
      bottom_diff, bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIPoolingLayer);

}  // namespace caffe
```


------


### PSROIAlign

这里参考 [https://github.com/afantideng/R-FCN-PSROIAlign](https://github.com/afantideng/R-FCN-PSROIAlign) 的实现：


**`psroi_align_layer.cu`**:

完整源码：

```c
// --------------------------------------------------------
// R-FCN
// Written by Afanti<afanti.deng@gmail.com>
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "caffe/layers/psroi_align_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
        __device__ void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & val) {

            // deal with cases that inverse elements are out of feature map boundary
            if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) return;

            if (h <= 0) h = 0;
            if (w <= 0) w = 0;


            int h_high;             // h_high 是比 h 大的最小整数
            int w_high;             // w_high 是比 w 大的最小整数
            int h_low = (int) h;    // h_low  是比 h 小的最大整数
            int w_low = (int) w;    // w_low  是比 w 小的最大整数

            if (w_low >= width - 1) {
                w_low = width - 1;
                w_high = width-1;
                w = (Dtype) w_low;
            } else
                w_high = w_low + 1;

            if (h_low >= height - 1) {
                h_high = height-1;
                h_low = height - 1;
                h = (Dtype) h_low;
            } else
                h_high = h_low + 1;


            Dtype l_dh = h - h_low;
            Dtype l_dw = w - w_low;
            Dtype h_dh = 1 - l_dh, h_dw = 1 - l_dw;

            // 进行双线性内插
            Dtype u1 = bottom_data[h_low * width + w_low];
            Dtype u2 = bottom_data[h_low * width + w_high];
            Dtype u3 = bottom_data[h_high * width + w_low];
            Dtype u4 = bottom_data[h_high * width + w_high];
            Dtype w1 = h_dh * h_dw, w2 = h_dh * l_dw, w3 = l_dh * h_dw, w4 = l_dh * l_dw;

            val = (w1 * u1 + w2 * u2 + w3 * u3 + w4 * u4);
        }

  template <typename Dtype>
  __global__ void PSROIAlignForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,  // 输出通道数
    const int group_size,  // k*k*(c+1) 中的 k
    Dtype* top_data,
    int* mapping_channel,
    Dtype* sample_pos_data,
    const int sample_num) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      // 获得当前RoI的宽和高在池化前特征图上的起始和结束索引值, 浮点数
      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
      Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
      Dtype hend   = static_cast<Dtype>(ph + 1) * bin_size_h;
      Dtype wend   = static_cast<Dtype>(pw + 1) * bin_size_w;

      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart + roi_start_h, Dtype(0)), Dtype(height-1));
      hend = min(max(hend + roi_start_h, Dtype(0)), Dtype(height-1));
      wstart = min(max(wstart + roi_start_w, Dtype(0)), Dtype(width-1));
      wend = min(max(wend + roi_start_w, Dtype(0)), Dtype(width-1));
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw; //

      // 在池化前特征图上采样点之间的距离，浮点数 (在 h 和 w 两个方向上)
      Dtype sample_h = bin_size_h / (sample_num + 1);
      Dtype sample_w = bin_size_w / (sample_num + 1);
      Dtype val = 0;
      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0.0;
      int p_counter = -1;
      Dtype* sample_pos_diff = sample_pos_data + index * sample_num * sample_num * 2;
      for (int i = 1; i <= sample_num; ++i) {
          for (int j = 1; j <= sample_num; ++j) {
              ++p_counter;
              Dtype cur_h = hstart + i * sample_h;
              Dtype cur_w = wstart + j * sample_w;
              if (cur_h >= hend || cur_w >= wend) continue;
              bilinear_interpolate(bottom_data, height, width, cur_h, cur_w, val);
              out_sum += val;
              sample_pos_diff[p_counter * 2 + 0] = cur_h;
              sample_pos_diff[p_counter * 2 + 1] = cur_w;
              // updated = true;
          }
      }
      // Dtype bin_area = (hend - hstart) * (wend - wstart);
      top_data[index] = is_empty ? 0. : out_sum / static_cast<Dtype>(sample_num * sample_num);
      mapping_channel[index] = c;
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* sample_pos_data = sample_pos_.mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr, sample_pos_data, sample_num_);
    CUDA_POST_KERNEL_CHECK;
  }


  template <typename Dtype>
  __global__ void PSROIAlignBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const Dtype* sample_pos_data,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois,
    const int sample_num) {
    // 遍历池化后特征图的每一个像素点
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // ------------------------------------ 计算当前 pooled 后的点在原图中的位置范围 ------------------------------------------------
      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3] + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4] + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // -------------------------------------------------------------------------------------

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
      const Dtype* sample_pos_diff = sample_pos_data + index * sample_num * sample_num * 2;
      Dtype diff_val = is_empty ? 0. : top_diff[index] / (sample_num * sample_num);
      // diff_val = 0.;
      // diff_value = diff_val;
      // printf("diff_val: %f\n", float(diff_val));
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          for(int i = 0; i < sample_num * sample_num; ++i){
              Dtype d_h = abs(sample_pos_diff[2*i + 0] - h);
              Dtype d_w = abs(sample_pos_diff[2*i + 1] - w);
              if(d_h < 1 && d_w < 1){
                    int bottom_index = h*width + w;
                    caffe_gpu_atomic_add((1 - d_h)*(1 - d_w)*diff_val, offset_bottom_diff + bottom_index);
              }
          }
        }
      }
    }
  }

  template <typename Dtype>
  void PSROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    const Dtype* sample_pos_data = sample_pos_.gpu_data();
    // Dtype diff_value = static_cast<Dtype>(0);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAlignBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr, sample_pos_data,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois, sample_num_);
    // LOG(INFO) << "diff_value: " << diff_value;
    CUDA_POST_KERNEL_CHECK;
  }


  INSTANTIATE_LAYER_GPU_FUNCS(PSROIAlignLayer);

}  // namespace caffe
```