# Group Normalization

自从 $Batch \ Normalization$ 2015年被提出后，其对于训练更深的神经网络的高效性，使得 $BN$ 在之后提出的基础网络模型中被广泛使用，自然也成为在众多 $Computer \ Vision$ task 被广泛应用的 trick，已将成为当前 $stat \ of \ the \ art$ 的基础（标配）。

但 $Batch \ Normalization$ 有以下几个缺点：

1. 如论文中所说 **```BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation.```**，$BN$ 过于依赖 $mini \ batch$ 的大小，在 $mini \ batch \ size$ 郭小的时候，因为对一个 $batch$ 内数据分布的错误估计，$BN$ 带来的错误率会迅速增加。$BN$ 往往需要一个足够大的 $batch \ size$ ，但是目前由于计算设备内存的限制，许多流行的 $CV$ 任务（诸如：$object \ detection$, $instance segementation $ 等）为了保证好的结果往往需要大分辨率的输入，导致 $(mini-)batch \ size$ 非常小，如仅为 $1～2$。$BN$ 的引入往往需要这些模型在网络结构上和 $batch \ size$ 上作出妥协和折中。



$BN$ 的其他变种，如 $Instance \ Normalization$ 和 $Layer \ Normalizaiton$，同样避免沿 $batch$ 维度进行 $normalization$，即今可能摆脱对于 $batch \ size$ 的依赖。$Instance \ Normalization$ 在 $generative \ models$ （如 $style \ transformation$ 和 $GANs$）中很有效，$Layer \ Normalizaiton$ 在 $equential \ models$（如 $RNN/LSTM$）很有效，但是 $IN$ 和 $LN$ 在视觉识别任务中收到限制，但是 $GN$ 不会。

$GN$ 和 $BN$、$LN$、$IN$ 之间的比较：

![GN_figure2](/Users/dbb/Desktop/dbb-gitbook/images/GN_figure2.png)

$GN$ 和 $BN$、$LN$、$IN$ 都执行下面相同的操作：
$$
\hat{x}_i = \frac{1}{\sigma_i}(x_i - \mu_i)
$$

$$
\mu_i = \frac{1}{m}\sum_{k\in \mathcal{S}_i} x_k, \quad \sigma_i = \sqrt{ \frac{1}{m}\sum_{k\in \mathcal{S}_i} (x_k-\mu_i)^2 + \epsilon},
$$
他们的不同之处主要在于 $\mathcal{S}_i$ 的选取。

$Batch \ Norm: \ \mathcal{S}_i = \{k~|~k_C=i_C\},$ 沿着 $C$ 维度进行归一化，即在不同 $batch$ 同一 $C$ 维度的像素一起归一化（这会依赖 $batch \ size$ ）

$Layer \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N\},$ 沿着 sample 的维度进行归一化，即每个同属于一个 sample 的像素一起归一化

$ Instance \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N, k_C=i_C\}.$ 对每一个 channel 和 sample 进行归一化

$Group \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N, \lfloor \frac{k_C}{C/G} \rfloor=\lfloor \frac{i_C}{C/G} \rfloor\}$



$Batch \ Norm: \ \mathcal{S}_i = \{k~|~k_C=i_C\},$ 

$Layer \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N\},$ 

$ Instance \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N, k_C=i_C\}.$

$Group \ Norm: \ \mathcal{S}_i = \{k~|~k_N=i_N, \lfloor \frac{k_C}{C/G} \rfloor=\lfloor \frac{i_C}{C/G} \rfloor\}$

$IN$ 相当于 $GN$ 在 $G = C$ 条件下的极端情况，相对于 $GN$ 只对同一 $channel$ 的特征进行归一化，只考虑了空间维度的数据分布，忽视了结合不同 $channel$ 之间信息的可能。

>  $IN$ can only rely on the spatial dimension for computing the mean and variance and it **misses the opportunity of exploiting the channel dependence**.

$LN$ 想当一 $GN$ 在 $G = 1$ 条件下的极端情况，相对于 $GN$，$LN$ 假设所有的 $channel$ 都有一样的分布（$GN$ 则只认为临近的 $channel$ 之间存在相似的数据分布，因为相邻 $channel$ 之间可能存在着某种特征之间的关联），但是这个假设在卷积网络中应用时可能不那么有效。$LN$ 那边论文中也提到了这一点。

> **$LN$ assumes all channels in a layer make “similar contributions”** [3]. Unlike the case of fully-connected layers studied in [3], **this assumption can be less valid with the presence of convolutions**, as discussed in [3].

**Python code of Group Norm based on TensorFlow:**

```python
def GroupNorm(x, gamma, beta, G, eps=1e−5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1] 
    # G: number of groups for GN
    
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True) 
    x = (x − mean) / tf.sqrt(var + eps)
    
    x = tf.reshape(x, [N, C, H, W]) 
    
    return x ∗ gamma + beta
```

**实验：**

不同的 $Normalization$ 方法在 $ImageNet$ 分类任务上的实验比较：

![GN_figure4](/Users/dbb/Desktop/dbb-gitbook/images/GN_figure4.png)

可以看出在训练集上，$GN$ 比 $BN$ 的错误率略低，说明 $GN$ 可以更轻松地优化网络。但是在验证集上，$GN$ 比 $BN$ 的错误率略高，本论文中解释说，因为 $GN$ 相比于 $BN$ 失去了部分 $regularization$ 的功能（$BN$ 通过随机取样达到了正则化的功能。

![GN_figure5](/Users/dbb/Desktop/dbb-gitbook/images/GN_figure5.png)

$GN$ 和 $BN$ 对于 $batch \ size$ 的敏感度比较。可以看到 $BN$ 严重依赖 $batch size$ 的大小。



------

> **BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation.**
>
> In this paper, we present Group Normalization (GN) as a simple alternative to BN. GN divides the channels into groups and computes within each group the mean and variance for normalization. **GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.**
>
> Moreover, **GN can be naturally transferred from pre-training to fine-tuning**. GN can outperform or compete with its BN-based counterparts for object detection and segmentation in COCO, and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks. GN can be easily implemented by a few lines of code in modern libraries.

### 1. Introduction

BN normalizes the features by the mean and variance computed within a (mini-)batch. 

The stochastic uncertainty of the batch statistics also acts as a regularizer that can benefit generalization. 

In particular, it is required for BN to work with a ***sufficiently large batch size*** ($e.g.$, 32 per worker [26, 58, 20]). **A small batch leads to inaccurate estimation of the batch statistics, and *reducing BN's batch size increases the model error dramatically***. 

The heavy reliance on BN's effectiveness to train models in turn prohibits people from exploring higher-capacity models that would be limited by memory.

The restriction on batch sizes is more demanding in computer vision tasks including detection [12, 46, 18], segmentation [37, 18], video recognition [59, 6], and other high-level systems built on them. For example, the Fast/er and Mask R-CNN frameworks [12, 46, 18] use a batch size of 1 or 2 images because of higher resolution, where BN is “frozen” by transforming to a linear layer [20]. The usage of BN often requires these systems to compromise between the model design and batch sizes.

We notice that many classical features like SIFT [38] and HOG [9] are *group-wise* features and involve *group-wise normalization*.

Analogously, we propose GN as a layer that divides channels into groups and normalizes the features within each group (Figure 2). GN does not exploit the batch dimension, and its computation is independent of batch sizes.

Moreover, although the batch size may change, GN can naturally transfer from pre-training to fine-tuning.

### 2. Related Work

It is well-known that normalizing the input data makes training faster [33].



### 3. Group Normalization

#### 3.1 Formulation

A family of feature normalization methods, including BN, LN, IN, and GN, perform the following computation:
$$
\hat{x}_i = \frac{1}{\sigma_i}(x_i - \mu_i).
$$
Here $x$ is the feature computed by a layer, and $i$ is an index. In the case of 2D images, $i=(i_N, i_C, i_H, i_W)$ is a 4D vector indexing the features in $(N, C, H, W)$ order, where $N$ is the batch axis, $C$ is the channel axis, and $H$ and $W$ are the spatial height and width axis.

$\mu$ and $\sigma$ are the mean and standard deviation (std) computed by:
$$
\mu_i = \frac{1}{m}\sum_{k\in \mathcal{S}_i} x_k, \quad \sigma_i = \sqrt{ \frac{1}{m}\sum_{k\in \mathcal{S}_i} (x_k-\mu_i)^2 + \epsilon},
$$
with $\epsilon$ as a small constant. $\mathcal{S}_i$ is the set of pixels in which the mean and std are computed, and $m$ is the size of this set.  

**Many types of feature normalization methods mainly differ in how the set $\mathcal{S}_i$ is defined.**

In $Batch \ Norm$, the set $\mathcal{S}_i$ is defined as:
$$
\mathcal{S}_i = \{k~|~k_C=i_C\},
$$
where $i_C$ (and $k_C$) denotes the sub-index of $i$ (and $k$) along the $C$ axis. This means that the pixels sharing the same channel index are normalized together, $i.e.$, for each channel, BN computes $\mu$ and $\sigma$ along the $(N, H, W)$ axes.

In $Layer \ Norm$, the set is:
$$
\mathcal{S}_i = \{k~|~k_N=i_N\},
$$
meaning that $LN$ computes $\mu$ and $\sigma$ along the $(C, H, W)$ axes for each sample.

In $Instance \ Norm$, the set is:
$$
\mathcal{S}_i = \{k~|~k_N=i_N, k_C=i_C\}.
$$
meaning that $IN$ computes $\mu$ and $\sigma$ along the $(H, W)$ axes for each sample and each channel.

As in [26], all methods of $BN$, $LN$, and $IN$ learn a per-channel linear transform to compensate for the possible lost of representational ability:
$$
y_i = \gamma \hat{x}_i + \beta,
$$
where $\gamma$ and $\beta$ are trainable scale and shift (indexed by $i_C$ in all case, which we omit for simplifying notations).

Formally, a Group Norm layer computes $\mu$ and $\sigma$ in a set $\mathcal{S}_i$ defined as:
$$
\mathcal{S}_i = \{k~|~k_N=i_N, \lfloor \frac{k_C}{C/G} \rfloor=\lfloor \frac{i_C}{C/G} \rfloor\}.
$$
Here $G$ is the number of groups, which is a pre-defined hyper-parameter ($G=32$ by default). $C/G$ is the number of channels per group.

$GN$ becomes $LN$ if we set the group number as $G = 1$. **$LN$ assumes all channels in a layer make “similar contributions” [3]. Unlike the case of fully-connected layers studied in [3], this assumption can be less valid with the presence of convolutions, as discussed in [3].**

$GN$ becomes $IN$ if we set the group number as $G = C$ ($i.e.$, one channel per group). **But $IN$ can only rely on the spatial dimension for computing the mean and variance and it misses the opportunity of exploiting the channel dependence.**

### 4. Experiment

**$GN$ has $lower\ training\ error$ than $BN$, indicating that $GN$ is effective for easing $optimization$.** **The slightly higher validation error of $GN$ implies that GN loses some regularization ability of BN.** This is understandable, because BN’s mean and variance computation introduces uncertainty caused by the stochastic batch sampling, which helps regularization[26].


​				
​			
​		
​	


​			
​		
​	
​	