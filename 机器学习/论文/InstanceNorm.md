# Instance Normalization: The Missing Ingredient for Fast Stylization

> It this paper we revisit the fast stylization method introduced in Ulyanov et al. (2016). We show how a small change in the stylization architecture results in a significant qualitative improvement in the generated images. The change is limited to swapping batch normalization with instance normalization, and to apply the latter both at training and testing times. The resulting method can be used to train high-performance architectures for real-time image generation. 

$Instance \ Normalization$ 这篇论文的提出，根本目的是提升图像风格迁移的质量。

经过风格迁移的图片，同时符合风格图片（$style \ image$） 和内容图片 $content \ image$ 的统计分布。风格特征往往从较浅层特征中提取，并且在空间位置上是均匀的，可以代表风格图像的“纹理”；而内容特征则从较深层的特征提取，并且保留了空间位置的信息，即内容图像的“结构”。

$Gatys$ 等人得方法已经能产生比较好的结果，但计算效率低下。$Ulyanov$ 和 $ Johnson$，通过学习等效的前馈生成网络试图解决效率低下的问题，但结果质量没有 $Gatys$ 那个较慢方法好。

本论文在 $Ulyanov$ 和 $Johnson$ 的前馈生成网络基础上，做一个小的改进，即将 $Batch \ Normalization$ 替换为 $Instance \ Normalization$ ，使结果有了很大提升。





------

**The stylized image matches simultaneously selected statistics of the style image and of the content image.**

**The style statistics are extracted from shallower layers and averaged across spatial locations whereas the content statistics are extracted form deeper layers and preserve spatial information.** In this manner, the style statistics capture the “texture” of the style image whereas the content statistics capture the “structure” of the content image.

The key idea (section 2) is to replace batch normalization layers in the generator architecture with instance normalization layers, and to keep them at test time (as opposed to freeze and simplify them out as done for batch normalization).


​			
​		
​	


​		
​	


​				
​			
​		
​	