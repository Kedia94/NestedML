# NestedNet TensorFlow Implementation
Copied code. Original code is https://github.com/niceday15/nested-network-cifar100.git

TensorFlow implementation of the folloiwng paper:

Eunwoo Kim, Chanho Ahn, and Songhwai Oh, "NestedNet: Learning Nested Sparse Structures in Deep Neural Networks", CVPR, 2018. 
(arXiv: https://arxiv.org/abs/1712.03781)

We currently release a demo code of an adaptive deep compression experiment for the CIFAR-100 dataset, 
where NestedNet consists of three nested levels by using channel scheduling. The code is based on the (wide) residual network, where the wide residual network with wide_factor=1 is equivalent to residual network. 
The code can be applied to the CIFAR-10 dataset.

More codes for other experimental scenarios will be provided.
