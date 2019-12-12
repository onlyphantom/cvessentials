# Kernels
## Definition
When performing an arithmetic computation on a given image, one approach is to apply said computation in a neighborhood-by-neighborhood manner. This approach is very braodly termed as a **convolution**. In other words, convolution is an operation between every part of an image ("pixel neighborhood") and an operator ("kernel")[^1][^2].

As the computation slides over each pixel neighborhood, we perform some arithmetic using the kernel, with the kernel typically being represented as a matrix or a fixed size array. 

This kernel describes how the pixels in that neighborhood are combined or transformed to yield a corresponding output.

- [ ] [Watch Kernel Convolution Explained Visually](#)

### Mathematical Definitions
You will notice from the video that the output image now has a **shape that is smaller** than the original input. Mathematically, the shape of this output would be:

$$(\frac{X_m-M_i}{s_x}), (\frac{X_n-M_j}{s_y})$$

Where the input matrix has a size of $(X_m, X_n)$, the kernel $M$ is of size $(M_i, M_j)$, $s_x$ represents the stride over rows while $s_y$ represents the stride over columns. 

In the linked video, we are sliding the kernel on both the x- and y- direction by 1 pixel at a time after each computation, giving a value of 1 for $s_x$ and $s_y$. The input matrix in our video is of size 7, and our kernel is of size 3x3, giving us an output size of:

$$(\frac{7-3}{1}, \frac{7-3}{1})$$

Expressed mathematically, the full procedure as implemented in `opencv`looks like this for a convolution:

$H(x, y) = \sum^{M_i-1}_{i=0}\sum^{M_j-1}_{j=0} I(x+i-a_i, y+j-a_j)K(i,j)$

We'll see the step-by-step given a kernel represented by matrix M:

$$M = \begin{bmatrix} 1 & 2 & 0 \\ -1 & 3 & 0 \\ 0 & -1 & 0  \end{bmatrix}$$

1. Place the kernel anchor (in this case, $3$) on top of a determined pixel, with the rest of the kernel overlaying the corrresponding local pixesl in the image
    - Typically the kernel anchor is the _central_ of the kernel
    - Typically the "determined pixel" at the first step is the most upperleft region of the image

2. Multiply the kernel coefficients by the corresponding image pixel values and sum the result  

3. Replace the value at the location of the _anchor_ in the input image with the result

4. Repeat the process for all pixels by sliding the kernel across the entire image, as specified by the stride

## Smoothing and Blurring


#### Practical Examples

## Role in Convolutional Neural Network

## Motivation

## Working with kernels in `opencv`


## Code Illustrations


## Summary and Key Points


## Learn-by-Building


## References
[^1]: Making your own linear filters, [OpenCV Documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html)

[^2]: Bradski, Kaehler, Learning OpenCV