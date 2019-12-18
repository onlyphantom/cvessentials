# Kernels
## Definition
When performing an arithmetic computation on a given image, one approach is to apply said computation in a neighborhood-by-neighborhood manner. This approach is very braodly termed as a **convolution**. In other words, convolution is an operation between every part of an image ("pixel neighborhood") and an operator ("kernel")[^1][^2].

As the computation slides over each pixel neighborhood, we perform some arithmetic using the kernel, with the kernel typically being represented as a matrix or a fixed size array. 

This kernel describes how the pixels in that neighborhood are combined or transformed to yield a corresponding output.

- [ ] [Watch Kernel Convolution Explained Visually](https://www.youtube.com/watch?v=6m44SWnNPZs)
    <iframe width="560" height="315" src="https://www.youtube.com/embed/6m44SWnNPZs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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
To see a practical application of this, we'll use the `cv2.filter2D` to convolve over our image using the following kernel:

$$K = \frac{1}{5\cdot5} \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1  \end{bmatrix}$$

The kernel we specified above is equivalent to a _normalized box filter_ of size 5. Having watched the video earlier, you may intuit that the outcome of such a convolution is that each pixel in the input image is replaced by the average of the 5x5 pixels around it. You are in fact correct. If you are skeptical and would rather see proof of it, we'll see this in the following Code Illustrations section.

> **A Note on Terminology**
> When all we've been talking about is kernels, why is it that we're using the "filter" terminology in `opencv` code instead? That depends on the context. In the case of a convolutional neural network, _kernel_ and _filters_ are used interchangably: they both refer to the same thing.
> Some computer vision researchers have proposed to use a stricter definition, prefering to use the term "kernel" for a 2D array of weights, like our matrix above, and the term "filter" for the 3D structure of multiple kernels stacked together[^3], a concept we'll explore further in the Convolutional Neural Network part of this course.

#### Code Illustrations: Mean Filtering 
1. `meanblur_01.py` demonstrates the construction of a 5x5 mean average filter using `np.ones((5,5))/25`. Because every coefficient is basically the same, this merely replace the value of each pixel in our input image with the average of the values in its 5x5 neighborhood. 

```py
img = cv2.imread("assets/canal.png")
mean_blur = np.ones((5, 5), dtype="float32") * (1.0 / (5 ** 2))
smoothed_col = cv2.filter2D(img, -1, mean_blur)
```

Alternatively, we can be explicit in our creation of the 5x5 kernel using `numpy`'s array:
```py
mean_blur = np.array(
[[0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.04]])
```

2. To be fully convinced that the mean filtering operation is doing what we expect it to do, we can inspect the pixel values before- and after- the convolution, to verify that the math checks out by hand. We do this in `meanblur_02.py`.

    ```py
    img = cv2.imread("assets/canal.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f'Gray: {gray[:5, :5]}')
    # [[ 31  27  21  17  21]
    # [ 77  85  86  87  90]
    # [205 205 215 227 222]
    # [224 230 222 243 249]
    # [138 210 206 218 242]]
    for i in range(3):
        newval = np.round(np.mean(gray[:5, i:i+5]))
        print(f'Mean of 25x25 pixel #{i+1}: {np.int(newval)}')
    # output:
    # Mean of 25x25 pixel #1: 152
    # Mean of 25x25 pixel #2: 158
    # Mean of 25x25 pixel #3: 160
    # 
    ```
    The code above shows that the output of such a convolution operation beginning at the top-left region of the image would be 152. As we slide along the horizontal direction and re-compute the mean of the neighborhood, we get 158. As we slide our kernel along the horizontal direction for a second time and re-compute the mean of the neighborhood we obtain the value of 160. 
    
    If you prefer you can verify these values by hand, using the raw pixel values from `gray[:5, :5]` (5x5 top-left region of the image).

    ```py
    mean_blur = np.ones(KERNEL_SIZE, dtype="float32") * (1.0 / (5 ** 2))
    smoothed_gray = cv2.filter2D(gray, -1, mean_blur)
    print(f'Smoothed: {smoothed_gray[:5, :5]}')
    # output:
    # [[122 123 125 127 128]
    # [126 127 128 131 132]
    # [148 149 152 158 160]
    # [177 179 184 196 202]
    # [197 199 204 222 229]]
    ```
    Notice that from the output of our mean-filter, the first anchor (center of the neighborhood) has transformed from 215 to 152, and the one to the right of it has transformed from 227 to 158, and so on. The math does work out and you can observe the blur effect directly by running `meanblur02.py`.

3. As it turns out, `opencv` provides a set of convenience functions to apply filtering onto our images. All the three approaches below yield the same output, as can be verified from the output pixel values after executing `meanblur_03.py`:

    ```py
    # approach 1
    mean_blur = np.ones(KERNEL_SIZE, dtype="float32") * (1.0 / (5 ** 2))
    smoothed_gray = cv2.filter2D(gray, -1, mean_blur) 

    # approach 2
    smoothed_gray = cv2.blur(gray, KERNEL_SIZE)
    
    # approach 3
    smoothed_gray = cv2.boxFilter(gray, -1, KERNEL_SIZE)
    ```




## Role in Convolutional Neural Network

## Motivation

## Working with kernels in `opencv`


## Code Illustrations


## Summary and Key Points


## Learn-by-Building


## References
[^1]: Making your own linear filters, [OpenCV Documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html)

[^2]: Bradski, Kaehler, Learning OpenCV

[^3]: https://stats.stackexchange.com/a/366940