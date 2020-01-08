## Affine Transformation

1. Which of the following constructs the correct transformation matrix to perform a 2x scaling? 
    - [ ] `np.float32([[2, 0, 0], [0, 2, 0]])`
    - [ ] `np.float32([[0, 2, 0], [0, 2, 0]])`
    - [ ] `np.float32([[2, 2, 2], [0, 0, 0]])`
    - [ ] `np.float32([[2, 1, 1], [1, 2, 1]])`

2. In the case on a 333x333 input image, with a strides of 1 using a kernel of size 5*5, what is the amount of zero-padding you should add to the borders of your image such that the output image is also 333x333?
    - [ ] 1
    - [ ] 2
    - [ ] 3
    - [ ] No zero-padding

## Kernels and Convolution

3. For an input image of size 140W (Width) x 600H (Height), supposed we perform a convolution with slide S=1 using a filter of size 7W x 7H and two pixels of constant-padding (padding our image with a constant value of 5), what would the dimension of our image be?
    - [ ] 135 Width x 595 Height
    - [ ] 140 Width x 600 Height
    - [ ] 138 Width x 598 Height
    - [ ] None of the answers above 

## Tresholding Edge Detection
4. In an image with lighting conditions that result in some parts of the image being shaded differently than the others, which of the thresholding techniques may yield a more robust output?
    - [ ] Pixel-intensity based thresholding
    - [ ] Otsu's global thresholding method
    - [ ] Adaptive thresholding

5. We want to retrieve only the extreme outer contours. We do not need to store all the boundary points to minimise redundancy and save memory requirements. Which are the values to be passed into the findContours() function?
    - [ ] RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
    - [ ] RETR_EXTERNAL, CHAIN_APPROX_NONE
    - [ ] RETR_OUTER, CHAIN_APPROX_SIMPLE
    - [ ] RETR_OUTER, CHAIN_APPROX_NONE
    - [ ] RETR_LIST, CHAIN_APPROX_NONE

6. The function call cv2.Canny(img, 50, 180) will determine which of the intensity gradients as definite edges?
    - [ ] 40
    - [ ] 100
    - [ ] 200

7. Which of the following is NOT part of the Canny Edge procedure?
    - [ ] Compute gradient in each direction
    - [ ] Suppress edges that are non-maximal
    - [ ] Discard pixels that are more likely noise than true edges
    - [ ] Retrieve only the extreme outer contours from the edges