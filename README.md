# Essentials of Computer Vision  

![](assets/blurb.png)

A math-first approach to learning computer vision in Python. The repository will contain all HTML, PDF, Markdown, Python Scripts, data, and media assets (images or links to supplementary videos). If you wish to contribute, I need translations for Bahasa Indonesia. Please submit a Pull Request.

## Study Guide
### Chapter 1
- Affine Transformation
    - [Definition](transformation/lecture_affine.html#definition)
        - [Mathematical Definitions](transformation/lecture_affine.html#mathematical-definitions)
    - [Practical Examples](transformation/lecture_affine.html#practical-examples)
    - [Motivation](transformation/lecture_affine.html#motivation)
    - [Getting Affine Transformation](transformation/lecture_affine.html#getting_affine-transformation)
        - [Trigonometry Proof](transformation/lecture_affine.html#trigonometry-proof)
    - [Code Illustrations](transformation/lecture_affine.html#code-illustrations)
    - [Summary and Key Points](transformation/lecture_affine.html#summary-and-key-points)
    - Optional video 
        - [Rotation Matrix Explained Visually](https://www.youtube.com/watch?v=tIixrNtLJ8U)
            - [w/ Bahasa Indonesia voiceover](https://www.youtube.com/watch?v=pWfXR_HmyUw)
    - References and learn-by-building modules

### Chapter 2
- Kernel Convolutions
    - [Definition](edgedetect/kernel.html#definition)
        - Optional video
            -  [Kernel Convolutions Explained Visually](https://www.youtube.com/watch?v=WMmHcrX4Obg)
        - [Mathematical Definitions](edgedetect/kernel.html#mathematical-definitions)
        - [Padding](edgedetect/kernel.html#a-note-on-padding)
    - [Smoothing and Blurring](edgedetect/kernel.html#smoothing-and-blurring)
    - [A Note on Terminology](edgedetect/kernel.html#a-note-on-terminology)
        - Kernels or Filters?
        -   Correlations vs Convolutions?
    - [Code Illustrations: Mean Filtering](edgedetect/kernel.html#code-illustrations-mean-filtering)
    - [Role in Convolution Neural Networks](edgedetect/kernel.html#role-in-convolutional-neural-networks)
    - [Handy Kernels for Image Processing](edgedetect/kernel.html#handy-kernels-for-image-processing)
        - [Gaussian Filtering](edgedetect/kernel.html#gaussian-filtering)
        - [Sharpening Kernels](edgedetect/kernel.html#sharpening-kernels)
        - [Gaussian Kernels for Sharpening](edgedetect/kernel.html#approximate-gaussian-kernel-for-sharpening)
        - [Unsharp Masking](edgedetect/kernel.html#unsharp-masking)
    - [Summary and Key Points](edgedetect/kernel.html#summary-and-key-points)
    - References and learn-by-building modules

### Chapter 3
- Edge Detection
    - [Definition](edgedetect/edgedetect.html#definition)
    - [Gradient-based Edge Detection](edgedetect/edgedetect.html#gradient-based-edge-detection)
        - [Sobel Operator](edgedetect/edgedetect.html#sobel-operator)
            - [Discrete Derivative](edgedetect/edgedetect.html#intuition-discrete-derivative)
            - [Code Illustrations: Sobel Operator](edgedetect/edgedetect.html#code-illustrations-sobel-operator)
        - [Gradient Orientation & Magnitude](edgedetect/edgedetect.html#dive-deeper-gradient-orientation-magnitude)
    - [Image Segmentation](edgedetect/edgedetect.html#image-segmentation)
        - [Intensity-based Segmentation](edgedetect/edgedetect.html#intensity-based-segmentation)
            - [Simple Thresholding](edgedetect/edgedetect.html#simple-thresholding)
            - [Adaptive Thresholding](edgedetect/edgedetect.html#adaptive-thresholding)
        - [Edge-based Contour Estimation](edgedetect/edgedetect.html#edge-based-contour-estimation)
            - [Contour Retrieval and Approximation](edgedetect/edgedetect.html#contour-retrieval-and-approximation)
    - [Canny Edge Detector](edgedetect/edgedetect.html#canny-edge-detector)
        - [Edge Thinning](edgedetect/edgedetect.html#edge-thinning)
        - [Hysteresis Thresholding](edgedetect/edgedetect.html#hysteresis-thresholding)
    - References and learn-by-building modules

### Chapter 4
- Digit Classification
    - [A Note on Deep Learning](digitrecognition/digitrec.html#what-about-deep-learning)
        - [Why not MNIST?](digitrecognition/digitrec.html#region-of-interest)
    - Region of Interest
        - [ROI identification](digitrecognition/digitrec.html#selecting-region-of-interest)
        - [Arc Length and Area Size](digitrecognition/digitrec.html#arc-length-and-area-size)
            - [Dive Deeper: ROI](digitrecognition/digitrec.html#dive-deeper-roi)
        - [ROI extraction](digitrecognition/digitrec.html#roi-extraction)
    - [Morphological Transformations](digitrecognition/digitrec.html#morphological-transformations)
        - [Erosion](digitrecognition/digitrec.html#erosion)
        - [Dilation](digitrecognition/digitrec.html#dilation)
        - [Opening and Closing](digitrecognition/digitrec.html#opening-and-closing)
        - [Learn-by-building: Morphological Transformation](digitrecognition/digitrec.html#learn-by-building-morphological-transformation)
    - [Seven-segment display](digitrecognition/digitrec.html#seven-segment-display)
        - [Practical Strategies](digitrecognition/digitrec.html#practical-strategies)
            - [Contour Properties](digitrecognition/digitrec.html#contour-properties)
    - [References and learn-by-building modules](digitrecognition/digitrec.html#references)

## Approach and Motivation
The course is foundational to anyone who wish to work with computer vision in Python. It covers some of the most common image processing routines, and have in-depth coverage on mathematical concepts present in the materials: 
- Math-first approach
- Tons of sample python scripts (.py)
- Multimedia (image illustrations, video explanation, quiz)
- Practical tips on real-world applications

The course's **only dependency** is `OpenCV`. Getting started is as easy as `pip install opencv-contrib-python` and you're set to go.

- Question: What about deep learning libraries?

No; While using deep learning for images made for interesting topics, they are probably better suited as an altogether separate course series. This course series (tutorial series) focused on the **essentials of computer vision** and,
for pedagogical reasons, try not to be overly ambitious with the scope it intends to cover. 

There will be similarity in concepts and principles, as modern neural network architectures draw plenty of inspirations from "classical" computer vision techniques that predate it. By first learning how computer vision problems are solved, the student can compare that to the deep learning equivalent, which result in a more comprehensive appreciation of what deep learning offer to modern day computer scientists. 

## Course Materials Preview:
### Python scripts
![](digitrecognition/assets/croproi.gif)

### PDF and HTML
![](assets/ecv_caption.gif)


# Workshops
I conduct in-person lectures using the materials you find in this repository. These workshops are usually paid because there are upfront costs to afford a venue and crew. Not just any venue, but a learning environment that is fully equipped (audio, desks, charging points for everyone, massive screen projector, walking space fo teaching assistants, dinner). 

You can follow me [on Instagram](http://instagram.com/officialsamuel/) to be updated about the latest workshops.

### Introduction to AI in Computer Vision
- 4th January 2020, Jakarta
    - Kantorkuu, Citywalk sudirman, Jakarta Pusat
    - Time: 1300-1600
    - 3 hour
    - Fee: Free for Algoritma Alumni, 100k IDR for public

### Computer Vision: Principles and Practice
- 21st and 22nd January 2020, Jakarta
    - Accelerice, Jl. Rasuna Said, Jakarta Selatan
    - Time: 1830-2130
    - 6 Hour
    - Fee: Free for Algoritma Alumni, 1.5m IDR for public

- 24th and 25th Feburary 2020, Bangkok
    - JustCo, Samyan Mitrtown
    - Time: 1830-2130
    - 6 Hour
    - Fee: Free for Algoritma Alumni, 9000 THB for public


## Image Assets
- `car2.png`, `pen.jpg`, `lego.jpg` and `sudoku.jpg` are under Creative Commons (CC) license.

- `sarpi.jpg`, `castello.png`, `canal.png` and all other photography used are taken during my trip to Venice and you are free to use them. 

- All assets in Chapter 4 (the `digitrecognition` folder) are mine and you are free to use them.

- All other illustrations are created by me in Keynote. 

- Videos are created by me, and Bahasa Indonesia voice over on my videos is by [Tiara Dwiputri](https://github.com/tiaradwiputri)

## Badge of Completion
To earn a badge of completion, [attempt the quizzes](https://corgi.re/courses/onlyphantom/cvessentials) on https://corgi.re. [Corgi](https://corgi.re) is an aggregation tool for **co**u**r**ses on **gi**thub (hence the name) with a primary focus on data science and computer programming. 

Link to earn a badge: [Computer Vision Essentials | Corgi](https://corgi.re/courses/onlyphantom/cvessentials)

If you need help in the course, attend my in-person workshops on this topic (Computer Vision Essentials, free) throughout the course of the year.

## Find me
- [Facebook](https://www.facebook.com/onlyphantom)
- [Instagram](http://instagram.com/officialsamuel/)
- [LinkedIn](http://linkedin.com/in/chansamuel/)