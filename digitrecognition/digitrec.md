# Background
In Chapter 4: Digit Recognition, we'll add a few new techniques to our image processing toolset by attempting to build a digit recognition pipeline from start to finish. Throughout the exercise, we will get to practice the image preprocessing tricks we've picked up from previous chapters:
- Image manipulations such as resizing, cropping, rotation, color conversion  
- Blurring and sharpening operations
- Thresholding and Edge Detection
- Contour approximation

New method and strategies that you'll be learning include:
- Drawing operations (rectangles, text) on our image  
- Region of interest and bounding rectangles
- Morphological transformations
- The Seven-Segment Display 

## What about Deep Learning?
To be clear, specialised deep learning libraries that have sprung out in recent years are a lot more robust in their approach. By utilizing machine learning principles (cost function, gradient descent etc), these specialised libraries can handle highly complex object recognition and OCR (optical character recognition) tasks at the cost of brute computing power.

The overarching motivation of this free course however, was to make clear to beginners what constitutes artificial intelligence, and to illustrate the principle benefits of machine learning. I try to achieve that by demonstrating -- over multiple chapters of this course -- how computer visions were traditionally, or rather "classically", performed prior to the emergence of deep learning. 

By learning the classical approaches to computer vision, the student (you) can compare the effort it takes to hand-tuning parameters and this adds a new dimension of appreciation towards self-learning methods that we'll discuss in the near future.

## Region of Interest
Do a quick google search on "digit recognition" and it's hard to find an introductory deep learning course that **doesn't use** the famous MNIST (Modified National Institute of Standards and Technology)[^1] database. This is a handwritten digit database that has long become the _de facto_ in pretty much any machine learning tutorials:

![](assets/mnist.png)

But I'd argue, that for a budding computer vision developer, your learning objectives are better served by taking a different approach. 

By choosing real life images, you are confronted with a few more key challenges that are not present from using a well-curated database such as MNIST. These challenges present new opportunities to learn about key concepts such as **region of interest**, and **morphological operations**, that you will come to rely upon greatly in the future. 

First, take a look at 4 real-life pictures of security tokens issued by banks and institutional agencies (left-to-right: Bank Central Asia, DBS, OCBC Bank, OneKey for Singapore Government e-services): 

![](assets/securitytokens.png)

Notice how noisy these images are, as each image is shot with a different background, different lighting conditions, each token is of a different size and shape, and the different colors etc. 

Your task, as a computer vision developer, is to develop a pipeline that, in each phase, 

### Seven-segment display
The seven-segment display (known also as "seven-segment indicator") is a form of electronic display device for displaying decimal numerals[^2] widely used in digital clocks, electronic meters, calculators and banking security tokens.

## Mathematical Definition


# References
[^1]: LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86, 2278–2324
[^2]: Seven-segment display, Wikipedia


## References
