# Computer Vision (Chapter 1 to 3)

## Administrative Details
- Prerequisites:
    - Python 3
    - OpenCV
    - Numpy (automatically installed as dependency to opencv)
    - Tip: Use `pip install -r requirements.txt` to install from the requirement file (`requirements.txt`) in the repo. Get help from Teaching Assistant (Tommy) or myself before the beginning of the class

- Any code editor
    - Atom, VSCode, Sublime etc... 
    - Personally, I use VSCode (free)

- Materials
    - https://github.com/onlyphantom/cvessentials

- WiFi 
    - Network: Accelerice
    - Password: gapura19 

## Day 1
1. Synonymous role to data preprocessing
Data Analysis
    - Read data (usually using pandas as pd)
    - Inspect your data (dat.shape)
    - Data Preprocessing
        - Reshape, ...

2. Basic Routine
    ```
    import cv2
    import numpy as np

    img = cv2.imread("Desktop/family.png")
    print(img.shape) # output: (h, w, c)
    
    gray = cv2.cvtColor(img, cv2.BGR2GRAY)
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    ```

3. Affine Transformation
    ```
    import cv2
    import numpy as np

    img = cv2.imread("Desktop/family.png")
    (h, w, c) = img.shape
    print(f'Height: {h}; Width: {w}')

    gray = cv2.cvtColor(img, cv2.BGR2GRAY)

    # option 1: create 2x3 matrix
    mat = np.float32([[1, 0, 0], [0, 1, 0]])
    # option 2: ask for a 2x3 matrix
    mat = cv2.getRotationMatrix2D(center, angle=180, scale=1)
    mat = cv2.getAffineTransform(src, dst)

    transformed = cv2.warpAffine(gray, mat, dsize=(h,w))

    cv2.imshow("Transformed", transformed)
    cv2.waitKey(0)
    ```