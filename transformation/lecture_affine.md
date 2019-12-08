# Affine Transformation

## Definition
Any transformation that can be expressed in the form of a _matrix multiplication_ (linear transformation) followed by a _vector addition_ (translation). 

$T = A \cdot \begin{bmatrix} x \\ y \end{bmatrix} + B$ 

In which:

$A = \begin{bmatrix} a_{00} + a_{01} \\ a_{10} + a_{11} \end{bmatrix} $ and $B = \begin{bmatrix} b_{00} \\ b_{10} \end{bmatrix}$

When concatenated horizontally, this can be expressed in a larger Matrix:

$M = \begin{bmatrix} A & B \end{bmatrix} = \begin{bmatrix} a_{00} & a_{01} & b_{00} \\  a_{10} & a_{11} & b_{10} \end{bmatrix}$

By the definition above (_matmul_ + _vector addition_), affine transformation can be used to achieve:
- Scaling (linear transformation)
- Rotations (linear transformation)
- Translations (vector additions)

We represent an Affine Transformation using a **2x3 matrix**.

### Mathematical Definitions
Consider the goal of transforming a 2D vector $X = \begin{bmatrix} x \\ y \end{bmatrix}$ using $A$ and $B$ to obtain $T$, we can do it like such:

$T = A \cdot \begin{bmatrix} x \\ y \end{bmatrix} + B$ 

Or equivalently:

$T = M \cdot [x,y,1]^T = \begin{bmatrix} 
a_{00}x + a_{01}y + b_{00} \\ a_{10}x + a_{11}y + b_{10}  \end{bmatrix}$ 


## Motivation
1. Imaging systems in the real-world are often subject to **geometric distortion**. The distortion may be introduced by perspective irregularities, physical constraints (e.g camera placements), or other reasons. 

2. In the field of GIS (geographic information systems), routinely one would use affine transformation to "convert" geographic coordinates into screen coordinates such that it can **be displayed and presented** on our handheld / navigational devices. 

3. One may also overlay coordinate data on a spatial data that reference a different coordinate systems; Or to **"stitch" together** different sources of data using a series of transformation

These are but a handful of examples where one may expect to see routine use of affine transformations. If you're spending any amount of time in computer vision, a high degree of familiarity with these remapping routines in OpenCV will come in very handy.

In your learn-by-building section, you will find a less-than-perfectly-digitalized map, `belitung_raw.jpg`. Your job is to apply what you've apply the necessary affine transformation to correct its perspective distortion and the resize the map accordingly.

## Primer on Matrix Multiplication


## Examples and Illustrations

## Learn-by-Building
In the `homework` directory, you'll find a digital map `belitung_raw.jpg`. Your job is to apply what you've learned in this lesson to restore the map by correcting its skew and resize it appropriately. 