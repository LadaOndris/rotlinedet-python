# Rotlinedet in Python

This repository contains a demonstration implementation of an algorithm 
called Rotlinedet for detecting lines in images. 

It finds lines in images 
by rotating the image by different angles.
For each rotation, it checks whether there is a sudden spike in the intensities of columns.

**This is only a demonstration application.**
It contains plots—column sum, peaks in intensities, and rotated image—
which helps in understanding and debugging the algorithm. 
**Production ready implementation is in a different repository.**

# Run Examples

Options:
* -i Path to the input image
* -r Rotation step in degrees (0.5 degree is a good option)



```
python3 main.py -i <image_path> -r 0.5
```