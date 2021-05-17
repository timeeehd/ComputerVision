# Assignment 1

To be able to run the imageSegmentation.py
use the following command:

python imageSegmentation.py image r t c basin search feature preprocessing

Please use the following as an example:
python imageSegmentation.py bulbasaur.jpg 20 0.01 4 True True 3D False

image: image with extension, i.e bulbasaur.jpg
r = radius of circle
t = threshold
c = search path variable
basin = Basin of attraction speedup
search = second speedup
feature = 3D or 5D
preprocessing: True or False, if you want to use Histogram Equalization
 
Note that the program needs to be run in the MeanShift folder
and that the program expects the images to be segmented to be located in the Data folder.

You can as well run the segmentation by running the imageSegmentation
in the IDE of your choice, at the bottom of the file,
you can adjust the variables accordingly
