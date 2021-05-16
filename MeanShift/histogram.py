import cv2


def histogram(image):
    """
    Does histogram equalization by changing to color scheme to BGR, then to YUV.
    After doing the histogram equalization in the YUV color scheme, transform back to
    BGR and then RGB. Output this image.
    Hints have been used by looking at this link:
    https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

    Args:
        image: image in RGB in the data folder.

    Output:
        Equalized image in RGB
    """
    img = cv2.imread(f"./Data/{image}.jpg", cv2.IMREAD_COLOR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # transform yuv to bgr
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # transform bgr to rgb
    img_output_rgb = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    return img_output_rgb
