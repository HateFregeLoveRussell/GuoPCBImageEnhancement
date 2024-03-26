import os.path

import cv2
import numpy as np


def match_intensity_and_color_YCrCb(input_image_path, enhanced_image_path, output_image_path):
    # Read the color image
    color_image = cv2.imread(input_image_path)

    # Convert color image to YCbCr
    ycbcr_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)

    # Read the enhanced grayscale image
    grayscale_image = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)

    # Replace the Y channel with the grayscale image
    ycbcr_image[:, :, 0] = grayscale_image

    # Convert back to BGR color space
    merged_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)

    # Save the merged image
    cv2.imwrite(output_image_path, merged_image)


def match_intensity_and_color_lab(input_image_path, enhanced_image_path, output_image_path):
    # Read the color image
    color_image = cv2.imread(input_image_path)

    # Convert color image to Lab color space
    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2Lab)

    # Read the enhanced grayscale image
    grayscale_image = cv2.imread(enhanced_image_path, cv2.IMREAD_GRAYSCALE)

    # Replace the L channel with the grayscale image
    lab_image[:, :, 0] = grayscale_image

    # Convert back to BGR color space
    merged_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

    # Save the merged image
    cv2.imwrite(output_image_path, merged_image)

if __name__ == '__main__':
    enhanced_path = os.path.join('Outputs', 'output_beta=0.53.png')
    input_path = os.path.join('Inputs', 'rotated_test_PCB.jpg')
    output_path = os.path.join('ColoredOutputs', 'colored_output_beta=0.53_YCrCb.png')
    match_intensity_and_color_YCrCb(input_path,enhanced_path,output_path)