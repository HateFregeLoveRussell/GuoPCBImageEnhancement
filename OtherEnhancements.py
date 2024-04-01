import os.path
import cv2
from ReColor import match_intensity_and_color_YCrCb
import numpy as np
from GuoMethod import check_processed,log_processed
import argparse

def blur(inpPath, outPath, sigma):
    img = cv2.imread(inpPath, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), sigma)
    cv2.imwrite(img, outPath)
    match_intensity_and_color_YCrCb(inpPath,outPath, outPath)


def enhanceImageUsingSobel(img, betalist, ddepth=cv2.CV_64F):
    # Read the image
    # Calculate the gradient in x and y directions
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3)

    # Calculate the magnitude of the gradient
    grad_magnitude = cv2.sqrt(cv2.pow(grad_x, 2) + cv2.pow(grad_y, 2))

    # Initialize list to store enhanced images
    enhanced_images = []

    for beta in betalist:
        # Scale the gradient magnitude by beta
        scaled_grad_magnitude = cv2.multiply(grad_magnitude, beta)

        # Convert scaled gradient magnitude to same type as original image to add
        if ddepth == cv2.CV_64F:
            scaled_grad_magnitude = scaled_grad_magnitude.astype('float32')  # Assuming original image is 8-bit

        # Enhance the original image by adding the scaled gradient magnitude
        enhanced_img = cv2.add(img.astype('float32'), scaled_grad_magnitude)
        enhanced_img = np.clip(enhanced_img, 0, 255).astype('uint8')

        # Add the enhanced image to the list
        enhanced_images.append(enhanced_img)

    return enhanced_images

def computeSobelEnhancement(args, betaList):
    imglist = os.listdir(args.input_folder)
    sortedImgList = sorted(imglist)

    start = args.start_index
    end = args.stop_index if args.stop_index <= len(sortedImgList) else len(sortedImgList)

    csv_path = 'other_processed_images.csv'
    for beta in betaList:
        save_path = os.path.join(args.save_folder, str(round(beta, 2)))
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Folder '{save_path}' created successfully.")

    for n in range(start, end):
        imgName = sortedImgList[n]
        # print(imgName)
        imgPath = os.path.join(args.input_folder, imgName)
        print(imgPath)
        # Check each if beta exists if already computed remove from images betalist
        tmpBetaList = [round(beta, 2) for beta in betaList if not check_processed(imgName, beta, csv_path)]
        # print(tmpBetaList)
        tmpBetaList = np.array(tmpBetaList)
        # Process Image on Beta Values not yet Recorded
        if tmpBetaList.size != 0:
            img = cv2.imread(imgPath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output = enhanceImageUsingSobel(img, tmpBetaList)
            for i, beta in enumerate(tmpBetaList):
                output_image_path = os.path.join(args.save_folder, str(round(beta, 2)), imgName)
                cv2.imwrite(output_image_path, output[i])
                match_intensity_and_color_YCrCb(imgPath, output_image_path, output_image_path)
                log_processed(imgName, beta, csv_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="path to input files (jpg)",
            type=str)
    parser.add_argument("save_folder", help="path to where output images are saved. Please ensure that this folder is created before running this file.",
            type=str)
    parser.add_argument("start_index", help="index to start processing",
            type=int)
    parser.add_argument("stop_index", help="index to stop processing",
            type=int)
    args = parser.parse_args()
    print(args.input_folder)
    print(args.save_folder)
    computeSobelEnhancement(args, np.linspace(0, 1, 11))
