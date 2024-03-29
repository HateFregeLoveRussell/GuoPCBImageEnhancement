import argparse
import cv2
import numpy as np
import cProfile
import pstats
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve
from ReColor import match_intensity_and_color_lab

sigma = 3
a=0.66
p_values = np.arange(- sigma, sigma + 1)
a_p = (1 - np.exp(-p_values / a)) / (1 + np.exp(-p_values / a))

super_p_values = np.arange(-2*sigma ,2*sigma + 1)
super_a_p = convolve(a_p, a_p, mode='full')

thetaList = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def bilinear_interpolate_vectorized(image, x, y):
    """
    Vectorized bilinear interpolation for a set of x and y coordinates in an image.

    :param image: 2D array representing the image.
    :param x: 2D  or 1D array of x-coordinates (sub-pixel allowed).
    :param y: 2D or 1D array of y-coordinates (sub-pixel allowed).
    :return: 2D or 1D array of interpolated values at the specified coordinates.
    """
    # Floor of the coordinates
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    # Ceil of the coordinates (for the opposite corners of the interpolation square)
    x1 = x0 + 1
    y1 = y0 + 1

    # Grab the values at the corners of the interpolation squares
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    # Calculate the fractional parts of the original coordinates
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    # Perform the bilinear interpolation
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def vectorSigmoid(neighborhood):
    x,y = 2*sigma + 1, 2*sigma + 1
    x_new = x + np.outer(np.cos(thetaList),super_p_values)
    y_new = y + np.outer(np.sin(thetaList),super_p_values)
    f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)
    return np.dot(f_p,super_a_p)

def DoubleSigmoidSinglePassTransform(image, betaList):
    height, width = image.shape
    pad_width = 2*sigma
    padded_image = cv2.copyMakeBorder(image, pad_width + 1, pad_width + 1, pad_width + 1, pad_width + 1,
                                      cv2.BORDER_REFLECT)
    output_images = np.zeros((len(thetaList), height, width))

    # Buffer for rows of the padded image, adjusted for a grayscale image
    row_buffer = np.zeros((4 * sigma + 3, padded_image.shape[1]))

    # Initial population of the row buffer
    row_buffer[:4 * sigma + 3, :] = padded_image[:4 * sigma + 3, :]

    for y in range(height):
        if y % 10 == 0:
            print(f'Convolution progress: {round(y / height * 100, 2)}%')

        for x in range(width):
            # Coordinates in the padded image
            x_sample, y_sample = x + pad_width + 1, y + pad_width + 1

            # Extract the neighborhood from the row_buffer
            neighborhood = row_buffer[:, x_sample - 2*sigma - 1:x_sample + 2*sigma + 2]
            outputs = vectorSigmoid(neighborhood)
            for i, output in enumerate(outputs):
                output_images[i, y, x] = output

        # After processing each row, update the row buffer for the next row
        if y < height - 1:  # Check to prevent index out of bounds on the last iteration
            # Scroll the row buffer: discard the top row and append the next row from the padded_image
            row_buffer[:-1, :] = row_buffer[1:, :]
            # Add the new bottom row to the buffer
            row_buffer[-1, :] = padded_image[2 * 2*sigma + 3 + y, :]

        # Calculate the absolute values of the images
    abs_images = np.abs(output_images)

    # Find the indices of the maximum values along the image dimension
    indices_of_max_abs = np.argmax(abs_images, axis=0)

    # Initialize an empty array to store the combined image
    sigmoid_image = np.zeros_like(output_images[0])

    # Iterate over each image index to populate the combined image
    for i in range(output_images.shape[0]):
        # Use the indices to mask out and select only the maximum values for each position
        mask = indices_of_max_abs == i
        sigmoid_image[mask] = output_images[i][mask]

    # visualize('Intermediates', f'Sigmoid Image', sigmoid_image)
    output_image = np.zeros((len(betaList), height, width))
    for i, beta in enumerate(betaList):
        output_image[i] = image - beta * sigmoid_image
    return output_image

def visualize(folder, name, image, vmin=-600, vmax=1200):
    # Visualize the array as a heat map
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()  # Show color scale
    plt.title('Heat Map Visualization')
    heatName = name + 'Heatmap.png'
    imgName = name +'.png'
    # Save the heat map as a PNG file
    heatPath = os.path.join(folder, heatName)
    imgPath = os.path.join(folder, imgName)
    plt.savefig(heatPath)
    cv2.imwrite(imgPath, image)
    plt.close()  # Close the plot to free up memory

def computeDoubleSigmoidTransform(args, betaList):
    imglist = os.listdir(args.input_folder)
    sortedImgList = sorted(imglist)

    start = args.start_index
    end = args.stop_index if args.stop_index <= len(sortedImgList) else len(sortedImgList)

    for n in range(start, end):
        imgPath = os.path.join(args.input_folder, sortedImgList[n])
        output_image_path = os.path.join(args.save_folder, sortedImgList[n])

        print(imgPath)
        print(output_image_path)

        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = DoubleSigmoidSinglePassTransform(img, betaList)[0]
        cv2.imwrite(output_image_path, output)

        match_intensity_and_color_lab(imgPath, output_image_path, output_image_path)
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
    computeDoubleSigmoidTransform(args, [0.25])