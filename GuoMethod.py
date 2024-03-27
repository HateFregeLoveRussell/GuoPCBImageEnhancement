import numpy as np
import cv2



import cv2
import numpy as np
import csv
import time
from datetime import datetime
import cProfile
import pstats
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve
from concurrent.futures import ProcessPoolExecutor

sigma = 3
a=0.66
p_values = np.arange(- sigma, sigma + 1)
a_p = (1 - np.exp(-p_values / a)) / (1 + np.exp(-p_values / a))

super_p_values = np.arange(-2*sigma ,2*sigma + 1)
super_a_p = convolve(a_p, a_p, mode='full')
print(f'super_p_values shape: {super_p_values.shape}')
print(f'super_p_values = {super_p_values}')
print(f'super_a_p shape: {super_a_p.shape}')
print(f'super_a_p = {super_a_p}')

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


def sigmoidSingular(neighborhood, theta):
    x,y =  sigma + 1, sigma + 1
    x_new = x + p_values * np.cos(theta)
    y_new = y + p_values * np.sin(theta)
    # print(f'neighborhood shape: {neighborhood.shape}, theta: {theta}')
    # print(f'x_new: {x_new}, y_new: {y_new}')
    f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)

    output = np.dot(f_p,a_p)
    return output

def vectorSigmoid(neighborhood):
    x,y = 2*sigma + 1, 2*sigma + 1
    x_new = x + np.outer(np.cos(thetaList),super_p_values)
    y_new = y + np.outer(np.sin(thetaList),super_p_values)
    f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)
    return np.dot(f_p,super_a_p)


def listSigmoid(neighborhood):
    x, y = sigma + 1, sigma + 1

    # Calculate x_new and y_new for all theta values simultaneously
    # Shapes are expanded to (len(thetaList), len(p_values)) to perform the calculations for all thetas at once
    x_new = x + np.outer(np.cos(thetaList), p_values)
    y_new = y + np.outer(np.sin(thetaList), p_values)

    # Perform the bilinear interpolation for all thetas and their corresponding p_values in a single call
    # f_p will have a shape of (len(thetaList), len(p_values))
    f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)

    # Compute the activation 'a_p' for each 'p_value', this does not depend on theta and is 1D
    # Now, for the dot product, since f_p is 2D (theta x p_values) and a_p is 1D, numpy will handle this as
    # a row-wise dot product, giving us the output for each theta.
    outputs = np.dot(f_p, a_p)

    return outputs

def SecondPassSigmoidTransformPerTheta(theta, output_image, pad_width, height, width):
    print(f'Processing theta {theta}')
    padded_output_image = cv2.copyMakeBorder(output_image, pad_width + 1, pad_width + 1, pad_width + 1, pad_width + 1,
                                             cv2.BORDER_REPLICATE)
    final_output_image = np.zeros((height, width))

    # Buffer for rows of the padded output image, adapted for grayscale
    row_buffer = np.zeros((2 * sigma + 3, padded_output_image.shape[1]))

    # Initial population of the row buffer
    row_buffer[:2 * sigma + 3, :] = padded_output_image[:2 * sigma + 3, :]

    for y in range(height):
        if y % 10 == 0:  # Print progress every 10 lines
            print(f'Convolution progress for theta {theta}: {round(y / height * 100, 2)}%')

        for x in range(width):
            x_sample, y_sample = x + pad_width + 1, y + pad_width + 1

            # Extract the neighborhood from the row_buffer instead of the padded_output_image
            neighborhood = row_buffer[:, x_sample - sigma - 1:x_sample + sigma + 2]

            final_output_image[y, x] = sigmoidSingular(neighborhood, theta)

        # Update the row buffer for the next row
        if y < height - 1:  # Prevent index out of bounds on the last iteration
            row_buffer[:-1, :] = row_buffer[1:, :]
            row_buffer[-1, :] = padded_output_image[2 * sigma + 3 + y, :]

    return theta, final_output_image

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

    visualize('Intermediates', f'Sigmoid Image', sigmoid_image)
    output_image = np.zeros((len(betaList), height, width))
    for i, beta in enumerate(betaList):
        output_image[i] = image - beta * sigmoid_image
    return output_image

def DoubleSigmoidTransform(image, betaList):
    height, width = image.shape
    pad_width = sigma
    padded_image = cv2.copyMakeBorder(image, pad_width + 1, pad_width + 1, pad_width + 1, pad_width + 1,
                                      cv2.BORDER_REFLECT)
    output_images = np.zeros((len(thetaList), height, width))

    # Buffer for rows of the padded image, adjusted for a grayscale image
    row_buffer = np.zeros((2 * sigma + 3, padded_image.shape[1]))

    # Initial population of the row buffer
    row_buffer[:2 * sigma + 3, :] = padded_image[:2 * sigma + 3, :]

    print('FIRST PASS')
    for y in range(height):
        if y % 10 == 0:
            print(f'Convolution progress: {round(y / height * 100, 2)}%')

        for x in range(width):
            # Coordinates in the padded image
            x_sample, y_sample = x + pad_width + 1, y + pad_width + 1

            # Extract the neighborhood from the row_buffer
            neighborhood = row_buffer[:, x_sample - sigma - 1:x_sample + sigma + 2]

            outputs = listSigmoid(neighborhood)
            for i, output in enumerate(outputs):
                output_images[i, y, x] = output

        # After processing each row, update the row buffer for the next row
        if y < height - 1:  # Check to prevent index out of bounds on the last iteration
            # Scroll the row buffer: discard the top row and append the next row from the padded_image
            row_buffer[:-1, :] = row_buffer[1:, :]
            # Add the new bottom row to the buffer
            row_buffer[-1, :] = padded_image[2 * sigma + 3 + y, :]

    print('SECOND PASS')
    final_output_images = np.zeros_like(output_images)
    # Use ProcessPoolExecutor to parallelize the second pass
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(SecondPassSigmoidTransformPerTheta, theta, output_images[i], pad_width, height, width)
            for i, theta in enumerate(thetaList)]
        for future in futures:
            theta, processed_image = future.result()
            index = thetaList.index(theta)
            final_output_images[index] = processed_image
            # visualize('Intermediates', f'DoubleSigmoid{round(theta, 2)}', processed_image)
            # visualize('Intermediates', f'SingleSigmoid{round(theta, 2)}', output_images[index])

    # Calculate the absolute values of the images
    abs_images = np.abs(final_output_images)

    # Find the indices of the maximum values along the image dimension
    indices_of_max_abs = np.argmax(abs_images, axis=0)

    # Initialize an empty array to store the combined image
    sigmoid_image = np.zeros_like(final_output_images[0])

    # Iterate over each image index to populate the combined image
    for i in range(final_output_images.shape[0]):
        # Use the indices to mask out and select only the maximum values for each position
        mask = indices_of_max_abs == i
        sigmoid_image[mask] = final_output_images[i][mask]

    # visualize('Intermediates', f'Sigmoid Image', sigmoid_image)
    output_image = np.zeros((len(betaList), height, width))
    for i, beta in enumerate(betaList):
        output_image[i] = image - beta*sigmoid_image
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

def computeDoubleSigmoidTransform():
    imgPath = os.path.join('Inputs', 'rotated_test_PCB.jpg')
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # visualize('Intermediates','test_pcb', img)
    betaList = np.linspace(0,1,20)
    outputs = DoubleSigmoidSinglePassTransform(img, betaList)
    for i, beta in enumerate(betaList):
        output_image_path = os.path.join('Outputs', f'output_beta={round(beta,2)}.png')
        cv2.imwrite(output_image_path, outputs[i])

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    computeDoubleSigmoidTransform()  # Execute the function

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
