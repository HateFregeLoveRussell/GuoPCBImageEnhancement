import numpy as np
import cv2



import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def bilinear_interpolate_vectorized(image, x, y):
    """
    Vectorized bilinear interpolation for a set of x and y coordinates in an image.

    :param image: 2D array representing the image.
    :param x: 1D array of x-coordinates (sub-pixel allowed).
    :param y: 1D array of y-coordinates (sub-pixel allowed).
    :return: 1D array of interpolated values at the specified coordinates.
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

def listsigmoid(sigma, x, y, neighborhood, thetaList, a=0.66):
    outputs = np.zeros_like(thetaList)
    for i, theta in enumerate(thetaList):
        p_values = np.arange(- sigma, sigma+1)
        x_new = x + p_values * np.cos(theta)
        y_new = y + p_values * np.sin(theta)
        f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)
        a_p = (1 - np.exp(-p_values / a)) / (1 + np.exp(-p_values / a))
        outputs[i] = np.dot(f_p,a_p)
    return outputs
def sigmoidSingular(sigma, x,y, neighborhood, theta, a = 0.66):
    p_values = np.arange(- sigma, sigma)
    x_new = x + p_values * np.cos(theta)
    y_new = y + p_values * np.sin(theta)
    f_p = bilinear_interpolate_vectorized(neighborhood, x_new, y_new)
    a_p = (1 - np.exp(-p_values / a)) / (1 + np.exp(-p_values / a))
    output = np.dot(f_p,a_p)
    return output

def DoubleSigmoidTransform(image, sigma, thetaList, betaList,a=0.66 ):
    height, width = image.shape[:2]
    pad_width = sigma
    padded_image = cv2.copyMakeBorder(image, pad_width+1, pad_width+1, pad_width+1, pad_width+1,
                                      cv2.BORDER_REFLECT)

    # Adjust the output array to hold an image for each theta
    output_images = np.zeros((len(thetaList), height, width))
    # Iterate over each pixel in the image
    print(f'FIRST PASS')
    for y in range(height):

        if y %10 == 0:
            print(f'Convolution progress: {round(y / height * 100, 2)}%')
        for x in range(width):
            # Extract the neighborhood for this pixel
            neighborhood = padded_image
              # Adjust slicing to capture the correct neighborhood size
            # Compute the list of outputs for the current pixel across all thetas
            outputs = listsigmoid(sigma, x + pad_width, y + pad_width, neighborhood, thetaList, a)
            # Assign the outputs to the corresponding pixel location in each output image
            for i, output in enumerate(outputs):
                output_images[i, y, x] = output
    print('SECOND PASS')
    final_output_images = np.zeros_like(output_images)
    for i, theta in enumerate(thetaList):
        print(f'Processing theta {theta}')
        padded_output_image = cv2.copyMakeBorder(output_images[i], pad_width + 1, pad_width + 1, pad_width + 1, pad_width + 1,
                           cv2.BORDER_REPLICATE)

        for y in range(height):
            if y % 10 == 0:  # Print progress every 10 lines
                print(f'Convolution progress: {round(y / height * 100, 2)}%')
            for x in range(width):
                # Directly use the output image from the first pass, no additional padding needed
                neighborhood = padded_output_image
                final_output_images[i, y, x] = sigmoidSingular(sigma, x + pad_width, y + pad_width , neighborhood, theta, a)
        visualize('Intermediates',f'DoubleSigmoid{round(theta,2)}', final_output_images[i])
        visualize('Intermediates', f'SingleSigmoid{round(theta, 2)}', output_images[i])

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

    visualize('Intermediates', f'Sigmoid Image', sigmoid_image)
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
    sigma = 3
    a = 0.66
    betaList = np.linspace(0,1, 20)
    thetaList = [0 ,np.pi/4, np.pi/2, 3*np.pi/4]
    img = cv2.imread('rotated_test_PCB.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    visualize('Intermediates','test_pcb', img)
    outputs = DoubleSigmoidTransform(img, sigma,thetaList, betaList, a)
    for i, beta in enumerate(betaList):
        output_image_path = os.path.join('Outputs', f'output_beta={round(beta,2)}.png')
        cv2.imwrite(output_image_path, outputs[i])

if __name__ == '__main__':
    # x = 200
    # y = 300
    # sigma = 3
    # img = cv2.imread('test_PCB.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(sigmoidSingular(sigma,200,300,img,0))
    #
    # print(sigmoidSingular(sigma,300,200,img,0))
    computeDoubleSigmoidTransform()