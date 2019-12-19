import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
import time
import imageio
from tqdm import trange
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='directory of input images')
    parser.add_argument('-o', '--output', help='directory of output images')
    parser.add_argument('--size', help='size of input and output images', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-n', help='number of augmented images for each original image', type=int, default=1)
    args = parser.parse_args()
    IMG_SIZE = (args.size[0], args.size[1])
    image_generator = ImageDataGenerator(   rescale=(1/255),
                                            rotation_range=30,
                                            shear_range=0.1,
                                            height_shift_range=0.1,
                                            width_shift_range=0.1,
                                            zoom_range=[0.9, 1.25],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                        )
    image_iterator = image_generator.flow_from_directory(args.input, target_size=IMG_SIZE, batch_size=1, shuffle=None, \
                                                        class_mode='input', interpolation='bicubic')
    for i in range(args.n):
        for i in trange(len(image_iterator.filenames)):
            batchX, batchY = image_iterator.next()
            filename = os.path.split(image_iterator.filenames[i])[-1] + '_' + str(time.time()) +'.jpg'
            imageio.imwrite(os.path.join(args.output, filename), (batchX[0]*255).astype(np.uint8))

