import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
import time
import imageio
from tqdm import trange
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of input images')
    parser.add_argument('output', help='directory of output images')
    parser.add_argument('--size', help='size of input and output images', nargs=2, type=int, default=[300, 300])
    parser.add_argument('-n', help='number of augmented images for each original image', type=int, default=1)
    parser.add_argument('--no-keras', help="generate augmented images from keras or not", action='store_true')
    parser.add_argument('--no-crop', help="generate crop image or not", action='store_true')
    parser.add_argument('--crop-dim', help="specify 4 values: left, upper, right, lower according to PIL.Image.crop()", nargs=4, \
                                    type=int, default=[10, 10, 10, 10])
    args = parser.parse_args()
    IMG_SIZE = (args.size[0], args.size[1])
    image_generator = ImageDataGenerator(   rescale=(1/255),
                                            rotation_range=30,
                                            shear_range=0.03,
                                            height_shift_range=0.03,
                                            width_shift_range=0.03,
                                            zoom_range=[0.9, 1.25],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                        )
    image_iterator = image_generator.flow_from_directory(args.input, target_size=IMG_SIZE, batch_size=1, shuffle=None, \
                                                        class_mode='input', interpolation='bicubic')

    image_crop_generator = ImageDataGenerator(rescale=1/255)
    image_crop_iterator = image_crop_generator.flow_from_directory(args.input, target_size=IMG_SIZE, batch_size=1, shuffle=None, \
                                                        class_mode='input', interpolation='bicubic')
    for i in range(args.n):
        for i in trange(len(image_iterator.filenames)):
            if not args.no_keras:
                batchX, batchY = image_iterator.next()
                filename = os.path.split(image_iterator.filenames[i])[-1] + '_' + str(time.time()) +'.jpg'
                img = (batchX[0]*255).astype(np.uint8)
                imageio.imwrite(os.path.join(args.output, filename), img)

            if not args.no_crop:         
                batchX, batchY = image_crop_iterator.next()
                img = Image.fromarray((batchX[0]*255).astype(np.uint8), 'RGB')
                w, h = img.size
                filename_crop = os.path.split(image_crop_iterator.filenames[i])[-1] + '_crop.jpg'
                left, up, right, down = args.crop_dim
                img.crop((left, up, w - right, h - down)).save(os.path.join(args.output, filename_crop))