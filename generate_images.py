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
    parser.add_argument('--no-keras', help="will not generate images augmented by keras", action='store_true')
    parser.add_argument('--no-crop', help="will not generate cropped images", action='store_true')
    parser.add_argument('--no-rotate', help='will not generate rotated images', action='store_true')
    parser.add_argument('--crop-dim', help="specify 4 values: left, upper, right, lower according to PIL.Image.crop()", nargs=4, \
                                    type=int, default=[10, 10, 10, 10])
    parser.add_argument('--rotate', help="specify the angle to rotate in degrees counter clockwise. Default is 10", type=int, \
                                        default=10)
    args = parser.parse_args()
    IMG_SIZE = (args.size[0], args.size[1])
    image_keras_generator = ImageDataGenerator(   rescale=(1/255),
                                            rotation_range=30,
                                            shear_range=0.03,
                                            height_shift_range=0.03,
                                            width_shift_range=0.03,
                                            zoom_range=[0.9, 1.25],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                        )
    image_keras_iterator = image_keras_generator.flow_from_directory(args.input, target_size=IMG_SIZE, batch_size=1, shuffle=None, \
                                                        class_mode='input', interpolation='bicubic')

    image_PIL_tasks_generator = ImageDataGenerator(rescale=1/255)
    image_PIL_tasks_iterator = image_PIL_tasks_generator.flow_from_directory(args.input, target_size=IMG_SIZE, batch_size=1, shuffle=None, \
                                                        class_mode='input', interpolation='bicubic')
    for i in range(args.n):
        for i in trange(len(image_keras_iterator.filenames)):
            if not args.no_keras:
                batchX, batchY = image_keras_iterator.next()
                filename = os.path.split(image_keras_iterator.filenames[i])[-1] + '_' + str(time.time()) +'.jpg'
                img = (batchX[0]*255).astype(np.uint8)
                imageio.imwrite(os.path.join(args.output, filename), img)

            batchX, batchY = image_PIL_tasks_iterator.next()
            img = Image.fromarray((batchX[0]*255).astype(np.uint8), 'RGB')
            if not args.no_crop:                    
                w, h = img.size
                left, up, right, down = args.crop_dim
                save_filename = os.path.split(image_PIL_tasks_iterator.filenames[i])[-1] + '_crop_' + str(left) + '_' + \
                    str(up) + '_' + str(down) + '_' + str(right) + '.jpg'
                img.crop((left, up, w - right, h - down)).save(os.path.join(args.output, save_filename))
            if not args.no_rotate:
                save_filename = os.path.split(image_PIL_tasks_iterator.filenames[i])[-1] + '_rotate_' + str(args.rotate) + '.jpg'
                img.rotate(args.rotate, expand=True).save(os.path.join(args.output, save_filename))

