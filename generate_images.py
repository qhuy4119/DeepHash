import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import time
import matplotlib.pyplot as plt
import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='directory of input images')
    parser.add_argument('-o', '--output', help='directory of output images')
    parser.add_argument('--size', help='size of input and output images', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-b', '--batch', help="specify batch size", type=int, default=32)
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
    for i in range(len(image_iterator.filenames)):
        batchX, batchY = image_iterator.next()
        
        # plt.imshow(batchX[0])
        # plt.show()
        # image = Image.fromarray(batchX[0])
        filename = os.path.split(image_iterator.filenames[i])[-1] + '_' + str(time.time()) +'.jpg'
        imageio.imwrite(os.path.join(args.output, filename), batchX[0])
        # image.save()
        # print(img)

