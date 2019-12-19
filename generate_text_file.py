import os
from os import listdir
from os.path import isfile, join
import argparse

def count_num_classes(fileList):
    num_classes = 0
    for i in range(len(fileList)):
        if fileList[i].count('.jpg') < 2:
            num_classes += 1
    return num_classes

def write_file(filePath, fileList, num_classes, prefix):
    with open(filePath, 'w') as outFile:
        i = 0
        current_class_index = 0
        while i < len(fileList):
            outFile.write(os.path.join(prefix, fileList[i]) + " ")
            write_one_hot_encoder(current_class_index, outFile, num_classes)
            j = i + 1
            while j < len(fileList):
                if fileList[j].startswith(fileList[i]):
                    outFile.write(os.path.join(prefix, fileList[j]) + " ")
                    write_one_hot_encoder(current_class_index, outFile, num_classes)
                    j += 1
                    continue
                else:
                    i = j
                    current_class_index += 1
                    break
            if j >= len(fileList):
                break
                              
def write_one_hot_encoder(class_index, outFile, num_classes):
    for i in range(num_classes):
        if i == class_index:
            outFile.write('1')
        else:
            outFile.write('0')
        outFile.write(' ')
    outFile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('d', help='specify the directory of image')
    parser.add_argument('--prefix', help='specify the name of folder to prefix when writing to text file. Default is: data', default='data')
    parser.add_argument('-o', '--output', help="specify the path of output file, including file name", default='train.txt')
    args = parser.parse_args()
    files = [f for f in listdir(args.d) if isfile(join(args.d, f))]
    files.sort()
    num_classes = count_num_classes(files)
    write_file(args.output, files, num_classes, args.prefix)