import os
from os import listdir
from os.path import isfile, join
import argparse
import logging
from collections import defaultdict
from tqdm import tqdm


def count_num_classes(fileList):
    num_classes = 0
    for i in range(len(fileList)):
        if fileList[i].count('.jpg') < 2:
            num_classes += 1
    return num_classes

def write_file(filePath, fileList, num_classes, prefix, num, logFile='log.txt'):
    logging.basicConfig(filename=logFile,level=logging.DEBUG, filemode='w')
    with open(filePath, 'w') as outFile:
        d = defaultdict(list)
        for f in fileList:
            origin_file = f.split("_")[0]
            d[origin_file].append(f)
        current_num_of_lines = 0
        for index, key in tqdm(enumerate(d)):
            for filename in d[key]:
                if num and index >= num:
                    return
                outFile.write(os.path.join(prefix, filename))
                outFile.write(" ")
                outFile.write(str(index) + "\n")
                current_num_of_lines += 1
                logging.info('writing line no %d' % current_num_of_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('d', help='specify the directory of image')
    parser.add_argument('--prefix', help='specify the name of folder to prefix when writing to text file. Default is: data', default='data')
    parser.add_argument('-o', '--output', help="specify the path of output file, including file name", default='train.txt')
    parser.add_argument('--log', help='specify path of log file', default='log.txt')
    parser.add_argument('--num', help='specify num of images', type=int)
    parser.add_argument('--query', action='store_true')
    args = parser.parse_args()
    files = [f for f in listdir(args.d) if isfile(join(args.d, f))]
    files.sort()
    if args.num:
        files = files[:args.num]
    num_classes = count_num_classes(files)
    if not args.query:
        print("Start writing file %s" % args.output)
        write_file(args.output, files, num_classes, args.prefix, args.num, args.log)
        print("Finished writing file")
    print('Num classes: ', num_classes)
    print('Num files:', len(files))