import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
import argparse
from tqdm import trange
import csv
import os
from os import listdir
from os.path import isfile, join

def calculate_custom_metrics(hashes, metric):
    rows, cols = hashes.shape
    result = np.zeros((rows, rows), dtype=np.int16)
    for i in trange(rows):
        ith_col = np.zeros(rows, dtype=np.int32)
        for j in range(rows):
            if metric == 'hamming':
                ith_col[j] = np.count_nonzero(hashes[i]!=hashes[j])
            elif metric == 'dot':
                ith_col[j] = np.dot(hashes[i], hashes[j])
        result[:,i] = ith_col
    return result

def get_metrics_matrix(hash_file:str, metric='hamming'):
    a = np.load(hash_file)
    print("shape of hash array: ", a.shape)
    print('Calculating distance matrix')
    if metric == 'hamming' or metric == 'dot':
        return calculate_custom_metrics(a, metric)
    else:
        return pairwise_distances(a, metric=metric)

def write_csv_file(matrix, outFile, img_dir, metric):
    dirname = os.path.split(outFile)[0]
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)
    print("getting list of image filenames")
    files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    files.sort()
    files = files[:matrix.shape[0]]
    with open(outFile, 'w') as dist_file:
        print("start writing csv file")
        count_lines = 0
        dist_writer = csv.writer(dist_file, delimiter=',')
        headers = ['image_1', 'image_2', metric]
        dist_writer.writerow(headers)
        rows, cols = matrix.shape
        for col in trange(cols):
            for row in range(rows):
                if col == row:
                    continue
                dist_writer.writerow([files[col], files[row], matrix[row][col]])
                count_lines += 1
        print('Written %d lines to file %s' % (count_lines, outFile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hash', help='Path to npy file containing hash codes')
    parser.add_argument('img_dir', help='Specify directory of images from which hash codes were generated')
    parser.add_argument('out', help='Path to output csv file')
    parser.add_argument('--metric', help='Choose which metric to be calculated. Possible values are: hamming, dot, and metrics \
                                        from sklearn.metrics.pairwise_distances. Default is hamming'  , default='hamming')
    args = parser.parse_args()
    print(args)
    matrix = get_metrics_matrix(args.hash, args.metric)
    write_csv_file(matrix,args.out, args.img_dir, args.metric)


