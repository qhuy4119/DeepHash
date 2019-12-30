import numpy as np
import logging 
import os

logging.basicConfig(level=logging.DEBUG, filename=os.path.expanduser('~/batch.log'), filemode='w')
class Dataset(object):
    def __init__(self, dataset, output_dim, num_similar_pairs=None, class_size=None):
        print ("Initializing Dataset")
        self.num_similar_pairs = num_similar_pairs
        self.class_size = class_size
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print ("Dataset already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self._train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        indexes = None
        if self._train:
            indexes = self._perm[start: start + self.num_similar_pairs]
            similar_images_indexes = self.get_similar_images_indexes(indexes, self.class_size)
            indexes = np.append(indexes, similar_images_indexes)
        else:
            indexes = self._perm[start:end]
        data, label = self._dataset.data(indexes)
        logging.info('The labels in this batch: %s\n\n', str(label))
        return (data, label)

    def get_similar_images_indexes(self, original_indexes, class_size: int):
        similar_indexes = [0] * len(original_indexes)
        for i in range(len(original_indexes)):
            start = (original_indexes[i] // class_size) * class_size
            end = ((original_indexes[i] // class_size) + 1) * class_size
            while True:
                num = np.random.randint(start, end) # generate random number in the half interval [start, end)
                if num != original_indexes[i]:
                    similar_indexes[i] = num  
                    break  
        return similar_indexes

    def feed_batch_output(self, batch_size, output):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start:end], :] = output
        return

    @property
    def output(self):
        return self._output

    @property
    def label(self):
        return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)

