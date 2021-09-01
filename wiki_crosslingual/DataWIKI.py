# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime, timedelta
import time
import torch
from torch.utils import data
import numpy as np
import gzip
import random
import subprocess
import itertools
#logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class DataWIKI(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pathWIKI, shuffle=False):
        'Initialization'
        #if not tot_line is None:
        #    self.tot = tot_line #if you already know it, you can skip the computation
        #else:
        fname = pathWIKI+".nlines.txt"
        # read buffer with number of lines from disk if exists
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf8') as fh:
                tot = fh.readlines()
                print("tot: ", tot)
                assert(len(tot)==1)
                self.tot = int(tot[0].strip())

        else:
            self.tot = int(subprocess.check_output("zcat " + pathWIKI + " | wc -l", shell=True).split()[0])
            # write buffer with number of lines to disk
            with open(fname, 'w', encoding='utf8') as fh:
                fh.write(str(self.tot))

        # shuffle file before opening
        if shuffle:
            print("Shuffling dataset", end="...")
            start_t = time.time()
            self.shuffle(pathWIKI)
            end_t = time.time()
            print(" took %s."%(str(timedelta(seconds=end_t - start_t))))

        #print("TOT")
        print("num samples: " + str(self.tot))
        fileBIG = gzip.open(pathWIKI, 'rt')
        self.itera = itertools.cycle(enumerate(fileBIG))
        self.n_reads = 0

    def shuffle(self, pathWIKI):
        """ Decompress gzipped file, shuffle it, and recompress it """
        assert(pathWIKI.endswith("gz")), "Incorrect file format: %s"%pathWIKI

        temp_fname = pathWIKI + ".temp"
        try:
            subprocess.check_output("zcat " + pathWIKI + " | shuf | gzip > " + temp_fname, shell=True)
            subprocess.check_output("mv " + temp_fname + " " + pathWIKI, shell=True)
        except:
            raise Exception("An error happened while decompressing/shuffling/recompressing file %s, %s"%(
                pathWIKI, temp_fname))
        assert( os.path.isfile(pathWIKI) ), "Could not find file: %s"%pathWIKI

    def __len__(self):
        'Denotes the total number of samples'
        return self.tot

    def preprocess(self, sample):
        idxs = sample[1].strip().split("\t")
        #label idx_tokens idx_masks idx_segs idx_idxs
        #procsample = (idxs[1].split(" "), idxs[2].split(" "), idxs[3].split(" "), idxs[4].split(" "), idxs[0].split(" "))
        procsample = (np.fromstring(idxs[1],dtype=int,sep=' '), np.fromstring(idxs[2],dtype=int,sep=' '), np.fromstring(idxs[3],dtype=int,sep=' '), np.fromstring(idxs[4],dtype=int,sep=' '), np.fromstring(idxs[0],dtype=int,sep=' '))
        #targets_mask = (procsample)
        return procsample

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample = next(self.itera)
        sample = self.preprocess(sample)
        self.n_reads += 1
        return sample
