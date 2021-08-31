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


class DataWIKI_MLM(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, pathWIKI, shuffle=False):
        'Initialization'

        fname = pathWIKI+".nlines.txt"
        # read buffer with number of lines from disk if exists
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf8') as fh:
                tot = fh.readlines()
                print("tot: ", tot)
                assert(len(tot)==1)
                self.tot = int(tot[0].strip())

        else:
            # compute buffer with number of lines and write to disk
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
        assert(len(idxs)==4)
        # input_ids_labels, attention_mask, (useless segment idx), input_ids_mlm
        #label idx_tokens idx_masks idx_segs idx_idxs
        #procsample = (np.fromstring(idxs[1],dtype=int,sep=' '), np.fromstring(idxs[2],dtype=int,sep=' '), np.fromstring(idxs[3],dtype=int,sep=' '), np.fromstring(idxs[4],dtype=int,sep=' '), np.fromstring(idxs[0],dtype=int,sep=' '))
        #procsample = (np.fromstring(idxs[0],dtype=int,sep=' '), np.fromstring(idxs[2],dtype=int,sep=' '), np.fromstring(idxs[3],dtype=int,sep=' '), np.fromstring(idxs[4],dtype=int,sep=' '), np.fromstring(idxs[0],dtype=int,sep=' '))

        # labels/targets , attention masks, segment ids (unused), input ids (unmasked)
        procsample = (np.fromstring(idxs[0],dtype=int,sep=' '), np.fromstring(idxs[1],dtype=int,sep=' '), np.fromstring(idxs[2],dtype=int,sep=' '), np.fromstring(idxs[3],dtype=int,sep=' '))

        # fix the ignore index to -100
        targets_mask = procsample[3] == procsample[0]
        procsample[0][targets_mask] = -100

        return procsample

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample = next(self.itera)
        #print(" sample, before: ", sample)
        sample = self.preprocess(sample)
        #print(" sample: ", sample)
        #print("DataWiki_MLM.py")
        #sys.exit(1)
        self.n_reads += 1
        return sample
