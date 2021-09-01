import sys
import argparse
import glob
import logging
import os
import random
import time
from datetime import datetime, timedelta
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch
import itertools
import re

from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import BertModel, XLMModel, XLMRobertaModel, XLMRobertaForMaskedLM, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup, PreTrainedModel

# We broke up the model in two classes: BERT backbone can be DataParallel'ed, HyperlinkPredictionHead can't
from wiki_crosslingual.Token_HyperlinkPredictionHead import Token_HyperlinkPredictionHead
from wiki_crosslingual.ConcatCLS_HyperlinkPredictionHead import ConcatCLS_HyperlinkPredictionHead
from wiki_crosslingual.ReplaceCLS10Percent_HyperlinkPredictionHead import ReplaceCLS10Percent_HyperlinkPredictionHead
from wiki_crosslingual.MMModel_WithDataParallel import MMModel_WithDataParallel
from wiki_crosslingual.MMModel_ConcatCLS_WithDataParallel import MMModel_ConcatCLS_WithDataParallel
from wiki_crosslingual.MMModel_ReplaceCLS10Percent_WithDataParallel import MMModel_ReplaceCLS10Percent_WithDataParallel
from wiki_crosslingual.DataWIKI import DataWIKI
from wiki_crosslingual.DataWIKI_MLM import DataWIKI_MLM

from wiki_crosslingual.sampling_probability import get_sampling_probability_from_counts


MODEL_CLASSES = {
    "bert": (BertModel, None),
    "xlm": (XLMModel, None),
    "xlm-roberta": (XLMRobertaForMaskedLM, None)
    #"xlm-roberta": (XLMRobertaModel, XLMRobertaForMaskedLM)
}

HYPERLINK_MODEL_TYPE = {
    "baseline": (MMModel_WithDataParallel, Token_HyperlinkPredictionHead), # this option does not matter when we are working with a baseline XLM-R
    "standard": (MMModel_WithDataParallel, Token_HyperlinkPredictionHead),
    "concat_cls": (MMModel_ConcatCLS_WithDataParallel, ConcatCLS_HyperlinkPredictionHead),
    "replace_cls": (MMModel_ReplaceCLS10Percent_WithDataParallel, ReplaceCLS10Percent_HyperlinkPredictionHead),
    #"standard": (MMModel),
    #"concat_cls": (MMModel_ConcatCLS),
    #"replace_by_cls": (MMModel_ReplaceCLS10Percent),
}

LR_SCHEDULERS = ["constant", "linear_with_warmup"]

#class MyDataParallel(nn.DataParallel):
#    def __getattr__(self, name):
#        try:
#            return super().__getattr__(name)
#        except AttributeError:
#            return getattr(self.module, name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

def load_model_reads(path_to_model):
    # number of reads for each dataset is stored manually in path_to_model  + "datasets_nreads.txt'
    fname = path_to_model+"_datasets_nreads.txt"
    assert( os.path.isfile(fname) ), "File not found: %s"%fname
    n_reads = np.loadtxt( fname )
    n_reads = n_reads.tolist()
    n_reads = [int(n) for n in n_reads]
    return n_reads

def datasets_fast_forward(nreads, training_generator, training_generator_foreign_list):
    # English iterator
    iteraEN = itertools.cycle(enumerate(training_generator))
    # all other languages' iterators
    itera_foreign_list = [itertools.cycle(enumerate(training_generator_foreign))
                          for training_generator_foreign in training_generator_foreign_list]
    itera_all = [iteraEN] + itera_foreign_list

    # for each dataset (language)
    for dataset_idx in range(len(nreads)):
        start_t = time.process_time()
        print("Fast-forwarding dataset idx %i ..."%dataset_idx)
        for curr_read in range(nreads[ dataset_idx ]):
            itera_curr = itera_all[ dataset_idx ]
            step, batch = next(itera_curr)
        end_t = time.process_time()
        elapsed_t = end_t - start_t
        print("...fast-forward by %i reads in %s."%(
            curr_read, str(timedelta(seconds=elapsed_t))))


def load_model(path_to_model, model_backbone, model_softmax):
    checkpoint_backbone, checkpoint_softmax = torch.load( path_to_model )
    model_backbone.load_state_dict( checkpoint_backbone )
    model_softmax.load_state_dict( checkpoint_softmax )
    return model_backbone, model_softmax


def get_train_fnames_from_language_codes_file(language_codes_file, path_to_files):
    fnames = []
    fname_template = os.path.join( path_to_files, "wiki.%s.links.top250K.idx.wordtok.xlmr.txt.gz" )
    with open(language_codes_file, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().lower()
            fname = fname_template%line
            assert( os.path.isfile(fname) ), "Could not locate file: %s"%fname
            fnames.append( fname )
    return fnames


def get_train_fnames_mlm_from_language_codes_file(language_codes_file, path_to_files):
    fnames = []
    fname_template = os.path.join( path_to_files, "wiki.%s.links.top250K.idx.wordtok.xlmr.mlm.txt.gz.shuf.gz" )
    with open(language_codes_file, 'r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip().lower()
            fname = fname_template%line
            assert( os.path.isfile(fname) ), "Could not locate file: %s"%fname
            fnames.append( fname )
    return fnames


def main():
    parser = argparse.ArgumentParser()

    # Uncomment below to debug
    #SAVE_EVERY = 1000
    SAVE_EVERY = 100000

    # Required parameters
    parser.add_argument(
        "--train_path_english",
        default=None,
        type=str,
        required=False,
        help="Path to the English input data file name (to train the Wiki hyperlink prediction).",
    )

    parser.add_argument(
        "--train_path_english_mlm",
        default=None,
        type=str,
        required=False,
        help="Path to the English input data file name (to train the MLM loss).",
    )

    parser.add_argument(
        "--train_path_other_languages",
        default=None,
        type=str,
        required=False, #nargs="+",
        help="The input data dir. Should be a path to the directory where the Wiki hyperlink prediction tgz data files are located.",
    )

    parser.add_argument(
        "--train_path_other_languages_mlm",
        default=None,
        type=str,
        required=False, #nargs="+",
        help="The input data dir. Should be a path to the directory where the MLM tgz data files are located.",
    )

    parser.add_argument(
        '--language_codes_file',
        default=None,
        required=False,
        type=str,
        help="Text file containing language codes for all non-English languages to use for Wiki hyperlink prediction training, one per line.",
    )

    parser.add_argument(
        '--language_codes_file_mlm',
        default=None,
        required=False,
        type=str,
        help="Text file containing language codes for all non-English languages to use for MLM training, one per line.",
    )

    parser.add_argument(
        "--hyperlink_model_type",
        default=None,
        type=str,
        required=False,
        choices=list(HYPERLINK_MODEL_TYPE.keys()),
        help="Model hyperlink type selected in the list: " + ", ".join(HYPERLINK_MODEL_TYPE.keys()),
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )

    parser.add_argument(
        "--path_save",
        default=None,
        type=str,
        required=True,
        help="""Path to pre-trained model or shortcut name selected in the list.
        If --path_save is not a valid file, it will be interpreted as a file name prefix to which we append the iteration number before saving checkpoints.
        If --path_save is a valid file, we will continue training from the checkpoint.""",
    )

    parser.add_argument(
        "--max_seq_length",
        default=254,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--hidden_dim",
        default=1024,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--force_shuffle",
        action='store_true',
        help="Whether to force the shuffling of the GZIPed datasets. Cannot be set at the same time as '--force_no_shuffle'",
    )

    parser.add_argument(
        "--force_no_shuffle",
        action='store_true',
        help="Whether to force NOT shuffling of the GZIPed datasets. Cannot be set at the same time as '--force_shuffle'",
    )

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=6250, type=int, help="The number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--num_training_steps", default=31250, type=int, help="The total number of training steps to use in the learning rate scheduler.")
    parser.add_argument("--learning_rate_scheduler", default="constant", type=str, choices=LR_SCHEDULERS,
            help="The learning rate scheduler to use for training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    #parser.add_argument(
    #    "--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform."
    #)
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0],
            help="GPU ids. Defaults to [0]. Can take multiple ids for DataParallel training.")
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY, help="Save a model checkpoint every --SAVE_EVERY forward passes (NOT model updates!).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
            help="Number of forward-passes on minibatches to do before running a backward pass and updating network weights.")
    parser.add_argument('--train_with_mlm', action="store_true", help="If this flag is set, we perform MLM and Wiki hyperlink prediction.")
    parser.add_argument('--baseline', action="store_true", help="If this flag is set, we only perform MLM (baseline).")
    parser.add_argument('--english_only', action="store_true", help="If this flag is set, we only train on English data.") 
    #parser.add_argument('--two_step_training', action="store_true",
    #        help="""First only train adaptive softmax parameters until convergence (keeping the rest of the model frozen).
    #                After it has converged, finetune the entire model jointly,
    #                where the number of finetuning layers is defined according to the `--finetune_*_layers` flag),
    #                and the warmup / learning rate scheduler is set according to their own flags.""")

    finetune_action = parser.add_mutually_exclusive_group(required=True)
    finetune_action.add_argument('--finetune_layer_norm_layers', action="store_true",
            help="If this flag is set, we finetune all layer norm layers in XLM-R, including input embedding and attention in transformer layers.")
    finetune_action.add_argument('--finetune_two_transformer_layers', action="store_true",
            help="If this flag is set, we finetune the top two transformer layers of XLM-R.")
    finetune_action.add_argument('--finetune_half_transformer_layers', action="store_true",
            help="If this flag is set, we finetune the top half transformer layers of XLM-R.")
    finetune_action.add_argument('--finetune_all_transformer_layers', action="store_true",
            help="If this flag is set, we finetune all the transformer layers of XLM-R (but not word embeddings).")
    finetune_action.add_argument('--finetune_all_layers', action="store_true",
            help="If this flag is set, we finetune all the parameters of XLM-R.")
    finetune_action.add_argument('--finetune_no_layers', action='store_true',
            help="Do not finetune XLM-R at all. Set this flag to train only the adaptive softmax parameters.")
  
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.finetune_no_layers:
        assert(not args.baseline), "When finetuning no layers one cannot have a baseline model, since there would be no parameters to be trained."

    if not args.baseline:
        assert(not args.hyperlink_model_type is None), "args.hyperlink_model_type must be set when not training a baseline!"
        assert(os.path.isfile(args.train_path_english)), "Could not find file: %s"%args.train_path_english

    if not args.english_only:
        if not args.baseline:
            assert(os.path.isfile(args.language_codes_file)), "File not found: %s"%args.language_codes_file
            assert(os.path.isdir(args.train_path_other_languages)), "Data directory for non-English languages not found: %s"%args.train_path_other_languages

        assert(os.path.isfile(args.language_codes_file_mlm)), "File not found: %s"%args.language_codes_file_mlm

    if args.train_with_mlm:
        assert(os.path.isfile(args.train_path_english_mlm)), "Could not find file: %s"%args.train_path_english_mlm
        if not args.english_only:
            assert(os.path.isdir(args.train_path_other_languages_mlm)), "Could not find directory: %s"%args.train_path_other_languages_mlm

    if args.force_shuffle or args.force_no_shuffle:
        assert( not (args.force_shuffle and args.force_no_shuffle) ), "Exclusive options: --force_shuffle and --force_no_shuffle"

    pathTrainEnglish = args.train_path_english
    if not args.english_only and not args.baseline:
        pathTrainOtherLanguages = get_train_fnames_from_language_codes_file( args.language_codes_file, args.train_path_other_languages )
    if args.train_with_mlm:
        pathTrainEnglish_mlm = args.train_path_english_mlm
        if not args.english_only:
            pathTrainOtherLanguages_mlm = get_train_fnames_mlm_from_language_codes_file( args.language_codes_file_mlm, args.train_path_other_languages_mlm )

    if not os.path.isfile(args.path_save) and not os.path.islink(args.path_save):
        assert( args.path_save.endswith(".") ), "File name prefix should end withh a '.'. Received: %s"%args.path_save

    pathSAVE = args.path_save
    BATCH_SIZE = args.batch_size  # 4
    #num_train_epochs = args.num_train_epochs  # 10
    num_train_epochs = 1
    # max_seq_length = args.max_seq_length# 200

    model_name = args.model_name_or_path  # "xlm-roberta-base"
    # help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    model_type = args.model_type  # "xlm-roberta"
    hyperlink_model_type = args.hyperlink_model_type # "standard" or "concat_cls" or "replace_cls"

    learning_rate = args.learning_rate  # 5e-6
    # adam_epsilon = 1e-8
    warmup_steps = args.warmup_steps
    # number of model updates to do during training. used in LR scheduler.
    num_training_steps = args.num_training_steps
    learning_rate_scheduler = args.learning_rate_scheduler
    gradient_accumulation_steps = args.gradient_accumulation_steps
    #two_step_training = args.two_step_training
    # weight_decay = 0.0
    seed = args.seed  # 1234
    max_grad_norm = args.max_grad_norm  # 1.0
    dim = args.hidden_dim  # 768  # bert dim
    hdim = dim  #
    cutoffs = [10000, 40000, 50000, 70000] # [4000, 20000, 40000, 80000] 
    n_classes = 250000
    set_seed(seed)

    def generate_batch(batch):
        # labels2Keep, listIds, input_mask, segment_ids, index2Keep
        # label idx_tokens idx_masks idx_segs idx_idxs
        # procsample = (idxs[1], idxs[2], idxs[3], idxs[4], idxs[0])

        labels = []
        token_id_list = []
        segment_list = []
        attention_masks = []
        idxFirstBpe1_list = []

        # maxLen = 0
        # nonet = False
        for entry in batch:
            # print(entry)
            label = entry[0]
            # map_object = map(int, label)
            # label = list(map_object)
            labels.append(label)

            indexed_tokens = entry[1]
            # indexed_tokens = [int(i) for i in indexed_tokens]
            token_id_list.append(indexed_tokens)

            input_mask = entry[2]
            # input_mask = [int(i) for i in input_mask]
            #map_object = map(int, input_mask)
            #input_mask = list(map_object)
            attention_masks.append(input_mask)

            #segment_ids = entry[3]
            #segment_list.append(segment_ids)

            idxFirstBpe1 = entry[4]
            # idxFirstBpe1 = [int(i) for i in idxFirstBpe1]
            # map_object = map(int, idxFirstBpe1)
            # idxFirstBpe1 = list(map_object)
            idxFirstBpe1_list.append(idxFirstBpe1)

        flatten = [int(item) for sublist in labels for item in sublist]
        return torch.tensor(flatten), torch.tensor(token_id_list), None, torch.tensor(
            attention_masks), idxFirstBpe1_list


    def generate_batch_mlm(batch):
        # input unmasked, attention mask, segment ids (unused), input masked
        labels = []
        attention_masks = []
        #segment_list = [] # unused
        token_id_list = []

        for entry in batch:
            # unmasked inputs are labels to be predicted
            label = entry[0]
            labels.append(label)

            input_mask = entry[1]
            attention_masks.append(input_mask)

            #segment_ids = entry[2]
            #segment_list.append(segment_ids)

            indexed_tokens = entry[3]
            token_id_list.append(indexed_tokens)

        return torch.tensor(labels), torch.tensor(attention_masks), torch.tensor(token_id_list)

    # if we are continuing training from a saved checkpoint, shuffle the datasets before starting
    do_shuffle = True if os.path.isfile( pathSAVE ) else False
    # obey if forcing shuffle/no shuffle
    if args.force_shuffle:
        do_shuffle = True
    if args.force_no_shuffle:
        do_shuffle = False

    training_set_foreign_list = []
    training_generator_foreign_list = []
    training_set_foreign_size_list = []
    training_set_foreign_total_steps_list = []
    # Generators
    if not args.baseline:
        print("Loading datasets...", file=sys.stderr)
        training_set = DataWIKI(pathTrainEnglish, shuffle=do_shuffle)
        training_generator = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False,
                                        collate_fn=generate_batch)

        if not args.english_only:
            for i, pathTrainOtherLanguage in enumerate(pathTrainOtherLanguages):
                # 603537
                #training_setEU = DataWIKI(pathTrainEU, 603537)
                #print("Foreign [%i %s]: Computing number of lines from gzip (slower)"%(i, pathTrainOtherLanguage))
                training_set_foreign = DataWIKI(pathTrainOtherLanguage, shuffle=do_shuffle)
                training_generator_foreign = DataLoader(training_set_foreign, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=generate_batch)
                training_set_foreign_list.append( training_set_foreign )
                training_generator_foreign_list.append( training_generator_foreign )

                t_size   = len(training_set_foreign)
                t_total2 = len(training_set_foreign) // gradient_accumulation_steps * num_train_epochs
                #print(t_total2)

                training_set_foreign_size_list.append( t_size )
                training_set_foreign_total_steps_list.append( t_total2 )
            
        # list with size of datasets, starting from English
        counts_all_languages = [len(training_set)] + training_set_foreign_size_list
        _, sampling_probs = get_sampling_probability_from_counts( counts_all_languages )
        print("Datasets sampling probabilities: ", sampling_probs)

        t_total1 = len(training_generator) // gradient_accumulation_steps * num_train_epochs
        print(t_total1)
    else:
        t_total1 = 0
        training_set_foreign_total_steps_list.append( 0 )
        training_set = None

    ######
    # MLM
    ######
    training_set_mlm = DataWIKI_MLM(pathTrainEnglish_mlm, shuffle=do_shuffle)
    training_generator_mlm = DataLoader(training_set_mlm, batch_size=BATCH_SIZE, shuffle=False,
                                    collate_fn=generate_batch_mlm)

    training_set_foreign_list_mlm = []
    training_generator_foreign_list_mlm = []
    training_set_foreign_size_list_mlm = []
    training_set_foreign_total_steps_list_mlm = []
    if not args.english_only:
        for i, pathTrainOtherLanguage_mlm in enumerate(pathTrainOtherLanguages_mlm):
            training_set_foreign_mlm = DataWIKI_MLM(pathTrainOtherLanguage_mlm, shuffle=do_shuffle)
            training_generator_foreign_mlm = DataLoader(training_set_foreign_mlm, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=generate_batch_mlm)
            training_set_foreign_list_mlm.append( training_set_foreign_mlm )
            training_generator_foreign_list_mlm.append( training_generator_foreign_mlm )

            t_size_mlm   = len(training_set_foreign_mlm)
            t_total2_mlm = len(training_set_foreign_mlm) // gradient_accumulation_steps * num_train_epochs
            #print(t_total2)

            training_set_foreign_size_list_mlm.append( t_size_mlm )
            training_set_foreign_total_steps_list_mlm.append( t_total2_mlm )
        
    # list with size of datasets, starting from English
    counts_all_languages_mlm = [len(training_set_mlm)] + training_set_foreign_size_list_mlm
    _, sampling_probs_mlm = get_sampling_probability_from_counts( counts_all_languages_mlm )
    print("Datasets sampling probabilities (MLM): ", sampling_probs_mlm)
    t_total1_mlm = len(training_generator_mlm) // gradient_accumulation_steps * num_train_epochs
    print(t_total1_mlm)



    #t_total2 = len(training_setEU) // gradient_accumulation_steps * num_train_epochs
    #print(t_total2)
    #model_class, model_config = MODEL_CLASSES[model_type]
    model_class = MODEL_CLASSES[model_type][0]

    # load XLM-R backbone and adaptive softmax weights
    hyperlink_model_class = HYPERLINK_MODEL_TYPE[hyperlink_model_type][0]
    model_backbone = hyperlink_model_class(model_class, model_name, hdim, n_classes, cutoffs)
    model_backbone.to('cuda:0')

    hyperlink_model_softmax_class = HYPERLINK_MODEL_TYPE[hyperlink_model_type][1]
    model_softmax = hyperlink_model_softmax_class(model_class, model_name, hdim, n_classes, cutoffs)
    model_softmax.to('cuda:0')

    # model_mlm is an instance of XLMRobertaForMaskedLM
    #if args.train_with_mlm:
    #    model_class_mlm = MODEL_CLASSES[model_type][1]
    #    model_mlm = model_class_mlm.from_pretrained( model_name )
    #    model_mlm.to('cuda:0')

    #    # dummy tokenizer, never used (only needed to create collator)
    #    tokenizer = XLMRobertaTokenizer.from_pretrained( model_name )
    #    collator = DataCollatorForLanguageModeling(tokenizer)

    #if args.finetune_all_layers:
    #    print("Finetune all layers")
    #    model_backbone.finetune_all_layers()
    #    #if args.train_with_mlm:
    #    #    model_mlm.finetune_all_layers()

    #print("Finetune all layers")
    #if two_step_training:
    #    # when training in two steps, first we only train the adaptive softmax parameters
    #    which_layers_to_finetune = "none"
    if args.finetune_all_layers:
        which_layers_to_finetune = "all"
    elif args.finetune_all_transformer_layers:
        which_layers_to_finetune = "all_transformer"
    elif args.finetune_half_transformer_layers:
        which_layers_to_finetune = "half_transformer"
    elif args.finetune_two_transformer_layers:
        which_layers_to_finetune = "two_transformer"
    elif args.finetune_layer_norm_layers:
        which_layers_to_finetune = "layer_norm"
    elif args.finetune_no_layers:
        which_layers_to_finetune = "none"
    else:
        raise Exception("")

    if os.path.isfile( pathSAVE ):
        print("Continuing training from %s"%pathSAVE)
        model_backbone, model_softmax = load_model( pathSAVE, model_backbone, model_softmax )
        n_reads = load_model_reads( pathSAVE )
        pathSAVE, iteration = pathSAVE.rsplit(".", 1)
        pathSAVE = pathSAVE + "."
        iteration = int(iteration)
        print("pathSAVE: ", pathSAVE)
        print("iteration: ", iteration)
        print("n_reads: ", n_reads)

        # Uncomment below to conduct dataset fast-forwarding.
        # We are instead shuffling datasets (on disk) each time it is loaded inside DataWiki.__init__ (more efficient)
        #datasets_fast_forward(n_reads, training_generator, training_generator_foreign_list)

    else:
        iteration = 0

    print("Finetuning ", which_layers_to_finetune, " layers")
    model_backbone.finetune_all_layers(which_layers_to_finetune)
    #if args.train_with_mlm:
    #    model_mlm.finetune_all_layers()
    #sys.exit(1)

    #if args.train_with_mlm:
    #    print("Tieing embedding and encoder weights.")
    #    PreTrainedModel._tie_encoder_decoder_weights( model_backbone.model.embeddings, model_mlm.roberta.embeddings, "" )
    #    PreTrainedModel._tie_encoder_decoder_weights( model_mlm.roberta.encoder, model_backbone.model.encoder, "" )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # always wrap model backbone in nn.DataParallel
    assert(torch.cuda.device_count() == len(args.gpu_ids)), \
            "There are %s GPUs available (torch.cuda.device_count() == %s), but args.gpu_ids == %s"%(
                str(torch.cuda.device_count()), str(torch.cuda.device_count()), str(args.gpu_ids))

    print("Using %s GPUs with nn.DataParallel!"%str(torch.cuda.device_count()))
    model_backbone = nn.DataParallel( model_backbone, device_ids=args.gpu_ids )
    #if args.train_with_mlm:
    #    model_mlm      = nn.DataParallel( model_mlm, device_ids=args.gpu_ids )
    #    model_all_parameters = list(model_backbone.parameters()) + list(model_softmax.parameters()) + list(model_mlm.parameters())
    #    print("number of trainable parameters: " + str(count_parameters(model_backbone)) +
    #            ", " + str(count_parameters(model_softmax)) + ", " + str(count_parameters(model_mlm)))

    #else:
    if args.finetune_no_layers:
        model_all_parameters = list(model_softmax.parameters())
    else:
        model_all_parameters = list(model_backbone.parameters()) + list(model_softmax.parameters())
    #print(model_all_parameters)
    #sys.exit(1)
    print("number of trainable parameters: " + str(count_parameters(model_backbone)) +
            ", " + str(count_parameters(model_softmax)))

    optimizer = AdamW(model_all_parameters, lr=learning_rate)
    #t_total = t_total1+t_total2
    t_total = t_total1 + sum(training_set_foreign_total_steps_list)
    t_total_mlm = t_total1_mlm + sum(training_set_foreign_total_steps_list_mlm)

    #def percentage(percent, whole):
    #    return (percent * whole) / 100.0
    #warmup_steps = int(percentage(10, t_total))

    #learning_rate = args.learning_rate  # 5e-6
    ## adam_epsilon = 1e-8
    #warmup_steps = args.warmup_steps
    #learning_rate_scheduler = args.learning_rate_scheduler
    if learning_rate_scheduler == "constant":
        print("Constant LR scheduler.")
        scheduler = get_constant_schedule(optimizer)

    elif learning_rate_scheduler == "linear_with_warmup":
        print("Linear LR scheduler with warm-up. Number of warm-up steps %i, total number of training steps %i."%(
            warmup_steps, num_training_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        
    else:
        raise Exception("Learning rate scheduler not supported: %s"%learning_rate_scheduler)

    is_backprop_once = False
    tr_num_model_updates = 0
    tr_loss = 0.0
    # model.zero_grad()
    init_time = datetime.now()
    print("Saving checkpoints every %i"%args.save_every)
    print("now =", init_time)
    print("START TRAINING")
    for epoch in range(num_train_epochs):
        # model.train()
        averageloss = 0
        #if args.train_with_mlm:
        #    model_mlm.zero_grad()
        model_backbone.zero_grad()
        model_softmax.zero_grad()
        #cntstep = 0
        cntstep = iteration
        # number of model updates (or "training_steps" in LR scheduler)
        tr_num_model_updates = cntstep % gradient_accumulation_steps

        if not args.baseline:
            ###################
            # Wiki prediction
            ###################
            # English iterator
            iteraEN = itertools.cycle(enumerate(training_generator))
            #iteraEU = itertools.cycle(enumerate(training_generatorEU))
            # Other languages' iterators
            itera_foreign_list = [itertools.cycle(enumerate(training_generator_foreign))
                                  for training_generator_foreign in training_generator_foreign_list]
            # English + other languages
            itera_all = [iteraEN] + itera_foreign_list
            # include dataset idx with eachh iterator
            itera_all = [(iidx, it) for iidx,it in enumerate(itera_all)]

        #######
        # MLM
        #######
        # English iterator
        iteraEN_mlm = itertools.cycle(enumerate(training_generator_mlm))
        # Other languages' iterators
        itera_foreign_list_mlm = [itertools.cycle(enumerate(training_generator_foreign_mlm))
                              for training_generator_foreign_mlm in training_generator_foreign_list_mlm]
        # English + other languages
        itera_all_mlm = [iteraEN_mlm] + itera_foreign_list_mlm
        # include dataset idx with eachh iterator
        itera_all_mlm = [(iidx, it) for iidx,it in enumerate(itera_all_mlm)]

        count_wikipred_or_mlm = 0
        while True:
            # alternate between Wikipedia hyperlink prediction and MLM prediction
            if count_wikipred_or_mlm % 2 == 0 and not args.baseline:
                task_name = "WikipediaHyperlinkPrediction"
                # sample a dataset (English + foreign): 0 to len, inclusive
                itera_curr = random.choices(itera_all, weights=sampling_probs)[0]
                dataset_idx, itera_curr_unpacked = itera_curr
                step, batch = next(itera_curr_unpacked)
                #print("Sampled dataset: ", dataset_idx)
                #print(itera_curr)

            else:
                task_name = "MLM"
                # sample a dataset (English + foreign): 0 to len, inclusive
                itera_curr_mlm = random.choices(itera_all_mlm, weights=sampling_probs_mlm)[0]
                dataset_idx_mlm, itera_curr_unpacked_mlm = itera_curr_mlm
                step, batch = next(itera_curr_unpacked_mlm)

            count_wikipred_or_mlm += 1

            if cntstep != 0 and cntstep % 100 == 0 and is_backprop_once:
                currentaverageloss = averageloss / cntstep
                print("STEP: " + str(cntstep) + " average loss: " + str(currentaverageloss))
                now = datetime.now()
                #print("now =", now)
                print("time elapsed: ", str(now-init_time))
                init_time = now

            if cntstep % args.save_every == 0 and is_backprop_once:
                print("saving", end="...")
                # if using DataParallel, make sure we're saving the unwrapped model
                #model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                #assert( isinstance(model_backbone, nn.DataParallel) )
                # recover actual model from nn.DataParallel.module
                model_backbone_tosave = model_backbone.module
                torch.save([model_backbone_tosave.state_dict(), model_softmax.state_dict()],
                        pathSAVE + str(cntstep))

                # get how many times each dataset was read, to fast-forward when we continue training
                datasets_nreads = []
                if not training_set is None and not training_set_foreign_list is None:
                    for d in [training_set] + training_set_foreign_list:
                        datasets_nreads.append( d.n_reads )
                else:
                    for d in training_set_foreign_list:
                        datasets_nreads.append( d.n_reads )
                nreads_out = np.array( [datasets_nreads] )

                fname = pathSAVE + str(cntstep) +"_datasets_nreads.txt"
                n_reads = np.savetxt( fname, nreads_out )
                print("done.")
                print("tr_num_model_updates: ", tr_num_model_updates)

            model_backbone.train()
            model_softmax.train()

            # prepare minibatch
            if task_name == "WikipediaHyperlinkPrediction" and not args.baseline:
                # Wikipedia hyperlink prediction
                token_id_tensor = batch[1].to('cuda:0')
                if type(batch[2]) != type(None):
                    segment_id_tensor = batch[2].to('cuda:0')
                else:
                    segment_id_tensor = None

                attention_tensor = batch[3].to('cuda:0')
                bpe1_list = batch[4]
                labels = batch[0].to('cuda:0')
                input_ids_labels = None

            else:
                assert(task_name == "MLM")
                # MLM
                # unmasked inputs (labels for MLM loss)
                input_ids_labels  = batch[0].to('cuda:0')
                # attention mask
                attention_tensor  = batch[1].to('cuda:0')
                # masked inputs
                token_id_tensor   = batch[2].to('cuda:0')
                labels            = None
                segment_id_tensor = None

            # model backbone is wrapped in a nn.DataParallel and is executed possibly across many GPUs
            r1, r2 = model_backbone(targets=labels,
                    input_ids=token_id_tensor,
                    attention_mask=attention_tensor,
                    token_type_ids=segment_id_tensor,
                    input_ids_labels=input_ids_labels,
                    task=task_name)

            if task_name == "WikipediaHyperlinkPrediction" and not args.baseline:
                # softmax is executed on a single GPU
                logits = model_softmax(targets=labels,
                        last_hidden_states=r1,
                        cls_tokens_batch=r2,
                        idx_list_list=bpe1_list)

                # COMPUTE LOSS
                loss = logits.loss

            else:
                assert(task_name == "MLM")
                # when using MLM, the first return value from the model is already the loss
                loss = r1
                loss = loss.mean()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            #print("loss: ", loss)
            loss.backward()
            stepz = cntstep+1
            if stepz % gradient_accumulation_steps == 0:
                is_backprop_once = True
                #if args.train_with_mlm:
                #    torch.nn.utils.clip_grad_norm_(model_mlm.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), max_grad_norm)
                #if task_name == "WikipediaHyperlinkPrediction":
                torch.nn.utils.clip_grad_norm_(model_softmax.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #if args.train_with_mlm:
                #    model_mlm.zero_grad()
                model_backbone.zero_grad()
                #if task_name == "WikipediaHyperlinkPrediction":
                model_softmax.zero_grad()

                tr_num_model_updates += 1

            averageloss += loss.item()
            cntstep += 1

        #averageloss = averageloss / cntstep
        #print("final average loss: " + str(averageloss))
        #torch.save(model.state_dict(), pathSAVE)


if __name__ == "__main__":
    main()
    print("DONE")
