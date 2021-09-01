# wiki_crosslingual

Code to reproduce the NAACL 2021 paper [Wikipedia entities as rendezvous across languages: grounding multilingual LMs by predicting wikipedia hyperlinks](https://aclanthology.org/2021.naacl-main.286/).

## Installation

Please use the `environment.yml` file to install all required packages using anaconda. The following should work: `conda env create --name envname --file=environment.yml`.

## Data

Please use [this link](https://drive.google.com/#file) to download the preprocessed and tokenized Wikipedia data we use to train our models with masked language modelling training, i.e., the file contains Wikipedia text without any markups to be used for MLM training. The file is a tarball containing a list of gzipped text files for each of the 100 Wikipedia languages used. Each line consists of Wikipedia text preprocessed for MLM training. The first field are prediction indices (index -100 is a mask not used in MLM), the second field is the attention mask used in BERT, and the fourth field denote the input text with randomly masked words. The third field is not used.

Please use [this link](https://drive.google.com/#file) to download the preprocessed and tokenized Wikipedia data we used to train our models with entity prediction, i.e., the file contains sentences extracted from Wikipedia with internal hyperlinks to be used in entity prediction. The file is a tarball containing a list of gzipped text files for each of the 100 Wikipedia languages used. Each line consists of some text from Wikipedia and the hyperlinks targets for all mentions in the text. The first field denotes mention indices (spans) in the input text, the second field denotes the entity identifier each mention (in field 1) links to, the third field is the input text with randomly masked words, the fourth field is an attention mask to be used in BERT. The fifth field is not used.

Extract the tarballs and store all gzipped files in a directory of your preference.

Before running the example below, remember to create the subdirectory `./save_path`.

## Example

Below we show how to perform one step of intermediate-task training with XLM-R-Large to predict entity mentions using English Wikipedia articles.

Before doing that we need to set a few environment variables. The first variables are `${BATCH_SIZE}` and `${G_ACC_STEPS}`, which denote the mini-batch size and the number of gradient accumulation steps, respectively. Assuming you have an NVIDIA V100 GPU card with 24GBs you can set `${BATCH_SIZE} = 16` and `${G_ACC_STEPS} = 16`, which makes the effective mini-batch size to be 16 times 16 or 256. If your GPU card has less memory or if you are having problems with GPU memory, try reducing the mini-batch size and increasing the gradient accumulation steps (e.g., `${BATCH_SIZE} = 4` and `${G_ACC_STEPS} = 64`). As long as the product between the two is 256, you should obtain similar results to ours.

Finally, set the environment variable `${DATA_PATH}` with the location of the gzipped files above, the environment variable `${SAVE_EVERY}` to the number of model updates used to periodically save the model, and run:

	python -u train_entity_prediction.py \
	    --train_path_english ${DATA_PATH}/wiki.en.links.top250K.idx.wordtok.xlmr.txt.gz \
	    --language_codes_file languages/languages-en.txt \
	    --language_codes_file_mlm languages/languages-en.txt \
	    --model_type xlm-roberta \
	    --model_name_or_path xlm-roberta-large \
	    --hidden_dim 1024 \
	    --path_save save_path/Model_xlm-roberta-large_train-with-mlm_finetune-two-transformer-layers_token_en_languages. \
	    --train_path_english_mlm ${DATA_PATH}/wiki.en.links.top250K.idx.wordtok.xlmr.mlm.txt.gz.shuf.gz \
	    --train_with_mlm \
	    --hyperlink_model_type standard \
	    --batch_size ${BATCH_SIZE} \
	    --gradient_accumulation_steps ${G_ACC_STEPS} \
	    --finetune_two_transformer_layers \
	    --gpu_ids 0 \
	    --english_only \
	    --save_every ${SAVE_EVERY} \
	    1>&2 | tee -a console.xlm-roberta-large.train-with-mlm.finetune-two-transformer-layers.token.lr5e5.en_languages.txt


In order to train a ReplaceCLS entity prediction model (see paper for details) on 100 Wikipedia languages, run:

	python -u train_entity_prediction.py \
	    --train_path_english ${DATA_PATH}/wiki.en.links.top250K.idx.wordtok.xlmr.txt.gz \
	    --train_path_other_languages ${DATA_PATH}/ \
	    --language_codes_file languages/languages-XLMR-all.txt \
	    --language_codes_file_mlm languages/languages-XLMR-all.txt \
	    --model_type xlm-roberta \
	    --model_name_or_path xlm-roberta-large \
	    --hidden_dim 1024 \
	    --path_save save_path/Model_xlm-roberta-large_train-with-mlm_finetune-two-transformer-layers_replace_cls_100_languages. \
	    --train_path_english_mlm ${DATA_PATH}/wiki.en.links.top250K.idx.wordtok.xlmr.mlm.txt.gz.shuf.gz \
	    --train_path_other_languages_mlm ${DATA_PATH}/ \
	    --train_with_mlm \
	    --hyperlink_model_type replace_cls \
	    --batch_size ${BATCH_SIZE} \
	    --gradient_accumulation_steps ${G_ACC_STEPS} \
	    --finetune_two_transformer_layers \
	    --gpu_ids 0 \
	    --save_every ${SAVE_EVERY} \
	    1>&2 | tee -a console.xlm-roberta-large.train-with-mlm.finetune-two-transformer-layers.replace_cls.lr5e5.100_languages.txt


To see all available training options, use the `--help` flag.


## Citing our work

If you find our paper and/or this codebase useful, please consider citing our work:

    @inproceedings{calixto-etal-2021-wikipedia,
        title = "{W}ikipedia Entities as Rendezvous across Languages: Grounding Multilingual Language Models by Predicting {W}ikipedia Hyperlinks",
        author = "Calixto, Iacer  and
          Raganato, Alessandro  and
          Pasini, Tommaso",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.naacl-main.286",
        doi = "10.18653/v1/2021.naacl-main.286",
        pages = "3651--3661",
    }

