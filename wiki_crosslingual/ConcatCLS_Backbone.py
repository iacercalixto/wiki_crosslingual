import torch
import torch.nn as nn
import re
#import torch.nn.functional as F

class MMModel_ConcatCLS_WithDataParallel(nn.Module):
    def __init__(self, model_class, model_name_or_path, hdim, n_classes, cutoffs):
        super(MMModel_ConcatCLS_WithDataParallel, self).__init__()

        self.model = model_class.from_pretrained(model_name_or_path, output_hidden_states = True)
        #for name, param in self.model.named_parameters():
        #    if "layer.22" in name or "layer.23" in name: # "layer.10" and 'layer.11' for mBERT
        #        param.requires_grad = True
        #    else:
        #        param.requires_grad = False


    def finetune_all_layers(self, layers="all"):
        valid_layers = ["all", "all_transformer", "half_transformer", "two_transformer", "layer_norm", "none"]
        assert( layers in valid_layers ), \
                "layers not valid.\nreceived: %s\nchoose from: %s"%(layers,valid_layers)

        def split_list(a_list):
            """ split a list in half. requires list size to be even. """
            assert(len(a_list) % 2 == 0)
            half = len(a_list)//2
            return a_list[:half], a_list[half:]

        # first, count number of transformer layers
        layers_idxs = []
        for param_name, param in self.model.named_parameters():
            match = re.search( ".+layer.(\d+)", param_name )
            if not match is None:
                layers_idxs.append( int(match.groups()[0]) )
        # all transformer layers
        layers_idxs = sorted(list(set(layers_idxs)))
        # last two transformer layers
        layers_idxs_two  = layers_idxs[ -2: ]
        # transformer layers split in half
        layers_idxs_half = split_list( layers_idxs )

        #print("layers_idxs: ", layers_idxs)
        #print("layers_idxs_two: ", layers_idxs_two)
        #print("layers_idxs_half: ", layers_idxs_half)

        for param_name, param in self.model.named_parameters():
            match = re.search( ".+layer.(\d+)", param_name )
            match_ln = "LayerNorm" in param_name

            if layers=="all":
                print( "finetuning ", param_name )
                param.requires_grad = True

            elif layers=="layer_norm" and match_ln:
                #print( "finetuning ", param_name )
                # if parameter is a layer norm parameter in the input embeddings or self-attention layers
                param.requires_grad = True

            elif layers=="all_transformer" and not match is None:
                #print( "finetuning ", param_name )
                # if parameter is one of the transformer layers
                param.requires_grad = True

            elif layers=="half_transformer" and not match is None and int(match.groups()[0]) in layers_idxs_half[-1]:
                #print( "finetuning ", param_name )
                # if parameter is in the top half of the transformer layers
                param.requires_grad = True
            
            elif layers=="two_transformer" and not match is None and int(match.groups()[0]) in layers_idxs_two:
                #print( "finetuning ", param_name )
                param.requires_grad = True

            else:
                match = re.search( "lm_head", param_name )
                if match is None:
                    #print( "NOT finetuning ", param_name )
                    param.requires_grad = False
                else:
                    # always finetune the LM head
                    #print( "finetuning ", param_name )
                    param.requires_grad = True


    def forward(self, targets, input_ids, attention_mask, token_type_ids, input_ids_labels, task="WikipediaHyperlinkPrediction"):
        assert( task in ["MLM", "WikipediaHyperlinkPrediction"] )

        if task == "WikipediaHyperlinkPrediction":
            # XLM type model that do not need token_type_ids
            if type(token_type_ids) == type(None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # BERT type model with token_type_ids
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            #last_hidden_states = outputs[0]
            all_hidden_states = outputs[1] # retrieve all hidden states (including embeddings) from model
            # last_hidden_states: [batch, seq_len, features]
            last_hidden_states = all_hidden_states[-1] # retrieve only hidden states of the last Transformer layer
            cls_tokens_batch = last_hidden_states[:,0,:]
            return last_hidden_states, cls_tokens_batch

        elif task == "MLM":
            # XLM type model that do not need token_type_ids
            if type(token_type_ids) == type(None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids_labels)
                loss = outputs[0]

            else:
                # BERT type model with token_type_ids
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=input_ids_labels)
                loss = outputs[0]

            # when performing MLM, we return the loss directly
            return loss, None

        else:
            raise Exception("task not found: "+task)

