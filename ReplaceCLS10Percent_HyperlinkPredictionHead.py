import random
import torch
import torch.nn as nn

class ReplaceCLS10Percent_HyperlinkPredictionHead(nn.Module):
    def __init__(self, model_class, model_name_or_path, hdim, n_classes, cutoffs):
        super(ReplaceCLS10Percent_HyperlinkPredictionHead, self).__init__()
        self.act = nn.AdaptiveLogSoftmaxWithLoss(hdim, n_classes, cutoffs, div_value=4.0, head_bias=True)


    def forward(self, targets, last_hidden_states, cls_tokens_batch, idx_list_list, train=True):
        # collect only the embeddings of the hyperlinks (in idx_list_list)
        ctx_embs = []
        # for each entry in batch
        for i, ids in enumerate(idx_list_list):
            # example_cls_token: [1, features]
            example_cls_token = cls_tokens_batch[i].unsqueeze(0)
            # for each hyperlink in example
            for idx in ids:
                draw = random.random() # real number between [0.0, 1.0)
                if draw <= 0.1:
                    # use CLS token instead of hyperlink hidden state 10% of the time
                    hyperlink_state = example_cls_token
                else:
                    hyperlink_state = last_hidden_states[i, idx].unsqueeze(0)
                # hyperlink_state: [1, 2*features]    (or [1, 2048])
                #hyperlink_state = torch.cat([example_cls_token, last_hidden_states[i, idx].unsqueeze(0)], dim=1)
                ctx_embs.append( hyperlink_state )
        
        final_emb = torch.cat(ctx_embs, dim=0)
        #print("final_emb.size(): ", final_emb.size())
        #final_emb = torch.cat((pooler_output, final_emb), 1) #CLS token in BERT
        if train:
            output = self.act(final_emb, targets)
            return output
        else:
            output = self.act.log_prob(final_emb)#Output: (N, n_classes)
            #output = self.act.predict(final_emb)#Output: (N)
            return output

        ## collect only the embeddings of the hyperlinks (in idx_list_list)
        ##print("last_hidden_states.shape: ", last_hidden_states.shape)
        ##print("idx_list_list: ", idx_list_list)
        #ctx_embs = []
        ## for each entry in batch
        #for i, ids in enumerate(idx_list_list):
        #    # when using DataParallel, manually collecting examples like we're doing breaks
        #    # we have to make sure we're only looking at the examples in the current copy
        #    # for each hyperlink in example
        #    for idx in ids:
        #        ctx_embs.append(last_hidden_states[i, idx].unsqueeze(0))
        #
        #final_emb = torch.cat(ctx_embs, dim=0)
        #if train:
        #    output = self.act(final_emb, targets)
        #    return output
        #else:
        #    output = self.act.log_prob(final_emb)#Output: (N, n_classes)
        #    return output

