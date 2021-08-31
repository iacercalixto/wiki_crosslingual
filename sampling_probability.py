def get_sampling_probability_from_counts(datasets_counts_list):
    # following: https://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf (Section 3.1)
    # given:   n_i is the number of examples in dataset i
    # compute: q_i = p_i^{\alpha} / \sum_{j=1}^{N}{ p_j^{\alpha} }
    #          p_i = n_i / \sum_{k=1}^{N}{ n_k }
    alpha = 0.5
    n_datasets = len(datasets_counts_list)
    count = [0] * n_datasets
    #for i in range(n_datasets):
    #    count[i] = len(datasets_list[i])
    count = datasets_counts_list
    N = float(sum(count))
    p = [0.] * n_datasets
    for i in range(n_datasets):
        p[i] = count[i] / N
    # create an output vector where the probability of sampling an example from a dataset is upsampled for the low-resource language
    # but not too much.
    final_weights_per_example = []
    final_weights_per_dataset = []
    for i in range(n_datasets):
        p_sum = 0.
        for ii in range(n_datasets):
            p_sum += (p[ii] ** alpha)
        qi = (p[i] ** alpha) / p_sum
        for _ in range(count[i]):
            final_weights_per_example.append( qi )
        final_weights_per_dataset.append( qi )
    return final_weights_per_example, final_weights_per_dataset

def get_sampling_probability(datasets_list):
    # following: https://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf (Section 3.1)
    # given:   n_i is the number of examples in dataset i
    # compute: q_i = p_i^{\alpha} / \sum_{j=1}^{N}{ p_j^{\alpha} }
    #          p_i = n_i / \sum_{k=1}^{N}{ n_k }
    alpha = 0.5
    n_datasets = len(datasets_list)
    count = [0] * n_datasets
    for i in range(n_datasets):
        count[i] = len(datasets_list[i])
    N = float(sum(count))
    p = [0.] * n_datasets
    for i in range(n_datasets):
        p[i] = count[i] / N
    # create an output vector where the probability of sampling an example from a dataset is upsampled for the low-resource language
    # but not too much.
    final_weights_per_example = []
    final_weights_per_dataset = []
    for i in range(n_datasets):
        p_sum = 0.
        for ii in range(n_datasets):
            p_sum += (p[ii] ** alpha)
        qi = (p[i] ** alpha) / p_sum
        for _ in range(count[i]):
            final_weights_per_example.append( qi )
        final_weights_per_dataset.append( qi )
    return final_weights_per_example, final_weights_per_dataset

def get_datasets_sampling_probability(datasets_list):
    _, res = get_sampling_probability(datasets_list)
    return res

if __name__=="__main__":
    datasets = [ list(range(100)), list(range(85)), list(range(120)) ]
    _, res = get_sampling_probability( datasets )
    print(res)
    res1 = get_datasets_sampling_probability( datasets )
    print(res1)

    datasets_counts = [ 100, 85, 120 ]
    _, res2 = get_sampling_probability_from_counts( datasets_counts )
    print(res2)

