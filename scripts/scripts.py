import subprocess

if __name__ == "__main__":

    mgnn_dir = "../src/"
    exp_dir = "../"

    experiment_type = "link_prediction"
    dataset_names = ["fb237v1"]
    train_graph = "train_graph.tsv"
    test_graph = "test_graph.tsv" 
    use_dummy_constants = True 
    train_examples =  "train_pos.tsv"
    test_positive_examples = "test_pos.tsv"
    test_negative_examples = "test_neg.tsv"
    encoding_scheme = "iclr22"      # canonical or iclr22
    aggregation = "max-max"
    derivation_threshold = "0.000000001"
    explanation_threshold = "0.1"
    non_negative_weights = "True"
    train = True 
    test = True 
    explain = True


    if non_negative_weights == 'True':
        monotonicity = 'monotonic'
    else:
        monotonicity = "nonmonotonic"
    if use_dummy_constants:
        train_with_dummies = "with_dummies"
    else:
        train_with_dummies = "without_dummies"

    for dataset_name in dataset_names:
        
        experiment_name = dataset_name + "-" + aggregation + '-' + monotonicity + '-' + encoding_scheme + '-' + train_with_dummies 
       
        #  TRAIN 
        if train:
            print("Training...") 
            subprocess.run(['python',
                            mgnn_dir + "train.py",
                            '--model-name', experiment_name, 
                            '--model-folder', exp_dir + "models",
                            '--encoder-folder', exp_dir + "encoders",
                            '--train-graph', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, train_graph),
                            '--train-examples', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, train_examples),
                            '--encoding-scheme', encoding_scheme, 
                            '--predicates', exp_dir + "data/{}/{}/predicates.csv".format(experiment_type, dataset_name),
                            '--aggregation', aggregation,
                            '--non-negative-weights', non_negative_weights ])
        
        #  TEST 
        if test: 
            print("Testing...")
            subprocess.run(['python',
                            mgnn_dir + "test.py",
                            '--load-model-name', exp_dir + "models/{}.pt".format(experiment_name),
                            '--canonical-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_canonical.tsv",
                            '--iclr22-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_iclr22.tsv",
                            '--threshold', derivation_threshold, 
                            '--predicates', exp_dir + "data/{}/{}/predicates.csv".format(experiment_type, dataset_name),
                            '--test-graph', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, test_graph),
                            '--test-positive-examples', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, test_positive_examples),
                            '--test-negative-examples', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, test_negative_examples),
                            '--output', exp_dir + "metrics/{}/{}/{}.txt".format(experiment_type, dataset_name, experiment_name),
                            '--print-entailed-facts', exp_dir + "predictions/{}/{}.tsv".format(dataset_name, experiment_name),
                            '--encoding-scheme', encoding_scheme])
    
        #  EXPLAIN FACTS 
        if explain: 
            print("Computing fact explanations...")
            subprocess.run(['python',
                            mgnn_dir + "fact_explanation.py",
                            '--load-model-name', exp_dir + "models/{}.pt".format(experiment_name),
                            '--canonical-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_canonical.tsv",
                            '--iclr22-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_iclr22.tsv",
                            '--encoding-scheme', encoding_scheme, 
                            '--threshold', explanation_threshold,
                            '--predicates', exp_dir + "data/{}/{}/predicates.csv".format(experiment_type, dataset_name),
                            '--dataset', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, test_graph),
                            '--facts', exp_dir + "predictions/{}/{}.tsv".format(dataset_name, experiment_name),
                            '--output', exp_dir + "explanations/{}/{}.txt".format(dataset_name, experiment_name)])

