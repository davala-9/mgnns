import subprocess

if __name__ == "__main__":

  
    configuration = {}
    for line in open('experiment.config').readlines():
        parameter = line.split()[0]
        if parameter == "dataset_names":
            value = line.split()[2:]
        else:
            value = line.split()[2]
        configuration[parameter] = value
    mgnn_dir = configuration["mgnn_dir"]
    exp_dir = configuration["exp_dir"]
    experiment_type = configuration["experiment_type"]
    dataset_names = configuration["dataset_names"]
    train_graph = configuration["train_graph"] 
    valid_graph = configuration["valid_graph"]
    test_graph = configuration["test_graph"] 
    use_dummy_constants = configuration["use_dummy_constants"]=="True" 
    train_examples = configuration["train_examples"] 
    valid_positive_examples = configuration["valid_positive_examples"] 
    valid_negative_examples = configuration["valid_negative_examples"] 
    test_positive_examples = configuration["test_positive_examples"] 
    test_negative_examples = configuration["test_negative_examples"] 
    encoding_scheme = configuration["encoding_scheme"]
    aggregation = configuration["aggregation"] 
    derivation_threshold = configuration["derivation_threshold"]
    explanation_threshold = configuration["explanation_threshold"] 
    non_negative_weights = configuration["non_negative_weights"]
    train = configuration["train"]=="True" 
    valid = configuration["valid"]=="True"
    test = configuration["test"]=="True" 
    explain = configuration["explain"]=="True" 
    minimal_rule = configuration["minimal_rule"]=="True"



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

        #  VALIDATE
        if valid: 
            print("Validating...")
            subprocess.run(['python',
                            mgnn_dir + "test.py",
                            '--load-model-name', exp_dir + "models/{}.pt".format(experiment_name),
                            '--canonical-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_canonical.tsv",
                            '--iclr22-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_iclr22.tsv",
                            '--predicates', exp_dir + "data/{}/{}/predicates.csv".format(experiment_type, dataset_name),
                            '--test-graph', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, valid_graph),
                            '--test-positive-examples', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, valid_positive_examples),
                            '--test-negative-examples', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, valid_negative_examples),
                            '--output', exp_dir + "metrics/{}/{}/{}.txt".format(experiment_type, dataset_name, experiment_name),
                            '--print-entailed-facts', exp_dir + "predictions/{}/{}.tsv".format(dataset_name, experiment_name),
                            '--encoding-scheme', encoding_scheme])

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
            argument = ['python',
                            mgnn_dir + "fact_explanation.py",
                            '--load-model-name', exp_dir + "models/{}.pt".format(experiment_name),
                            '--canonical-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_canonical.tsv",
                            '--iclr22-encoder-file', exp_dir + "encoders/{}".format(experiment_name) + "_iclr22.tsv",
                            '--encoding-scheme', encoding_scheme, 
                            '--threshold', explanation_threshold,
                            '--predicates', exp_dir + "data/{}/{}/predicates.csv".format(experiment_type, dataset_name),
                            '--dataset', exp_dir + "data/{}/{}/{}".format(experiment_type, dataset_name, test_graph),
                            '--facts', exp_dir + "predictions/{}/{}.tsv".format(dataset_name, experiment_name),
                            '--output', exp_dir + "explanations/{}/{}.txt".format(dataset_name, experiment_name)]
            if minimal_rule:
                argument.append('--minimal-rule')
            subprocess.run(argument)

