from problems.synthetic import bbob_func_names

train_datasets = {
    k: [str(i) for i in range(200)] for k in bbob_func_names
}

validation_datasets = {
    k: [str(i) for i in range(200, 210)] for k in bbob_func_names
}

test_datasets = {
    k: [str(i) for i in range(210, 220)] for k in bbob_func_names
}