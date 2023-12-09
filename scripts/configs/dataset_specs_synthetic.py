from problems.synthetic import bbob_func_names

train_datasets = {
    k: [str(i) for i in range(50)] for k in bbob_func_names
}

validation_datasets = {
    k: [str(i) for i in range(50, 60)] for k in bbob_func_names
}

test_datasets = {
    k: [str(i) for i in range(60, 70)] for k in bbob_func_names
}