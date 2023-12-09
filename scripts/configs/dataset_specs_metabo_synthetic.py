names = (
    'Branin2',
    'Hartmann3',
)

train_datasets = {
    name: [str(i) for i in range(50)] for name in names
}

validation_datasets = {
    name: [str(i) for i in range(50, 60)] for name in names
}

test_datasets = {
    name: [str(i) for i in range(60, 70)] for name in names
}