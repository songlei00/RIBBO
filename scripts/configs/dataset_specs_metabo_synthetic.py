names = (
    'Branin2',
    'Hartmann3',
)

train_datasets = {
    name: [str(i) for i in range(200)] for name in names
}

validation_datasets = {
    name: [str(i) for i in range(200, 210)] for name in names
}

test_datasets = {
    name: [str(i) for i in range(210, 220)] for name in names
}