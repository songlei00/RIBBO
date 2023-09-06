import os

with open('./tmp.log', 'r') as f:
    d = f.read()

exec('file_names=' + d)
base_dir = 'data/generated_data/hpob'
for file_name in file_names:
    seed = file_name.rstrip('.json').split('_')[-1]
    path = os.path.join(base_dir, 'seed{}'.format(seed), file_name)
    os.system('rm {}'.format(path))