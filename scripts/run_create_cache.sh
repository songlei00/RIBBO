#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

# python others/create_hpob_dataset_stats.py
python others/create_hpob_dataset_cache.py
# python others/filter_bad_hpob_dataset.py