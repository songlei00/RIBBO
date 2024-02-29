# Reinforced In-Context Black-Box Optimization

This repository contains the Python code for [*Reinforced In-Context Black-Box Optimization (RIBBO)*](https://arxiv.org/abs/2402.17423), a method to reinforce-learn a BBO algorithm from offline data in an end-to-end fashion.

## Requirements

- Python == 3.10
- PyTorch == 2.0.1
- offlinerllib==0.1.1
- utilsrl==0.6.3
- google-vizier==0.1.9
- gpytorch==1.11
- botorch=0.9.4

## File Structure

- ```algorithms``` directory is the main implement of RIBBO, BC, BC Filter, and OptFormer
- ```data_gen``` directory is the implement of behavior algorithms and data collection
- ```datasets``` directory provides the interface of the offline datasets
- ```problems``` directory is the implement of the benchmark problems
- ```scripts``` directory provides some scripts for reproduction

## Usage

Run ```bash scripts/run_main.sh``` to evaluate RIBBO and other baselines

## Datasets

The datasets were not released during the review stage, but we provided detailed guidelines in our paper and the code for data generation. We will open-source the datasets and code after the final decision
