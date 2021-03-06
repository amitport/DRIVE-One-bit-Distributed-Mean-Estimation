# Drive evaluation: federated learning

This sub-project contains the federated tasks presented in the evaluation. 

## Requirements

### install requirements:

```setup
pip install -r requirements.txt
```

### Initialize git submodule

Run the following to make sure that the remote Google's [federated research repo](https://github.com/google-research/federated) is cloned as a submodule:

```setup
git submodule update --init --recursive
```

### Update PYTHONPATH

Add `experiments/federated/google_tff_research` to `PYTHONPATH`.

## Training

Each line in `experiments.txt` corresponds to command line parameters of one experiment.

In order to reproduce the paper's results, execute `experiments/federated/run_experiment.py` (the current working directory should be the repo's root) for every line in `experiments.txt` and add the `--root_output_dir` command line parameter to specify the output directory.

You can view the documentation for every command line parameter using `experiments/federated/run_experiment.py --help`.

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```

## Results

Execute `fl_plot.ipynb` using [Jupyter](https://jupyter.org/) to re-create Figure 2 from the paper. 