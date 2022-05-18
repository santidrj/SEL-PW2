This zip file contains the necessary folders to run the experiments for the Random Forest and Decision Forest algorithms.

The zip is structured as follows:
    - documentation         [The folder contains a PDF report with the pseudocode of this implementation, a discussion over the results and how to execute the code]
    - data                  [The folder contains three datasets of different sizes used to perform the experiments]
    - source                [The folder contains the implementation of the Random Forest, Decision Forest, and CART algorithms and an auxiliary file to load the data]
    - run_experiments.py    [An executable python file to run the experiments]
    - README.txt            [A README explaining the contents of the zip file]

Running the test
To run the test you need to create a python virtual environment and install in it the following dependencies:
    python = "^3.9"
    numpy = "^1.22.3"
    pandas = "^1.4.2"
    scikit-learn = "^1.1.0"
    matplotlib = "^3.5.2"
    Jinja2 = "^3.1.2"

After the installation, you can run the experiments with the run_experiments.py script.
!!!The run_experiments.py must be in the same location that the source and data folders!!!

Also, you might need to add the current folder to your PYTHONPATH environment variable.
To do so in Unix-like systems you can run:
export PYTHONPATH=`pwd`:$PYTHONPATH
In Windows run:
$Env:PYTHONPATH += pwd


usage: run_experiments.py [-h] [-d MAX_DEPTH] [--min_size MIN_SIZE] [-j JOBS] [-n N_SPLITS] [-s SEED] [-v] dataset {RandomForest,DecisionForest,all}

positional arguments:
  dataset               The dataset to use for testing. Available datasets are: iris, heart, nursery, all. You can also pass the relative path to a dataset in CSV format.
  {RandomForest,DecisionForest,all}
                        The classifier to test.

optional arguments:
  -h, --help            show this help message and exit
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        The maximum depth of the trees. Use -1 for no limit. Using -1 may cause a RecursionError if the data is not easily split, please consider setting a suitable depth. Defaults to -1.
  --min_size MIN_SIZE   The minimum size that each node must have. Defaults to 1.
  -j JOBS, --jobs JOBS  The number of jobs to run in parallel. If -1 uses all the available CPUs. Defaults to 1.
  -n N_SPLITS, --n_splits N_SPLITS
                        The number of splits for the StratifiedKFold. Defaults to 5.
  -s SEED, --seed SEED
  -v, --verbose
