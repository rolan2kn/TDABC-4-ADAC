TDAbcTK: A classification framework 
entirely based on Topological Data Analysis


* Install:
1. Install Anaconda
2. make a new environment

conda create --name tdabc_env

3. Install gudhi according to (https://gudhi.inria.fr/python/latest/installation.html). 
   We choose the conda instalation way: 

conda install -c conda-forge gudhi

4. Install scikit-learn

conda install -c conda-forge scikit-learn

5. Install matplotlib

conda install -c conda-forge matplotlib

para que funcione con latex

sudo apt install texlive-fonts-recommended 
texlive-fonts-extra

sudo apt install dvipng

6. Install umap (optional). 
   In case your want to install it, you need to 
   downgrade your python to python 3.8, and every 
   package as well to be compatible with this version. 
   
conda install -c conda-forge umap-learn
pip install umap-learn

7. Install h5py

conda install -c conda-forge h5py

* Tutorial

the entry point is the main.py

to run the project it is enough with:

python main.py

however the main accepts arguments to configure the execution:

usage: main.py [-h] [-o OPTION] [-d DATASET] [-path PATH] [-i ITERATION] [-n NUMBER_DATASETS] [-m METRIC] [-t TRANSFORMATION]

Where:

-> -o --option help="Select option to execute 1: Multi KRNN experiment, 2: KRNN, 3: Conventional experiment " type=int

-> -d --dataset help="Select the datasets" type=int

-> -path --path help="Define folder to save results" type=str

-> -i --iteration" help="Iteration number for option=1" type=int

-> -n --number_datasets help="Number of datasets for option=1 type=int

-> -m --metric help="Metric to build the simplicial comples. A collection of possible  metrics are: ['cityblock', 'cosine', "
                                                   "'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean','braycurtis',"
                                                   " 'canberra', 'chebyshev','correlation', 'dice', 'hamming', 'jaccard',"
                                                   " 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', "
                                                   "'russellrao', 'seuclidean', 'sokalmichener', "
                                                   "'sokalsneath', 'sqeuclidean', 'yule']" type=str

-> -t --transformation help="Data Transformation type to apply, types are: 0=NONE, SIMPLE=1, SQUARED=2, NLOG=3, NORM=4, INV=5, ELOG=6, REDUCTION=7" type=int

When select a data transformation of reduction, it means that you want to perform a dimensionality reduction, we
implement three algorithms PCA=0, TSNE=1, and UMAP=2 there is not interface to access to those method from the entry point.


Any comment, please do not hesitate to write a mail to Rolando Kindelan Nuñez (rolan2kn@gmail.com, rkindela@dcc.uchile.cl), twitter: @rolan2kn.


* Execution

- To run the experiment in the eight datasets:

python main.py -o 3 -p YOUR_RESULT_PATH

you can also define -m YOUR_DESIRED_METRIC but it is optional because the euclidean distance is default.

- To run the experiment of generating the results of all datasets (Figure 7 in the paper)

python main.py -o 1 -i 1 -p "./docs/ADAC_TDABC_RESULTS"

This option 1 what does is to collect all dataset info inside the path directory and looking for *metrics.txt files
which are where all metric results are stored. Note that this method, assumes all datasets folder has the same number
of experiements.

To generate the RKNN experiments with 16 variations of samples, we need to provide the folder where all KRNN datasets
are stored. The algorithms detects by itself according to the structure of data if its the case of datasets
or the KRNN one.

To RUN the RKNN experiment, we must specify the iterations that we need to consider.

python main.py -o 1 -i 3 -n 16 -p "./docs/KRNN_EXPERIMENTS"

Generates n datasets 50:50, 50:100, 50:150, ..., 50:800 and executes i iterations of each one of our TDABC
and also generate the plots.

NOTE that both experiments with option -o 1 are mutually excluyent and they should not run on the same directory
