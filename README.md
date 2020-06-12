# Mini-batch Size and Convergence Diagnostics for SGD
Mini-project for CS-439 Optimization for Machine Learning 2020

## Abstract
In this paper, we present our research on the topic of optimization algorithms for deep learning tasks. In particular, we extend the knowledge about a recent modification of SGD called SGD^(1/2). We will show the influence of mini-batch size on the training time and accuracy, with a comparison across other optimizers. Finally, we will discuss the different properties of the optimization algorithms with respect to their convergence behaviour. The results proved we can recommend higher mini-batch sizes for adaptive learning rate optimization and we support the utilization of the improved SGD for its convergence benefits.

## File structure
```
│   data.py - Module containing the code for loading the datasets
│   experiments.py - Module containing the main code for executing the experiments
│   experiment_utils.py - Module containing helper functions for executing procedures required for the experiments
│   models.py - Module containing code required to instantiate the shallow and deep model
│   plotting.py - Module containing functions for generating the visualizations of the experiment results
│   report.pdf - Detailed project report
│   run.py - Main script - contains the code to run the experiments
			   and generate the figures using the obtained results
│   settings.py - Module containing the implementation of the procedure
		  required to parse the command line arguments
│
├───results
│   │   circle_convergence_region_experiment_log.csv
│   │   circle_mini_batch_size_lr_experiment_log.csv
│   │   circle_mini_batch_size_lr_training_logs.csv
│   │   fashion_mnist_convergence_region_experiment_log.csv
│   │   fashion_mnist_mini_batch_size_lr_experiment_log.csv
│   │   fashion_mnist_mini_batch_size_lr_training_logs.csv
│   │   mnist_convergence_region_experiment_log.csv
│   │   mnist_mini_batch_size_lr_experiment_log.csv
│   │   mnist_mini_batch_size_lr_training_logs.csv
│   │
│   ├───plots
│   │   │   experiment_1_learning_rate_vs_convergence_time.png
│   │   │   experiment_1_mini_batch_size_vs_target_quantities.png
│   │   │   experiment_1_training_vs_validation_accuracy.png
│   │   │   experiment_2_convergence_region_comparison.png
│   │   │
│   │   └───convergence_regions
│   │           convergence_region_circle_adam_cross_entropy.png
│   │           convergence_region_circle_adam_mse.png
│   │           convergence_region_circle_sgd_cross_entropy.png
│   │           convergence_region_circle_sgd_mse.png
│   │           convergence_region_circle_sgd_to_half_cross_entropy.png
│   │           convergence_region_circle_sgd_to_half_mse.png
│   │           convergence_region_fashion_mnist_adam_cross_entropy.png
│   │           convergence_region_fashion_mnist_adam_mse.png
│   │           convergence_region_fashion_mnist_sgd_cross_entropy.png
│   │           convergence_region_fashion_mnist_sgd_mse.png
│   │           convergence_region_fashion_mnist_sgd_to_half_cross_entropy.png
│   │           convergence_region_fashion_mnist_sgd_to_half_mse.png
│   │           convergence_region_mnist_adam_cross_entropy.png
│   │           convergence_region_mnist_adam_mse.png
│   │           convergence_region_mnist_sgd_cross_entropy.png
│   │           convergence_region_mnist_sgd_mse.png
│   │           convergence_region_mnist_sgd_to_half_cross_entropy.png
│   │           convergence_region_mnist_sgd_to_half_mse.png
```

## Run the code
To run the code from the root folder of the project run `python run.py`.

For more info on which command line arguments can be passed refer to `settings.py` or run `python run.py -h`.

### Prerequisites
- Python 3.7 is installed
- The datasets are automatically downloaded and stored in the `data\` folder in the project root.

### Dependencies
- `numpy>=1.18.4`: Mathematical library, used for various computations
- `pandas>=1.0.3`: Data manipulation library, used for various operations with the tabular experiment logs
- `scikit_learn>=0.23.1`: Machine learning library, used for various standard ML procedures, [reference](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- `torch>=1.3.1`: PyTorch deep learning framework, used for the model definition and training, [reference](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library)
- `torchvision>=0.4.2`: PyTorch library used to load the MNIST and FashionMNIST datasets
- `matplotlib>=3.1.1` and `seaborn>=0.9.0`: Plotting libraries, used for generating and configuring the figures

## Authors
- Mattia Atzeni
- Olivier Cloux
- Andrej Janchevski
