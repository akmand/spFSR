# spFSR
This repository is a collection of methods using Simultaneous Perturbation Stochastic Approximation (SPSA) for feature selection and ranking in Python. It currently contains the following: 

(1) An implementation of feature selection and ranking via SPSA based on the article "K-best feature selection and ranking via stochastic approximation"(https://www.sciencedirect.com/science/article/abs/pii/S0957417422018826) This algorithm searches for a locally optimal set of features that yield the best predictive performance using a specified error measure such as mean squared error (for regression problems) and accuracy rate (for classification problems). This particular implementation makes use of Barzilai and Borwein non-monotone gains for much faster convergence. The related files are spFSR.py and spFSR_example_github.py.

(2) An implementation of feature selection and weighting via SPSA. The related files are SpFtWgt.py and spFtWgt_example_github.py.
