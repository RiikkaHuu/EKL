README for Entangled Kernel Learning code publication
Code written by Riikka Huusari 

@inproceedings{huusari2019entangled,
  title={Entangled Kernels},
  author={Huusari, Riikka and Kadri, Hachem },
  booktitle={28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2019}
}


The EKL algorithm appears here as explained in the above publication, with an illustration that appears in the JMLR version of the paper.


This directory contains the following code files:
* EKL_algo.py
* exp_digits_dimredct.py
only the first one will be useful for people wanting to apply EKL to their problems. 


EKL_algo.py contains the EKL algorithm implementation (class EKL) along with the helper functions. If you want to apply EKL to your machine learning problem this file is all you need. 
To use EKL, you should do something like the following:  

  from EKL_algo import EKL
  # 1) initialize EKL object, give it data, labels and parameters
  ekl = EKL(features, labels, l, r, g)  # l: lambda, r: rank, g: gamma  (as in the EKL paper)
  # 2) solve the kernel learning problem and optimize the predictive function
  ekl.solve()  # manifold optimisation can be slow... 
  # 3) predict with EKL
  predictions = ekl.predict(test_features)
  ptr_predictions = ekl.ptr_predict(test_features)  # the partial trace formulation, see paper

If you want to predict with new lambda values but keeping the entangled kernel (gamma parameter) the same, you can do it by

  predictions = ekl.predict(test_features, l=new_lambda)
  ptr_predictions = ekl.ptr_predict(test_features, l=new_lambda)


exp_digits_dimredct.py contains the code for small illustration of EKL that appears in the JMLR version of the paper. It samples the digits dataset and runs EKL with various parameters, and finally displays the results shown in the illustration. 
