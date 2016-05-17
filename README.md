Chongxuan Li, Jun Zhu and Bo Zhang, [Learning to Generate with Memory](http://arxiv.org/abs/1602.07416) (ICML16).
Please cite this paper when using this code for your research.

Some lines of code are redundent. TODO: a more clean version with same reproducibility.

For questions and bug reports, please send me an e-mail at _chongxuanli1991[at]gmail.com_.


## Prerequisites

1. Some libs we used in our experiments:
    - Python (version 2.7)
    - Numpy
    - Scipy
    - Theano (version 0.8.0)
    - Lasagne (version 0.2.dev1)
    - Parmesan (version 0.1.dev1)
    - matplotlib
    - networkx

2. Download OCR-letters

    - wget http://www.capsec.org/datasets/ocr_letters.h5

## Log-likelihood estimation and random generation:
1. MNIST (set iw_samples=5 or 50 for comparision with IWAE)
    - VAE: python mem_dgm_mlp.py -dataset sample -has_lre 0,0,0 -n_slots 0,0,0 -lambdas 0,0,0 -has_memory 0,0,0 -iw_samples 1
    - Large_VAE: python mem_dgm_mlp.py -dataset sample -has_lre 0,0,0 -n_slots 0,0,0 -lambdas 0,0,0 -has_memory 0,0,0 -iw_samples 1 -n_hiddens 530,530
    - MEM_VAE: python mem_dgm_mlp.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -iw_samples 1

2. OCR-letters (set iw_samples=5 or 50 for comparision with IWAE)
    - VAE: python mem_dgm_mlp.py -dataset ocr_letter -mode valid -has_lre 0,0,0 -n_slots 0,0,0 -lambdas 0,0,0 -has_memory 0,0,0 -drops_enc 0,0 -iw_samples 1 -n_layers 2 -n_hiddens 200,200 -nlatent 50
    - MEM_VAE: python mem_dgm_mlp.py -dataset ocr_letter -mode valid -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -drops_enc 0,0 -iw_samples 1 -n_layers 2 -n_hiddens 200,200 -nlatent 50

3. Frey faces (set batch_size=100 and nepochs=3000 to achieve better log-likelihood)
    - VAE: python mem_dgm_mlp.py -dataset fray_faces -has_lre 0,0 -n_slots 0,0 -lambdas 0,0 -has_memory 0,0 -iw_samples 1 -n_layers 1 -drops_enc 0 -n_hiddens 200 -nlatent 10 -batch_size 10 -nepochs 1000
    - MEM_VAE: python mem_dgm_mlp.py -dataset fray_faces -has_lre 0,0 -n_slots 0,20 -lambdas 0,0 -has_memory 0,1 -iw_samples 1 -n_layers 1 -drops_enc 0 -n_hiddens 200 -nlatent 10 -batch_size 10 -nepochs 1000

## Missing value imputation 

The results may be slightly different with those in the paper because the noisy data are generated randomly. However, the gap remains the same.

1. Generate different types of noisy data
    - rectangle: python generate_pertubed_data_mnist.py 3 12 (size of rectangle, an even number less than 28)
    - random: python generate_pertubed_data_mnist.py 4 0.6 (drop ratio, a real number in range (0, 1))
    - half: python generate_pertubed_data_mnist.py 5 0 14 (integer less than 28)

2. Test with noisy data (set imputation_mode=random or rectangle and imputation_para=0.6 or 12 for other types of noise)
    - VAE: python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,0,0 -n_slots 0,0,0 -lambdas 0,0,0 -has_memory 0,0,0 -analysis_mode imputation -imputation_mode half -imputation_para 14 -model_file [dir/modelepoch3000]
    - MEM_VAE: python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode imputation -imputation_mode half -imputation_para 14 -model_file [dir/modelepoch3000]

## Classification

1. Get features
    - VAE: python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,0,0 -n_slots 0,0,0 -lambdas 0,0,0 -has_memory 0,0,0 -analysis_mode classification -model_file [dir/modelepoch3000]
    - MEM_VAE: python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode classification -model_file [dir/modelepoch3000]

2. Test with linear SVM    
    - python svm.py [feature_file]

## Disable memory for visualization

1. Only for MEM_VAE
    - python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode visualization -model_file [dir/modelepoch3000]

## Compute statistics and visualize preference of memory slots over classes

1. Compute statistics 
    - python mem_dgm_mlp_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode statis_computation -model_file [dir/modelepoch3000]

2. Visualize (may have different layout as shown in the paper)
    - python plot [dir/mem_cov.mat]

## Map memory slots of a simpler model to data level

1. Train a model with softmax attention and element-wise addition composition function
    - python mem_dgm_mlp_for_vis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -iw_samples 1 -com_type plus -atten_type normalized

2. Analyze the sparsity and preference of memory slots over different classes
    - python mem_dgm_mlp_for_vis_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode sparse_over_class -com_type plus -atten_type normalized -model_file [dir/modelepoch3000]

3. Map memory slots to images
    - python mem_dgm_mlp_for_vis_analysis.py -dataset sample -has_lre 0,1,1 -n_slots 0,70,30 -lambdas 0,0.1,0.1 -has_memory 0,1,1 -analysis_mode visualization_mem -com_type plus -atten_type normalized -model_file [dir/modelepoch3000]