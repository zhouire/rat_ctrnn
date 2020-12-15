# rat_ctrnn

9.49 Final Project: attempt to reproduce the results of Cueva & Wei 2018 (Emergence of grid-like representations by training recurrent neural
networks to perform spatial localization) with an LSTM structure, with spatially organized neurons via weight masking. 

navigation_LSTM.ipynb: Code for the model and visualization, with optional parameters for masking and regularization. Works best in Google Colab with GPU.
movement.py: code to generate animal navigation dataset.
rat_ctrnn_data.p: animal navigation dataset containing 2000 paths of 500 timesteps each (generated using movement.py) in a square arena.
rat_ctrnn_data_randstart.p: similar animal navigation dataset with random starting points.
LSTM_masked_noreg_trainedmodel_10000: contains a trained model with masking and no regularization, after 10000 epochs. 


Cueva, C. J., & Wei, X. X. (2018). Emergence of grid-like representations by training recurrent neural
networks to perform spatial localization. arXiv preprint arXiv:1803.07770.
