training.py - running the training procedure
mymodels.py - configuring the model classes, architectures etc.
dataset_module.py - all the dataset load and transformation code
config.py - training hyperparams
hierarchy.py - maps image ids according to the hierarchy
hier_plot.py - sunburst plot of hierarchy
trained_models.py - find final epochs of saved models, save in a dict
evaluation.py - compute eval metrics from saved dict
confusion_matrix_gen.py - generate confusion matrices from saved models, turn into pickle
load_cms.py - extract CMs from pickle
cm_sse.py - calculate L2 norms of CMs
cm_comparison.py - side-by-side plots of CMs
eval_plot.py - hierarchical accuracy slope plots
tb_plot_histogram.py - extract activations histogram from TensorBoard (appendix - sigmoid saturation)
models_management.py - bulk delete saved models to save space and run training again
