
# training set-up
train_config = {
    'dataset': 'binarized_MNIST',
    'output_distribution': 'bernoulli',
    'batch_size': 64,
    'n_iterations': 5,
    'learning_rate': 0.0002,
    'kl_min': 0,
    'cuda_device': 0
}

# model architecture
arch = {
    'encoding_form': ['bottom_error', 'top_error'],
    'variable_update_form': 'direct',
    'whiten_input': False,
    'constant_prior_variances': False,

    'top_size': 25,

    'n_latent': [100],

    'n_det_enc': [0],
    'n_det_dec': [0],

    'n_layers_enc': [3],
    'n_layers_dec': [3],

    'n_units_enc': [500],
    'n_units_dec': [500],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'sequential',
    'connection_type_dec': 'sequential',

    'batch_norm_enc': False,
    'batch_norm_dec': False,

    'weight_norm_enc': False,
    'weight_norm_dec': False
}
