# training set-up
train_config = {
    'dataset': 'binarized_MNIST',
    'output_distribution': 'bernoulli',
    'batch_size': 256,
    'n_iterations': 5,
    'encoder_learning_rate': 0.0002,
    'decoder_learning_rate': 0.0002,
    'kl_min': 0.,
    'cuda_device': 1,
    'resume_experiment': None
}

# model architecture
arch = {
    'encoding_form': ['posterior', 'bottom_error', 'top_error'],
    'concat_variables': False,
    'variable_update_form': 'highway',
    'whiten_input': False,
    'constant_prior_variances': False,
    'top_size': 25,

    'n_latent': [64, 32, 16],

    'n_det_enc': [64, 32, 0],
    'n_det_dec': [64, 32, 16],

    'n_layers_enc': [2, 2, 2, 2],
    'n_layers_dec': [2, 2, 2, 2],

    'n_units_enc': [512, 512, 512, 0],
    'n_units_dec': [512, 512, 512, 512],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'highway',
    'connection_type_dec': 'highway',

    'batch_norm_enc': True,
    'batch_norm_dec': True,

    'weight_norm_enc': False,
    'weight_norm_dec': False,

    'dropout_enc': 0.,
    'dropout_dec': 0.
}
