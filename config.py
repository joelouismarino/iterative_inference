# training set-up
train_config = {
    'dataset': 'CIFAR_10',
    'output_distribution': 'gaussian',
    'batch_size': 64,
    'n_iterations': 5,
    'encoder_learning_rate': 0.0002,
    'decoder_learning_rate': 0.0002,
    'average_gradient': True,
    'kl_min': 0,
    'cuda_device': 1,
    'display_iter': 25,
    'resume_experiment': None
}

# model architecture
arch = {
    'encoding_form': ['posterior', 'top_norm_error', 'bottom_norm_error'],
    'concat_variables': False,
    'variable_update_form': 'highway',
    'posterior_form': 'point_estimate',
    'whiten_input': False,
    'constant_prior_variances': True,
    'learn_top_prior': False,
    'top_size': 25,

    'n_latent': [256],

    'n_det_enc': [0],
    'n_det_dec': [0],

    'n_layers_enc': [3, 0],
    'n_layers_dec': [1, 1],

    'n_units_enc': [1024, 0],
    'n_units_dec': [1024, 1],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'highway',
    'connection_type_dec': 'highway',

    'batch_norm_enc': False,
    'batch_norm_dec': False,

    'weight_norm_enc': False,
    'weight_norm_dec': False,

    'dropout_enc': 0.0,
    'dropout_dec': 0.0
}
