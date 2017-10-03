# training set-up
train_config = {
    'dataset': 'MNIST',
    'output_distribution': 'bernoulli',
    'batch_size': 64,
    'n_samples': 1,  # not yet implemented
    'n_iterations': 16,
    'encoder_optimizer': 'adam',
    'decoder_optimizer': 'adam',
    'encoder_learning_rate': 0.0002,
    'decoder_learning_rate': 0.0002,
    'average_gradient': True,
    'encoder_decoder_train_multiple': 1,
    'kl_min': 0,
    'cuda_device': 1,
    'display_iter': 5,
    'resume_experiment': None
}

# model architecture
arch = {
    'model_form': 'dense',  # 'dense', 'conv'

    'encoder_type': 'inference_model',  # 'em', 'inference_model'

    'inference_model_type': 'recurrent',  # 'feedforward', 'recurrent'
    'encoding_form': ['scaled_log_gradient', 'sign_gradient', 'mean', 'log_var'],
    'variable_update_form': 'highway',

    'concat_variables': False,
    'posterior_form': 'gaussian',
    'whiten_input': False,
    'constant_prior_variances': True,
    'learn_top_prior': False,
    'top_size': 1,

    'n_latent': [64],

    'n_det_enc': [0],
    'n_det_dec': [0],

    'n_layers_enc': [1, 0],
    'n_layers_dec': [2, 1],

    'n_units_enc': [512, 0],
    'n_units_dec': [512, 1],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'sequential',
    'connection_type_dec': 'sequential',

    'batch_norm_enc': False,
    'batch_norm_dec': False,

    'weight_norm_enc': False,
    'weight_norm_dec': False,

    'dropout_enc': 0.0,
    'dropout_dec': 0.0
}
