# training set-up
train_config = {
    'dataset': 'rcv1',
    'output_distribution': 'multinomial',
    'batch_size': 512,
    'n_samples': 1,
    'n_iterations': 5,
    'encoder_optimizer': 'adam',
    'decoder_optimizer': 'adam',
    'encoder_learning_rate': 0.0001,
    'decoder_learning_rate': 0.0001,
    'average_gradient': True,
    'encoder_decoder_train_multiple': 1,
    'kl_min': 0,
    'kl_warm_up': False,
    'cuda_device': 0,
    'display_iter': 5,
    'eval_iter': 2000,
    'resume_experiment': None
}

# model architecture
arch = {
    'model_form': 'dense',  # 'dense', 'conv'

    'encoder_type': 'inference_model',  # 'em', 'inference_model'

    'inference_model_type': 'feedforward',  # 'feedforward', 'recurrent'
    'encoding_form': ['posterior', 'mean', 'log_var', 'mean_gradient', 'log_var_gradient'],
    'variable_update_form': 'highway',

    'concat_variables': False,
    'posterior_form': 'gaussian',
    'whiten_input': False,
    'constant_prior_variances': False,
    'single_output_variance': False,
    'learn_top_prior': False,
    'top_size': 1,

    'n_latent': [256],

    'n_det_enc': [0],
    'n_det_dec': [0],

    'n_layers_enc': [2, 0],
    'n_layers_dec': [2, 1],

    'n_units_enc': [512, 0],
    'n_units_dec': [512, 1],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'highway',
    'connection_type_dec': 'sequential',

    'batch_norm_enc': False,
    'batch_norm_dec': False,

    'weight_norm_enc': False,
    'weight_norm_dec': False,

    'dropout_enc': 0.0,
    'dropout_dec': 0.0
}
