# training set-up
train_config = {
    'dataset': 'CIFAR_10',
    'output_distribution': 'gaussian',
    'batch_size': 128,
    'n_samples': 10,
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
    'display_iter': 30,
    'eval_iter': 2000,
    'resume_experiment': None
}

# model architecture
arch = {
    'model_form': 'dense',  # 'dense', 'conv'

    'encoder_type': 'inference_model',  # 'em', 'inference_model'

    'inference_model_type': 'feedforward',  # 'feedforward', 'recurrent'
    'encoding_form': ['posterior', 'layer_norm_mean_gradient', 'layer_norm_log_var_gradient', 'mean', 'log_var'],
    'variable_update_form': 'highway',

    'concat_variables': True,
    'posterior_form': 'gaussian',
    'whiten_input': False,
    'constant_prior_variances': False,
    'single_output_variance': False,
    'learn_top_prior': False,
    'top_size': 1,

    'n_latent': [1024, 512],

    'n_det_enc': [0, 0],
    'n_det_dec': [0, 0],

    'n_layers_enc': [3, 3, 0],
    'n_layers_dec': [1, 1, 1],

    'n_units_enc': [2048, 2048, 0],
    'n_units_dec': [2048, 2048, 1],

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
