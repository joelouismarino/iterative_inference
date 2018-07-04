# model set-up
model_config = {
    'inference_input_form': ['gradient', 'observation'], # 'observation', 'gradient', 'error'

    'constant_variances_gen': False,

    'output_dist': 'normal',

    'n_latent': [1024],

    'n_layers_inf': [3],
    'n_layers_gen': [1],

    'n_units_inf': [2048],
    'n_units_gen': [2048],

    'non_linearity_inf': 'elu',
    'non_linearity_gen': 'elu',

    'connection_type_inf': 'highway',
    'connection_type_gen': 'highway',

    'batch_norm_inf': False,
    'batch_norm_gen': False,
}
