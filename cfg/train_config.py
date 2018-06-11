# training set-up
train_config = {
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
    'kl_warm_up': True,
}
