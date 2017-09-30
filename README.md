# Iterative Inference
Implementation of iterative inference in deep latent variable models

# To-do
- [ ] Figure out why we're using GPU0. Suspect it's from the data loader.
- [x] Recurrent encoding networks.
- [x] Direct gradient encoding.
- [x] Variational EM training, evaluation.
- [ ] Multiple samples/estimates.
- [ ] Convolutional model.
- [ ] Normalizing flows and IAF.
- [ ] Random re-initialization of modules during training.
- [ ] Clean up code so that we don't need top_size in config.
- [ ] Parallelize across GPUs. Involves re-writing models and latent levels.
- [x] Implement phase training (train just the encoder or just the decoder).


# Visualization to-do
- [x] Visualize errors at the input level and reconstructions over inference iterations.
- [ ] Latent traversal of lowest variance dimensions.
- [ ] Plot number of inactive units.
- [x] 2D latent visualization.

