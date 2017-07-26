

def train(model, data, optimizers):

    enc_opt, dec_opt = optimizers

    # todo: batch the data
    
    x = None

    enc_opt.zero_grad()

    for i in range(n_iter-1):
        model.encode(x)
        model.decode()
        elbo = model.ELBO(x)
        elbo.backward(retain_variables=True)

    dec_opt.zero_grad()
    model.encode(x)
    model.decode()
    elbo = model.ELBO(x)
    elbo.backward()

    enc_opt.step()
    dec_opt.step()


def validate(model, data_labels):
    pass