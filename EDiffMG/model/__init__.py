from .EDiffMG_3D import EDiffMG



def get_model(config):
    if config.network == 'EDiffMG':
        return EDiffMG(config)

    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
