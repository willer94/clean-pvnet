from .resnet18  import get_res_pvnet


_network_factory = {
    'res': get_res_pvnet
}


def get_network(cfg):
    arch = cfg.network
    get_model = _network_factory[arch]
    spherical_used = cfg.loss != 'norm'
    network = get_model(cfg.heads['vote_dim'], cfg.heads['seg_dim'], spherical_used)
    return network
