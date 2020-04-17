#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : __init__.py
# Author            : WangZi
# Date              : 15.04.2020
# Last Modified Date: 15.04.2020
# Last Modified By  : WangZi
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
