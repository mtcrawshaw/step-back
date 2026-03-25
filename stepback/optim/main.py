from math import sqrt, cos
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
import warnings
from typing import Tuple

from .momo import Momo
from .momo_adam import MomoAdam
from .sps import SPS
from .adabound import AdaBoundW
from .adabelief import AdaBelief
from .lion import Lion
from .muon import Muon
from .nesgd import NESGD

def get_optimizer(opt_config: dict) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'momo':
        opt_obj = Momo
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': False
                  }
    
    elif name == 'momo-adam':
        opt_obj = MomoAdam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': False
                  }
        
    elif name == 'momo-star':
        opt_obj = Momo
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': True
                  }
        
    elif name == 'momo-adam-star':
        opt_obj = MomoAdam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': True
                  }
          
    elif name == 'prox-sps':
        opt_obj = SPS
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'lb': opt_config.get('lb', 0.),
                  'prox': True
                  }
    
    elif name == 'adabound':
        opt_obj = AdaBoundW
        
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-8),
                  'final_lr': opt_config.get('final_lr', 0.1)
                  }

    elif name == 'adabelief':
        opt_obj = AdaBelief
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.95, 0.95)),
                  'eps': opt_config.get('eps', 1e-16),
                  }
        
    elif name == 'lion':
        opt_obj = Lion
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.99)),
                  }
    elif name == 'muon':
        opt_obj = Muon
        lmo = True
        l2_prod_norm = False
        rms_layer_norm = False
        nuc_approx = None
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  }
    elif name == 'muon-gd':
        opt_obj = Muon
        lmo = False
        l2_prod_norm = False
        rms_layer_norm = False
        nuc_approx = None
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  }
    elif name == 'muon-gd-stale':
        opt_obj = Muon
        lmo = False
        l2_prod_norm = False
        rms_layer_norm = False
        nuc_approx = "past"
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  }
    elif name == 'muon-l2':
        opt_obj = Muon
        lmo = True
        l2_prod_norm = True
        rms_layer_norm = False
        nuc_approx = None
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  }
    elif name == 'muon-gd-l2':
        opt_obj = Muon
        lmo = False
        l2_prod_norm = True
        rms_layer_norm = False
        nuc_approx = None
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': True,
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'l2_prod_norm': l2_prod_norm,
                  'nuc_approx': nuc_approx,
                  'rms_layer_norm': rms_layer_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  }
    elif 'nesgd' in name:
        opt_obj = NESGD
        lmo = 'lmo' in name
        if "l2_prod" in name or "hybrid_prod" in name:
            assert not ("l2_prod" in name and "hybrid_prod" in name)
            prod_norm = "l2" if "l2_prod" in name else "hybrid"
        else:
            prod_norm = "linfty"
        nuc_approx = "past" if "stale" in name else None
        if "adam_infty" in name or "adam_2" in name:
            assert not ("adam_infty" in name and "adam_2" in name)
            embed_norm = "adam_infty" if "adam_infty" in name else "adam_2"
        else:
            embed_norm = "linfty"

        lr = opt_config.get('lr', 1e-3)
        if "muon_lr_ratio" in opt_config or "muon_lr" in opt_config:
            assert not ("muon_lr_ratio" in opt_config and "muon_lr" in opt_config)
            if "muon_lr_ratio" in opt_config:
                if lmo:
                    spectral_scale = opt_config["muon_lr_ratio"]
                else:
                    spectral_scale = sqrt(opt_config["muon_lr_ratio"])
            else:
                muon_lr = opt_config["muon_lr"]
                spectral_scale = muon_lr / lr if lmo else sqrt(muon_lr / lr)
        else:
            spectral_scale = 1.0

        truncate_loss = opt_config["truncate_loss"] if "momo" in name else None

        hyperp = {'lr': lr,
                  'wd': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.95),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'lmo': lmo,
                  'prod_norm': prod_norm,
                  'nuc_approx': nuc_approx,
                  'spectral_scale': spectral_scale,
                  'embed_norm': embed_norm,
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'truncate_loss': truncate_loss,
        }

    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_scheduler(config: dict, opt: torch.optim.Optimizer, max_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('lr_schedule', 'constant')

    # default is to step scheduler end of epoch
    # but with this arg we can step scheduler after each step
    step_on_epoch = not config.get('stepwise_schedule')

    warmup_steps = config.get('warmup_steps', 0)
    
    # value is multiplied with initial lr in all cases
    if name == 'constant':
        #lr_fun = lambda t:  warmup_lr + (1-warmup_lr)*t/warmup_steps if t < warmup_steps else 1
        lr_fun = lambda t: 1
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        #lr_fun = lambda t: warmup_lr + (1-warmup_lr)*t/warmup_steps if t < warmup_steps else (t-warmup_steps+1)**(-1/2)
        lr_fun = lambda t: (t+1)**(-1/2)
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs/steps
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)
        
    elif name == 'warmup-linear':
        assert warmup_steps == 0 # gross hack: don't use unified warmup implementation
        assert step_on_epoch

        warmup_epochs = round(config['warmup_fraction'] * max_epoch)

        def get_lr(step):
            if step < warmup_epochs:
                return (step + 1) / warmup_epochs
            else:
                return (max_epoch - step) / (max_epoch - warmup_epochs)

        scheduler = LambdaLR(opt, lr_lambda=get_lr)

    elif name == 'cosine':
        assert step_on_epoch
        warmup_fraction = config.get('warmup_fraction', 0.05)
        warmup_epochs = round(warmup_fraction * max_epoch)

        def get_lr(step):
            if step < warmup_epochs:
                return (step + 1) / warmup_epochs
            else:
                progress = (step - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + cos(math.pi * progress))
        scheduler = LambdaLR(opt, lr_lambda=get_lr)

    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    if warmup_steps > 0:
        warmup_lr = 1e-10
        _warmup = lambda t: warmup_lr + (1-warmup_lr)*t/warmup_steps
        warmup_scheduler = LambdaLR(opt, lr_lambda=_warmup)
        scheduler = SequentialLR(opt, [warmup_scheduler, scheduler], milestones=[warmup_steps])

    return scheduler, step_on_epoch
