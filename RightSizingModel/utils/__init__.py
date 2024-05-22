from .logger1 import log, log_test, eval_log
from .plotter1 import plot_loss_curve, plot_recon_kl_curve

import os

def log_model(model, output_dir):
    with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
        print(model, file=f)
