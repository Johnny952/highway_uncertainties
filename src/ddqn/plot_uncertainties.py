import sys
import os

sys.path.append('..')
from shared.utils.plot import plot_time

if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")

    test_paths = [
        "uncertainties/test/vae.txt",
        "uncertainties/test/vae.txt",
    ]
    names = [
        "VAE",
        "VAE"
    ]
    uncertainties = [
        1,
        2,
    ]
    log_scales = [
        False,
        False,
    ]
    # python test.py -ES 20 -M vae -FC param/best_vae_trained_A -ER -EV 5
    plot_time(test_paths, names, log_scales, uncertainties, red_lines=[10, 20], nb_eval=0)