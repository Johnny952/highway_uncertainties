import sys
import os

sys.path.append('..')
from shared.utils.plot import plot_time

if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")

    test_paths = [
        "uncertainties/test/ae.txt",
        # "uncertainties/test/ae.txt",
        "uncertainties/test/bnn.txt",
    ]
    names = [
        "AE",
        # "AE",
        "BNN",
    ]
    uncertainties = [
        1,
        # 2,
        1,
    ]
    log_scales = [
        False,
        # False,
        False,
    ]
    # python test.py -ES 20 -M ae -FC param/best_ae_trained.pkl -ER -EV 5
    plot_time(test_paths, names, log_scales, uncertainties, red_lines=[11, 21], nb_eval=4)