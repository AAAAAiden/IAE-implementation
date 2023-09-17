
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import os

import numpy as np
import dataset

def generate_data(m):
    num_points = 60000#00  # Adjusted for more data points
    nu = np.random.randn(num_points)

    # 1. Moving Average (MA)
    ma_window_size = 10
    xt_ma = [np.mean(nu[i - ma_window_size:i]) for i in range(ma_window_size, num_points)]

    # 2. Linear Autoregressive (LAR)
    xt_lar = [nu[0]]
    for i in range(1, num_points):
        xt_lar.append(0.5 * xt_lar[i - 1] + nu[i])


    # 3. Nonlinear Autoregressive (NLAR)
    xt_nlar = [nu[0], nu[1]]  # initialize with the first two elements
    for i in range(2, num_points):
        xt_nlar.append(0.5 * xt_nlar[i - 1] + 0.41 * (xt_nlar[i - 2] < 0.7) + nu[i])


    # Creating training windows (For synthetic data)
    windows_ma = [xt_ma[i:i+m] for i in range(0, len(xt_ma) - m + 1)]
    windows_lar = [xt_lar[i:i+m] for i in range(0, len(xt_lar) - m + 1)]
    windows_nlar = [xt_nlar[i:i+m] for i in range(0, len(xt_nlar) - m + 1)]

# You can then use windows_ma, windows_lar, and windows_nlar for training your IAE.
    np.save('../data/windows_ma.npy', windows_ma)
    np.save('../data/windows_lar.npy', windows_lar)
    np.save('../data/windows_nlar.npy', windows_nlar)
if __name__ == "__main__":
    # [same as before: package_path and files]
    m = 20  # This should be your chosen value of m from the paper.
    generate_data(m)