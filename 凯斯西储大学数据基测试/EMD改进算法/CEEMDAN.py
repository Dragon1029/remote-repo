import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD

# Load .mat file
mat_data = scipy.io.loadmat('D:/git_test/origin/凯斯西储大学数据基测试/十分类数据集/12k/105.mat')

# Extract data
x = mat_data['X105_DE_time'].flatten()
x = torch.tensor(x, dtype=torch.float32)


# Define CEEMDAN function
def ceemdan(x, Nstd, NR, MaxIter):
    x = x.flatten()
    desvio_x = torch.std(x)
    x = x / desvio_x

    modes = torch.zeros_like(x)
    temp = torch.zeros_like(x)
    aux = torch.zeros_like(x)
    acum = torch.zeros_like(x)
    iter = torch.zeros((NR, int(np.log2(len(x))) + 5))

    white_noise = [torch.randn_like(x) for _ in range(NR)]
    modes_white_noise = [EMD().emd(white_noise[i]) for i in range(NR)]

    for i in range(NR):
        temp = x + Nstd * white_noise[i]
        temp, _, it = EMD().emd(temp, max_modes=1, max_imf=1, num_siftings=MaxIter)
        temp = temp[0]
        aux += temp / NR
        iter[i, 0] = it

    modes = aux.clone()
    k = 1
    aux = torch.zeros_like(x)
    acum = torch.sum(modes, dim=0)

    while torch.nonzero(torch.diff(torch.sign(torch.diff(x - acum)))).size(0) > 2:
        for i in range(NR):
            tamanio = modes_white_noise[i].shape
            if tamanio[0] >= k + 1:
                noise = modes_white_noise[i][k]
                noise = noise / torch.std(noise)
                noise = Nstd * noise
                try:
                    temp, _, it = EMD().emd(x - acum + torch.std(x - acum) * noise, max_modes=1, max_imf=1, num_siftings=MaxIter)
                    temp = temp[0]
                except:
                    it = 0
                    temp = x - acum
            else:
                temp, _, it = EMD().emd(x - acum, max_modes=1, max_imf=1, num_siftings=MaxIter)
                temp = temp[0]
            aux += temp / NR
            iter[i, k] = it
        modes = torch.cat((modes, aux.unsqueeze(0)), dim=0)
        aux = torch.zeros_like(x)
        acum = torch.zeros_like(x)
        acum = torch.sum(modes, dim=0)
        k += 1

    modes = torch.cat((modes, (x - acum).unsqueeze(0)), dim=0)
    a, b = modes.shape
    iter = iter[:, :a]
    modes = modes * desvio_x

    return modes, iter

# Parameters
Nstd = 0.2  # Noise standard deviation
NR = 100  # Number of realizations
MaxIter = 100  # Maximum number of sifting iterations

# Perform CEEMDAN
modes, _ = ceemdan(x, Nstd, NR, MaxIter)

# Plot original signal and modes
plt.figure(figsize=(10, 6))
plt.plot(x, label='Original Signal', color='black')
for i in range(modes.shape[0]):
    plt.plot(modes[i], label=f'Mode {i+1}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Original Signal and Modes')
plt.legend()
plt.grid(True)
plt.show()
