import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def img_2_save(cor,recon,filename):
    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.title('input')
    cor_plt = Image.fromarray(np.clip(cor, 0, 255).astype(np.uint8))
    plt.imshow(cor_plt,cmap='gray')
    
    plt.subplot(122)
    plt.title('reconstruction')
    recon_plt = Image.fromarray(np.clip(recon, 0, 255).astype(np.uint8))
    plt.imshow(recon_plt,cmap='gray')
    
    plt.savefig(filename)
    plt.show()
    
    
