import numpy as np

def col2img(X_col, row, col, row_new, col_new,block_size):
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec
