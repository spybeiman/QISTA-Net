import numpy as np

def img2col(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
    return img_col
