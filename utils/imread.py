from PIL import Image
import numpy as np

def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    
    mat = np.array(
        [[ 65.481, 128.553, 24.966 ],
         [-37.797, -74.203, 112.0  ],
         [  112.0, -93.786, -18.214]])
    offset = np.array([16, 128, 128])
    
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def imread(imgName,block_size):
    Iorg = np.array(Image.open(imgName), dtype='float32')
    if Iorg.ndim==3:
        Iorg = rgb2ycbcr(Iorg)
        Iorg = np.array(Iorg[:,:,0], dtype='float32')
        img_rec_name = "%s_groundtruth.png" % (imgName)
        Iorg_save = Image.fromarray(Iorg.astype(np.uint8))
        Iorg_save.save(img_rec_name)
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]


