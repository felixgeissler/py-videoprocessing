import numpy as np
import scipy.fftpack as sft


def filterFunc(frame, r, c):

    # Assets for 8x8 LP filtering
    # for rows:
    Mr = np.ones(8)
    Mr[(int(8/4.0)):r] = np.zeros((int(3.0/4.0*8)))
    # for columns:
    Mc = Mr
    # Grid fpr 8x8 block
    gc = np.zeros((1, c))
    gc[0, 0:c:8] = np.ones(c/8)
    gr = np.zeros((r, 1))
    gr[0:r:8, 0] = np.ones(r/8)

    # First reshape green frame as frame with rows of width 8, (rows: order= 'C' ),
    # and apply DCT to each row of length 8 of all blocks:
    frame = np.reshape(frame[:,:],(-1,8), order='C')
    X = sft.dct(frame/255.0,axis=1,norm='ortho')
    # apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X = np.dot(X,np.diag(Mr))
    # shape it back to original shape:
    X = np.reshape(X,(-1,c), order='C')
    # Shape frame with columns of hight 8 by using transposition .T:
    X = np.reshape(X.T,(-1,8), order='C')
    X = sft.dct(X,axis=1,norm='ortho')
    # apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X = np.dot(X,np.diag(Mc))
    # shape it back to original shape:
    X = (np.reshape(X,(-1,r), order='C')).T
    # Set to zero the 7/8 highest spacial frequencies in each direction:
    # X=X*M
    # frame = np.abs(X) # fancy only

    # Inverse 2D DCT,
    # Rows:
    X = np.reshape(X,(-1,8), order='C')
    X = sft.idct(X,axis=1,norm='ortho')
    # shape it back to original shape:
    X = np.reshape(X,(-1,c), order='C')
    # Shape frame with columns of hight 8 (columns: order='F' convention):
    X = np.reshape(X.T,(-1,8), order='C')
    x = sft.idct(X,axis=1,norm='ortho')
    # shape it back to original shape:
    x = (np.reshape(x,(-1,r), order='C')).T

    return np.around(x*255).astype(np.uint8)
