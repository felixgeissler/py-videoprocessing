import argparse
import numpy as np
import cv2
import filterFunc
import scipy.signal

parser = argparse.ArgumentParser(description='Processing a video stream.')
parser.add_argument('-d', '--debug',  nargs='?', const='on', metavar='', help='Toggle debug mode (on/off)')
args = parser.parse_args()


def main():

    # Webcam capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    [r, c, d] = frame.shape

    # filter is created by convolving two rectangular filters
    # Triangular filter kernel (pyramidal):
    # N = 2
    # filt1 = np.ones((N,N))/N;
    # filt2 = scipy.signal.convolve2d(filt1, filt1)/N

    print('Processing a {}x{}px videostream with {} color components.'.format(c, r, d))

    while(True):

        # Capturing frame
        ret, frame = cap.read()

        # RGB->YCbCr Conversion
        YCbCr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

        # Splitting components from YCbCr
        Y, Cb, Cr = cv2.split(YCbCr)
        cv2.imshow('YCbCr Cr Component', Cr)

        # Pyramidal Filter maybe cv2.pyrUp
        # Two color components are filtered first
        # Crfilt = scipy.signal.convolve2d(Cr, filt2, mode='same')
        # Cbfilt = scipy.signal.convolve2d(Cb, filt2, mode='same')

        # Downsampling Cb & Cr (slicing via Cb[1::N, 1::N] or cv2.resize method)
        N = 2.0
        dsCb = cv2.resize(Cb, (0, 0), fx=1.0/N, fy=1.0/N, interpolation=cv2.INTER_CUBIC)
        dsCr = cv2.resize(Cr, (0, 0), fx=1.0/N, fy=1.0/N, interpolation=cv2.INTER_CUBIC)

        # 8x8 DCT LP filter
        lpCb = filterFunc.filterFunc(dsCb, int(r/N), int(c/N))
        lpCr = filterFunc.filterFunc(dsCr, int(r/N), int(c/N))

        # Upsampling Cb & Cr
        usCb = cv2.resize(lpCb, (0, 0), fx=N, fy=N, interpolation=cv2.INTER_CUBIC)
        usCr = cv2.resize(lpCr, (0, 0), fx=N, fy=N, interpolation=cv2.INTER_CUBIC)

        # Merge components to YCbCr
        resYCbCr = cv2.merge((Y, usCb, usCr))

        # YcbCr->RGB Conversion
        RGB = cv2.cvtColor(resYCbCr, cv2.COLOR_YCrCb2RGB)

        cv2.imshow('Original RGB', frame)
        cv2.imshow('Processed RGB', RGB)

        if(args.debug == 'on'):
            cv2.imshow('YCbCr Cr Component', Cr)
            cv2.imshow('Downsampled Cr Component', dsCr)
            cv2.imshow('LP-filtered Cr Component', lpCr)
            cv2.imshow('Upsampled Cr Component CUB', usCr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if(args.debug == 'on'):
        print('Debugmode: on')
    else:
        print('Debugmode: off')
    main()
