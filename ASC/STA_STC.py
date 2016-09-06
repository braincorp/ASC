import os
import sys
import numpy as np
import numpy.linalg
import pyximport
pyximport.install(reload_support=True)
import ASC
import cPickle as pickle
from matplotlib import pylab as plt

if __name__ == '__main__':
    assert len(sys.argv) == 2, "USAGE: python STA_STC.py [path to model.pkl]"
    filename = sys.argv[1]

    SM=pickle.load(open(filename, "r"))
    SM.stop_learning()

    num_frames = 10000
    feed_multiple = 2

    V1S_STAs = 0
    V1C_STAs = 0
    V1S_STCs = 0
    V1C_STCs = 0
    outer_stim = np.zeros((num_frames, SM.Vs[0].m * SM.Vs[0].m))

    for k in range(100):
        stim=np.random.randn(num_frames, SM.Vs[0].m)
        V1S=[]
        V1C=[]
        im = np.zeros((80, 80, 3), dtype=np.uint8)
        for i in range(num_frames):
            im[0:10, 0:10, :] = np.minimum(255, np.maximum(0, stim[i, :].reshape((10,10,3))*127/3+127.5))
            for j in range(feed_multiple):
                respS, respC = SM.feed(im)
            V1S.append(respS[0][:,0].todense())
            V1C.append(respC[0][:,0])

        V1S=np.array(V1S).squeeze()
        V1C=np.array(V1C).squeeze()
        
        V1S_STAs = V1S_STAs + ASC.parallel_dot(V1S.T, stim, num_threads=20)
        V1C_STAs = V1C_STAs + ASC.parallel_dot(V1C.T, stim, num_threads=20)

        print k
        SM.Vs[0].show((V1S_STAs/(0.0001+np.sqrt(np.sum(V1S_STAs**2, axis=1)[...,None])))[0:SM.Vs[0].K,:].T, "V1S_STA_%d_"%k)
        SM.Vs[0].show((V1C_STAs/(0.0001+np.sqrt(np.sum(V1C_STAs**2, axis=1)[...,None])))[0:SM.Vs[0].K,:].T, "V1C_STA_%d_"%k)

        for i in range(num_frames):
            outer_stim[i, :] = np.outer(stim[i, :], stim[i, :]).reshape((1, -1))

        V1S_STCs = V1S_STCs + ASC.parallel_dot(V1S.T, outer_stim, num_threads=20).astype(np.float32)
        V1C_STCs = V1C_STCs + ASC.parallel_dot(V1C.T, outer_stim, num_threads=20).astype(np.float32)

    # save data for later
    pickle.dump((V1S_STCs, V1C_STCs), open("STCs.pkl", "w"))

    V1S_STCs=V1S_STCs.reshape((-1, SM.Vs[0].m, SM.Vs[0].m))
    V1C_STCs=V1C_STCs.reshape((-1, SM.Vs[0].m, SM.Vs[0].m))

    for i in range(0, SM.Vs[0].K, SM.Vs[0].K/20):
        u, e, v = numpy.linalg.svd(V1C_STCs[i,:,:].squeeze(), full_matrices=0, compute_uv=1)
        SM.Vs[0].show(u[:, np.hstack((np.arange(0, 50), np.arange(-50, 0)))], "STC_%d_"%i)
        plt.clf()
        plt.plot(e/np.mean(e),'.')
        plt.axis("off")
        plt.gcf().set_size_inches(1, 1)
        plt.xlim(-10, 310)
        plt.savefig("Eigvals_%d"%i, dpi=100)

