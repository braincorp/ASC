# ==================================================================================
# Copyright (c) 2016, Brain Corporation
#
# This software is released under Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0 (BY-NC-SA) license.
# Full text available here in LICENSE.TXT file as well as:
# https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode
#
# In summary - you are free to:
#
#    Share - copy and redistribute the material in any medium or format
#    Adapt - remix, transform, and build upon the material
#
# The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
#    * Attribution - You must give appropriate credit, provide a link to the
#                    license, and indicate if changes were made. You may do so
#                    in any reasonable manner, but not in any way that suggests
#                    the licensor endorses you or your use.
#    * NonCommercial - You may not use the material for commercial purposes.
#    * ShareAlike - If you remix, transform, or build upon the material, you
#                   must distribute your contributions under the same license
#                   as the original.
#    * No additional restrictions - You may not apply legal terms or technological
#                                   measures that legally restrict others from
#                                   doing anything the license permits.
# ==================================================================================

import cv2
import os.path
import numpy as np
import cPickle as pickle
import multiprocessing as mp
from ASC.image_sparser import ImageSparser

import pyximport
pyximport.install(reload_support=True)
import ASC_utils


class SparseManager(object):
    def __init__(self, name='SparseManager', im_shape=(80, 80), have_display=True, feed_multiple=1):
        self.im_shape = im_shape
        self.Vs = []
        self.done_priming = False
        self.priming_steps = 0
        self.num_resets = 0
        self.name = name
        self.have_display = have_display
        self.feed_multiple = feed_multiple
        self.prime_data = None
        self.prime_labels = None
        
    def __setstate__(self, dict):  # called when unpickled
        self.__dict__.update(dict)
        
        if 'V1' in dict:
            self.Vs = [Vn for Vn in [self.V1, self.V2, self.V3, self.V4, self.V5] if Vn is not None]
        if 'feed_multiple' not in dict:
            self.feed_multiple = 1
        if 'prime_data' not in dict:
            self.prime_data = None
            self.prime_labels = None
        
    def create_V1(self, Xb=8, Yb=8, num_color=3, **kwargs):
        assert (self.im_shape[0]/Xb)*Xb==self.im_shape[0]
        assert (self.im_shape[1]/Yb)*Yb==self.im_shape[1]

        V1 = ImageSparser(name=self.name+"_V1", Xb=Xb, Yb=Yb, num_color=num_color, **kwargs)
        if len(self.Vs) < 1:
            self.Vs.append([])
        self.Vs[0] = V1

    def create_Vn(self, n=2, **kwargs):
        assert len(self.Vs) >= n-1, "must create V%d before you create V%d" % (n-1, n)
        assert ((self.im_shape[0]/self.Vs[0].Xb) % 2**(n-1)) == 0, "number of X tiles must be multiple of %d" % (2**(n-1))
        assert ((self.im_shape[1]/self.Vs[0].Yb) % 2**(n-1)) == 0, "number of Y tiles must be multiple of %d" % (2**(n-1))
        
        Vn = ImageSparser(name=self.name+"_V%d"%(n), Xb=(self.Vs[0].K+1)*4, Yb=1, **kwargs)
        if len(self.Vs) < n:
            self.Vs.append([])
        self.Vs[n-1] = Vn

    def create_all(self, num_levels=4, Xb=8, Yb=8, num_color=3, **kwargs):
        self.create_V1(Xb=Xb, Yb=Yb, num_color=num_color, **kwargs)

        for i in range(1, num_levels):
            n = i + 1
            self.create_Vn(n, top_level=n==num_levels, **kwargs)
        
    def load_all(self, base_filename):
        def get_filename(base_filename, n):
            return base_filename+"_V%d.pkl" % (n)
            
        self.Vs = []
        n = 1
        while os.path.exists(get_filename(base_filename, n)):
            Vn = pickle.load(open(get_filename(base_filename, n), "r"))
            if n==1:
                assert (self.im_shape[0]/Vn.Xb)*Vn.Xb==self.im_shape[0]
                assert (self.im_shape[1]/Vn.Yb)*Vn.Yb==self.im_shape[1]
            else:
                assert ((self.im_shape[0]/self.Vs[0].Xb) % 2**(n-1)) == 0, "number of X tiles must be multiple of %d" % (2**(n-1))
                assert ((self.im_shape[1]/self.Vs[0].Yb) % 2**(n-1)) == 0, "number of Y tiles must be multiple of %d" % (2**(n-1))
            self.Vs.append(Vn)
            n += 1
        assert n > 1, "no level files found with base filename %s" % base_filename   
                               
    def stop_learning(self):
        for V in self.Vs:
            V.learning = False
            
    def group_NxN_input(self, im, N, numInput, numX, numY):
        return im.reshape((numInput, numX/N, N, numY/N, N)).transpose((2, 4, 0, 1, 3)).reshape((N*N*numInput, -1))
        
    def gen_context(self, Vn, numX, numY):
        if Vn is None:
            return None

        if Vn.use_feedback == 4:  # allow all 4 tiles to have different feedback weights, very expensive
            context = np.zeros((4, Vn.K+1, numX, numY))
            if Vn.prev_Complex_alpha is not None:
                c = Vn.prev_Complex_alpha.reshape((Vn.K+1, numX/2, numY/2))
                context[0, :, 0::2, 0::2] = c
                context[1, :, 1::2, 0::2] = c
                context[2, :, 0::2, 1::2] = c
                context[3, :, 1::2, 1::2] = c
            
            return context.reshape((4*(Vn.K+1), numX*numY)).T
        else:
            context = np.zeros((Vn.K+1, numX, numY))
            if Vn.prev_Complex_alpha is not None:
                c = Vn.prev_Complex_alpha.reshape((Vn.K+1, numX/2, numY/2))
                context[:, 0::2, 0::2] = c
                context[:, 1::2, 0::2] = c
                context[:, 0::2, 1::2] = c
                context[:, 1::2, 1::2] = c
            
            return context.reshape((Vn.K+1, numX*numY)).T
    
    def feed(self, im):
        assert im.dtype == np.uint8
        im = cv2.resize(im, dsize=self.im_shape, interpolation=cv2.INTER_AREA)
        im = im - 127.5
        
        ss = [None]*len(self.Vs)
        cs = [None]*len(self.Vs)
        inp = im
        for i, Vn in enumerate(self.Vs):
            n = i + 1
            if self.Vs[i].use_feedback and (i+1) < len(self.Vs):
                context = self.gen_context(self.Vs[i+1], self.im_shape[0]/self.Vs[0].Xb/(2**i), self.im_shape[1]/self.Vs[0].Yb/(2**i))
            else:
                context = None  # top level doesn't have any feedback
            s, c = self.Vs[i].sparsify(inp, context=context)
            ss[i] = s
            cs[i] = c
            
            if c is None:  # sparsify returns None if a layer isn't trained enough to return a response
                break
            if n < len(self.Vs):
                # input for next level
                inp = self.group_NxN_input(c, 2, self.Vs[0].K+1, self.im_shape[0]/self.Vs[0].Xb/(2**i), self.im_shape[1]/self.Vs[0].Yb/(2**i))
        
        return ss, cs
        
    def prime(self, im, priming_mask, debug=False):
        self.priming_steps += 1
        assert not self.done_priming

        if self.have_display and debug:
            cv2.imshow("im", im)
        
        for i in range(self.feed_multiple):
            ss, cs = self.feed(im)

        if self.prime_data is None:
            self.prime_data = [[] for v in self.Vs]
            self.prime_labels = [[] for v in self.Vs]

        V1_Xb = self.Vs[0].Xb
        V1_Yb = self.Vs[0].Yb
        num_Xb = self.im_shape[0]/V1_Xb
        num_Yb = self.im_shape[1]/V1_Yb
        for i, c in enumerate(cs):
            n = i + 1
            s = 2**i
            mask = (cv2.resize(priming_mask, dsize=(self.im_shape[0]/s, self.im_shape[1]/s), interpolation=cv2.INTER_AREA) > 127)*1.0
            tmp_im = cv2.resize(im, dsize=(self.im_shape[0]/s, self.im_shape[1]/s), interpolation=cv2.INTER_AREA)
            
            Vn_mask = mask.reshape((num_Xb/s, V1_Xb, num_Yb/s, V1_Yb, 1)).transpose((1, 3, 4, 0, 2)).reshape((V1_Xb*V1_Yb, num_Xb*num_Yb/s/s))
            tmp_im = tmp_im.reshape((num_Xb/s, V1_Xb, num_Yb/s, V1_Yb, 3)).transpose((1, 3, 4, 0, 2)).reshape((V1_Xb*V1_Yb*3, num_Xb*num_Yb/s/s))

            for j in range(c.shape[1]):
                self.prime_data[i].append(c[:,j])
                self.prime_labels[i].append(Vn_mask[:,j])
            
        if self.have_display and debug:
            cv2.waitKey(10)
            
    def train_tracker(self):
        self.done_priming = True
        self.heatmap_models = []
#        smallest_len = np.min([len(d) for d in self.prime_data])
        smallest_len = 1024
        for i in range(len(self.Vs)):
            num_threads = mp.cpu_count()/2

            block_size = 1024
            SI = 0
            StS = 0
            for j in range(int(np.ceil(len(self.prime_data[i])/block_size))):
                inds = np.arange(j*block_size, np.minimum(len(self.prime_data[i]),(j+1)*block_size), dtype=np.int32)
                data = np.array([self.prime_data[i][ind] for ind in inds]).T  # (K+1) x num_samples
                labels = np.array([self.prime_labels[i][ind] for ind in inds]).T  # num_positions x num_samples
                SI = SI + ASC_utils.parallel_dot(data, labels.T, num_threads=num_threads)
                StS = StS + ASC_utils.parallel_dot(data, data.T, num_threads=num_threads)
            W = np.asfortranarray(np.linalg.solve((StS+np.eye(StS.shape[0])*0.000001), SI))

#            W = np.asfortranarray(np.random.randn(self.Vs[i].K+1, len(self.prime_labels[i][0]))/self.Vs[i].K/100)
            dW = None
            small_step_cnt = 0

            prev_mean_error = np.inf
            learning_rate = 0.05
            improved_last = 0
            for k in range(50):
                if k-improved_last>5:
                    break
                inds = np.random.permutation(len(self.prime_data[i]))[0:smallest_len]
                data = np.array([self.prime_data[i][ind] for ind in inds]).T  # (K+1) x num_samples
                labels = np.array([self.prime_labels[i][ind] for ind in inds]).T  # num_positions x num_samples

                inds2 = np.random.permutation(len(self.prime_data[i]))[0:smallest_len*16]
                data2 = np.array([self.prime_data[i][ind] for ind in inds2]).T  # (K+1) x num_samples
                labels2 = np.array([self.prime_labels[i][ind] for ind in inds2]).T  # num_positions x num_samples

                # normalize the data
#                data /= np.sqrt(np.mean(data**2, axis=1, keepdims=True)) + 0.1/smallest_len
#                labels /= np.mean(labels, axis=1, keepdims=True) + 0.1/smallest_len

                prev_W = W.copy(order='F')
                a2 = ASC_utils.parallel_dot(W.T, data2, num_threads=num_threads)
                a2 = np.maximum(0, a2)# + np.minimum(0, a)*0.01
                e2 = labels2 - a2

                mean_error = np.mean(e2**2)
                for j in range(100):
                    a = ASC_utils.parallel_dot(W.T, data, num_threads=num_threads)
                    a = np.maximum(0, a)# + np.minimum(0, a)*0.01
                    e = labels - a

                    d = e / data.shape[1] * ((a>=0) + 0.01)#*(a<0))
                    dW = ASC_utils.parallel_dot(data, d.T, num_threads=num_threads, AX_order='F', AX=dW)
                    prev_W = W.copy(order='F')
                    ASC_utils.add_scale(W, dW, learning_rate)

                    a2 = ASC_utils.parallel_dot(W.T, data2, num_threads=num_threads)
                    a2 = np.maximum(0, a2)# + np.minimum(0, a)*0.01
                    e2 = labels2 - a2

                    mean_error = np.mean(e2**2)
                    if prev_mean_error - mean_error < 0.00001: 
                        if j == 0:
                            learning_rate /= 2
                        else:
                            improved_last = k
#                        print k, j, data.shape[1], prev_mean_error, learning_rate
                        W = prev_W.copy(order='F')
                        break
                    else:
                        prev_mean_error = mean_error


            a = np.minimum(1,np.maximum(0, ASC_utils.parallel_dot(W.T, data, num_threads=num_threads)))
            l = ((labels>0)*1.0)
            
            V1_Xb = self.Vs[0].Xb
            V1_Yb = self.Vs[0].Yb
            a = a.reshape((V1_Xb, V1_Yb, -1)).transpose((2, 0, 1)).reshape((-1, V1_Yb))[:,:,None][:,:,[0, 0, 0]]
            l = l.reshape((V1_Xb, V1_Yb, -1)).transpose((2, 0, 1)).reshape((-1, V1_Yb))[:,:,None][:,:,[0, 0, 0]]
            
            self.heatmap_models.append(W)
        self.prime_data = None
        self.prime_labels = None

    def track(self, im, debug=False):
        if not self.done_priming:
            self.train_tracker()

        for i in range(self.feed_multiple):
            ss, cs = self.feed(im)
        self.tracking_steps += 1

        heatmaps = []
        
        V1_Xb = self.Vs[0].Xb
        V1_Yb = self.Vs[0].Yb
        num_Xb = self.im_shape[0]/V1_Xb
        num_Yb = self.im_shape[1]/V1_Yb
        for i, c in enumerate(cs):
            n = i + 1
            s = 2**i
            Vn_heatmap = np.maximum(0, ASC_utils.parallel_dot(self.heatmap_models[i].T, c))
            Vn_heatmap_im = Vn_heatmap.reshape((V1_Xb, V1_Yb, 1, num_Xb/s, num_Yb/s)).transpose((3, 0, 4, 1, 2)).reshape((num_Xb*V1_Xb/s, num_Yb*V1_Yb/s))
            Vn_heatmap_im = cv2.resize(Vn_heatmap_im, self.im_shape, interpolation=cv2.INTER_NEAREST)
            heatmaps.append(Vn_heatmap_im)

        if len(heatmaps) >= 4 and debug:
            heatmaps_im = np.minimum(1.0, np.maximum(0.0, 0.5*np.hstack((np.vstack((heatmaps[0], heatmaps[1])), np.vstack((heatmaps[2], heatmaps[3]))))[..., None]))
            if self.have_display:
                cv2.imshow("im_att", np.hstack((np.vstack((im/255.0, np.zeros(im.shape))), heatmaps_im[:,:,[0, 0, 0]])))
            cv2.imwrite("im_att%d_%04d.png" % (self.num_resets, self.tracking_steps), (np.hstack((np.vstack((im/255.0, np.zeros(im.shape))), heatmaps_im[:,:,[0, 0, 0]]))*255).astype(np.uint8))

        return heatmaps
        
    def reset_priming(self):
        self.priming_steps = 0
        self.tracking_steps = 0
        if 'num_resets' not in self.__dict__:
            self.num_resets = 0
        else:
            self.num_resets += 1
        self.done_priming = False
        self.attention_models = [] 
        
    def set_num_threads(self, num_threads=-1):
        if num_threads == -1:
            num_threads = mp.cpu_count()/2
        
        for Vn in self.Vs:
            Vn.num_threads = num_threads
        
    def save(self, filename=None):
        if filename is None:
            filename = self.name+".pkl"
        pickle.dump(self, open(filename, 'w'), protocol=-1)
