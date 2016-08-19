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

import os
#os.putenv('OMP_WAIT_POLICY', 'active')
import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import multiprocessing as mp
import cPickle as pickle

import pyximport
pyximport.install(reload_support=True)
import ASC_utils

np.seterr(invalid='raise', divide='raise', over='raise')  # generate exceptions when invalid computations (Nan,Inf) are encountered

try:
    ASC_utils.set_flush_denormals()
except:
    # Unable to set the flags
    print "Setting flush denormals CPU flag not available."


class ImageSparser(object):
    def __init__(self, K, Xb, Yb, num_color=1, num_time=1, target_active_cnt=70, show_D=False, use_feedback=True, top_level=False,
                 update_D_steps=1000, num_threads=-1, name="model", positive_input=False, use_neighborhood=5, max_epoch_size=np.Inf):
        self.K = K
        self.Xb = Xb
        self.Yb = Yb
        self.positive_input = positive_input
        if positive_input:
            self.num_color = num_color * 2
        else:
            self.num_color = num_color
        self.num_time = num_time
        self.m = self.Xb * self.Yb * self.num_color * self.num_time
        self.input_history = None
        self.D = np.asfortranarray(np.random.randint(low=0, high=256, size=(self.m, self.K)).astype(np.float)/255.0-0.5)
        self.D /= np.sqrt(np.sum(self.D**2, axis=0))+0.001
        self.DtD = np.dot(self.D.T, self.D)

        self.target_active_cnt = target_active_cnt  # N from the paper
        self.A = 0  # matrix E from the paper
        self.B = 0
        self.dA = np.zeros((self.K+1, self.K+1))
        self.dB = np.zeros((self.m, self.K+1))
        self.A_cnt = 0
        self.dA_cnt = 0
        self.var_Complex = 0
        self.inv_std_Complex = None
        self.learning = True
        self.learn_Complex = True
        self.learn_Simple = True
        self.show_D = show_D
        self.step = 0
        self.update_D_steps = update_D_steps
        self.sparsify_time = np.zeros(3)
        self.last_update = 0
        self.check_point = True
        self.ave_cnts = target_active_cnt
        self.name = name
        self.ave_stop_l = 1.0
        self.cum_stop_l = 0.0
        self.stop_l_cnt = 0
        self.max_epoch_size = max_epoch_size

        self.num_threads_auto = (num_threads == -1)
        if self.num_threads_auto:
            num_threads = mp.cpu_count()/2
        assert num_threads > 0
        self.num_threads = num_threads

        self.inv_std_Simple = None
        self.inv_std_Simple_sp = None
       
        assert use_feedback is False or use_feedback == 1 or use_feedback == 4
        self.use_feedback = False if top_level else use_feedback
        assert use_neighborhood is False or use_neighborhood == 1 or use_neighborhood == 5
        if use_neighborhood == 4 and top_level:
            use_neighborhood = 1;
        self.use_neighborhood = use_neighborhood
        self.prev_context = None
        self.C = None  # delay creation of the C matrix until the size of the context input is known
        self.dC = 0
        self.dC_cnt = 0
        self.cum_Cerr = 0
        self.oldD = 0
        self.prev_Complex_alpha = None
        self._dC = None

    def __setstate__(self, dict):  # called when unpickled
        """ __setstate__() is called during unpickling.  Here can handle converting old formats to the current"""
        self.__dict__.update(dict)

        if 'num_C' in dict:
            self.num_color = self.num_C
        if 'num_T' in dict:
            self.num_time = self.num_T
        if 'S' in dict:
            self.C = self.S  # C matrix used to be called S
            self._dC = self._dS
            self.dC_cnt = self.dS_cnt
            self.var_Complex = self.var_C
        if 'use_feedback' not in dict:
            self.use_feedback = (self.C.shape[0]/self.C.shape[1] == 7)
        if 'inv_std_S' in dict:
            self.inv_std_Simple = self.inv_std_S
            self.inv_std_Simple_sp = self.inv_std_S_sp
            self.inv_std_Complex = self.inv_std_C
            self.prev_Complex_alpha = self.prev_C_alpha
        if 'learn_D' in dict:
            self.learn_Simple = self.learn_D
            self.learn_Complex = self.learn_S
        if 'positive_input' not in dict:
            self.positive_input = False
        if 'use_neighborhood' not in dict:
            self.use_neighborhood = False
            self.dave_context = 0
            self.ave_context = 0
            self.prev_context = None
        if 'num_threads_auto' not in dict or self.num_threads_auto:
            num_threads = mp.cpu_count()/2
            self.num_threads = num_threads

    def tile_img(self, img=None):
        """ reorders and reshapes image into (self.Xb*self.Yb*self.num_color*self.num_time, num_tiles) """
        if img is None:
            return self.tiled_img, self.tile_num_Xb, self.tile_num_Yb
        num_Xb = img.shape[0]/self.Xb
        num_Yb = img.shape[1]/self.Yb
        img = img.reshape((num_Xb, self.Xb, num_Yb, self.Yb, self.num_color*self.num_time)).transpose((1, 3, 4, 0, 2)).reshape((self.Xb*self.Yb*self.num_color*self.num_time, num_Xb*num_Yb))
        return img, num_Xb, num_Yb

    def sparsify(self, img, context=None):
        if self.step == 0 and not self.learning:
            return None, None

        assert img.dtype == np.float

        t = time.time()

        if self.positive_input:
            img = np.concatenate((np.maximum(0, img), np.maximum(0, -img)), axis=img.ndim-1)

        if self.num_time > 1:
            if self.input_history is None:
                self.input_history = np.zeros(tuple(list(img.shape) + [self.num_time]))
            self.input_history[..., 1:] = self.input_history[..., 0:-1] * (1.0 - 1.0/self.num_time)
            self.input_history[..., 0] = img
            img = self.input_history.reshape((img.shape[0], img.shape[1], -1))

        X, num_Xb, num_Yb = self.tile_img(img)
        self.tiled_img = X
        self.tile_num_Xb = num_Xb
        self.tile_num_Yb = num_Yb
        X = np.asfortranarray(X.astype(np.float))

        # Adaptive Sparse Coding
        Simple_alpha, stop_l = ASC_utils.ASC(self.D, self.DtD, X, self.target_active_cnt, num_threads=self.num_threads, 
                                        add_one=True, ave_stop_l=self.ave_stop_l, 
                                        inv_std=None if self.learn_Simple and self.learning else self.inv_std_Simple)

        if self.learning and self.learn_Simple:
            self.cum_stop_l += stop_l
            self.stop_l_cnt += 1
                
        self.ave_cnts = self.ave_cnts * 0.9 + 0.1*np.sum(Simple_alpha.data>0)/Simple_alpha.shape[1]

        self.sparsify_time[0] += time.time()-t
        
        if self.learning:
            self.step += 1
            if self.learn_Simple and self.step % self.update_D_steps == 0:
                # if no unit has responded in the update period, fake a response to a single random input
                zero_inds = np.where(self.dA.diagonal()==0)[0]
                if len(zero_inds) > 0:
                    m = Simple_alpha.max()
                    rinds = np.random.randint(Simple_alpha.shape[1], size=len(zero_inds))
                    Simple_alpha[zero_inds, rinds] = m

            t2=time.time()
            if self.learn_Simple:
                if Simple_alpha.indices.size == Simple_alpha.shape[1]*(self.target_active_cnt+1):
                    rows = Simple_alpha.indices.reshape((Simple_alpha.shape[1], self.target_active_cnt+1))
                    data = Simple_alpha.data.reshape((Simple_alpha.shape[1], self.target_active_cnt+1))
                    ASC_utils.parallel_dense_add_a_dot_at(self.dA, rows, data, num_threads=self.num_threads)
                    self.dB = ASC_utils.parallel_dense_add_dense_dot_at(self.dB, X, rows, data, num_threads=self.num_threads)
                else:
                    print self.step, "sparse matrix data is not rectangular, falling back to slower operation"
                    self.dA = self.dA + Simple_alpha.dot(Simple_alpha.T)  # sparse matrix does not have += operator
                    self.dB += Simple_alpha.dot(X.T).T
                self.dA_cnt += Simple_alpha.shape[1]

        # if we don't even have the rates of each unit yet, don't compute Complex layer
        if self.inv_std_Simple is None:
            if self.learning and self.step % self.update_D_steps == 0:
                self.update_D()
            self.sparsify_time[1] += time.time()-t2
            self.sparsify_time[2] += time.time()-t
            self.prev_Complex_alpha = np.zeros(Simple_alpha.shape)
            return None, None

        if self.learn_Simple and self.learning:
            Simple_alpha = Simple_alpha.multiply(self.inv_std_Simple_sp)  # normalization happens in ASC() after D is learned

        if self.use_neighborhood:
            if self.use_neighborhood == 5:
                sN = int(np.sqrt(self.prev_Complex_alpha.shape[1]))
                v_zeros = np.zeros((sN, 1, self.K+1))
                h_zeros = v_zeros.transpose((1, 0, 2))
                prev_Complex_alpha = self.prev_Complex_alpha.T.reshape((sN, sN, -1))
                neighborhood_list = [Simple_alpha.T.toarray(), self.prev_Complex_alpha.T,
                                     np.concatenate((prev_Complex_alpha[1:, :, :], h_zeros), axis=0).reshape((sN*sN, -1)),
                                     np.concatenate((h_zeros, prev_Complex_alpha[:-1, :, :]), axis=0).reshape((sN*sN, -1)),
                                     np.concatenate((prev_Complex_alpha[:, 1:, :], v_zeros), axis=1).reshape((sN*sN, -1)),
                                     np.concatenate((v_zeros, prev_Complex_alpha[:, :-1, :]), axis=1).reshape((sN*sN, -1))
                                    ]
                                    
                if context is not None:
                    neighborhood_list.append(context)

                context = np.hstack(neighborhood_list)
            else:
                if context is None:
                    context = np.hstack((Simple_alpha.T.toarray(), self.prev_Complex_alpha.T))
                else:
                    context = np.hstack((Simple_alpha.T.toarray(), self.prev_Complex_alpha.T, context))
        else:
            if context is None:
                context = Simple_alpha.T.toarray()
            else:
                context = np.hstack((Simple_alpha.T.toarray(), context))

        if self.C is None:
            self.C = np.asfortranarray(np.random.rand(context.shape[1], self.K+1)*0.0)

        if type(self.C) == int:
            # no complex weight matrix yet, need to wait for first update
            Complex_alpha = Simple_alpha.T.toarray()
        else:
            # use alpha.T.dot(self.C)).T because alpha is sparse and self.C is dense; dense.dot(sparse) doesn't work
            if scipy.sparse.issparse(context):
                Complex_alpha = np.array(context.dot(self.C)).T
            else:
                Complex_alpha = ASC_utils.parallel_dot(context, self.C, num_threads=self.num_threads).T
        Complex_alpha = np.maximum(0, Complex_alpha)

        if self.prev_context is None:
            self.prev_context = context

        if self.learning:
            if self.learn_Complex:
                if False and self.name[-1] == '1':
                    ind = np.argmax(np.abs(Simple_alpha.toarray()-self.prev_Complex_alpha))
                    row = ind / Simple_alpha.shape[1]
                    col = ind % Simple_alpha.shape[1]
                    print row, col, Simple_alpha[row,col], self.prev_Complex_alpha[row,col], np.max(np.abs(self.C[:,row]))
                self.cum_Cerr += np.sum(np.mean(np.array(Simple_alpha-self.prev_Complex_alpha)**2, axis=0)**0.5)
                delta = (np.array(Simple_alpha - self.prev_Complex_alpha)*((self.prev_Complex_alpha>0)+0.01)).T/Complex_alpha.shape[1]
                if scipy.sparse.issparse(self.prev_context):
                    dC = np.asfortranarray(self.prev_context.T.dot(delta))
                else:
                    self._dC = ASC_utils.parallel_dot(self.prev_context.T, delta, num_threads=self.num_threads, AX_order='F', AX=self._dC)
                    dC = self._dC
                learning_rate = 1.0/(10000.0+self.step/10.0)
                self.C *= 1.0-learning_rate*0.00001
                diag = np.arange(0,self.C.shape[1])
                self.C[diag,diag] *= 1.0-learning_rate*0.9
                max_abs = np.maximum(np.max(dC), -np.min(dC))
                ASC_utils.add_scale(self.C, dC, learning_rate/np.maximum(1.0, max_abs+0.0000001))
                self.dC_cnt += Complex_alpha.shape[1]
                self.var_Complex = self.var_Complex*(1.0-learning_rate) + learning_rate*np.mean(Complex_alpha**2, axis=1)
                self.inv_std_Complex = (1.0/(np.sqrt(self.var_Complex)+0.0000001)).reshape((self.K+1, 1))

            self.sparsify_time[1] += time.time()-t2
            
            self.prev_context = context

        self.prev_Complex_alpha = Complex_alpha

        Complex_alpha = self.inv_std_Complex * Complex_alpha if self.inv_std_Complex is not None else Simple_alpha.toarray()

        if self.learning and self.step % self.update_D_steps == 0:
            self.update_D()

        self.sparsify_time[2] += time.time()-t
        return Simple_alpha, Complex_alpha

    def update_D(self):
        if self.learn_Simple:
            self.A_cnt = self.A_cnt/2.0 + self.dA_cnt
            self.A = np.array(self.A/2.0 + self.dA)
            var_S = np.diag(self.A).reshape(-1)/self.A_cnt
            self.inv_std_Simple = (1.0/(np.sqrt(var_S)+0.0000001)).reshape((self.K+1, 1))  #np.max(var_S)/(100000.0**2)
            self.inv_std_Simple_sp = scipy.sparse.csc_matrix(self.inv_std_Simple)

            self.B = self.B/2.0 + self.dB

            dA = np.diag(self.A)[0:self.K]
            A = scipy.sparse.csc_matrix(self.A)
            self.oldD = self.D.copy()
            for i in range(self.K):
                self.D[:, i] += (self.B[:, i]-np.array(A[0:self.K, i].T.dot(self.D.T)).reshape(-1))/(dA[i]+0.0000001)
                self.D[:, i] /= np.sqrt(np.sum(self.D[:, i]**2))+0.0000001
            self.DtD = np.dot(self.D.T, self.D)

            self.ave_stop_l = self.cum_stop_l / self.stop_l_cnt
            self.stop_l_cnt = 0
            self.cum_stop_l = 0.0

        print time.ctime(), self.step, "updating D, average FPS", (self.step - self.last_update)/self.sparsify_time[2], self.ave_cnts, "mean L2 D", np.mean((self.D-self.oldD)**2)**0.5, self.cum_Cerr/(self.dC_cnt+0.0001)

        self.dC_cnt = 0
        self.cum_Cerr = 0
        self.dA_cnt = 0
        self.dA = np.zeros((self.K+1, self.K+1))
        self.dB = np.zeros((self.m, self.K+1))
        self.last_update = self.step
        self.update_D_steps += np.minimum(self.max_epoch_size, self.update_D_steps / 10)
        self.sparsify_time[:] = 0

        if self.show_D:
            self.show()
            self.show(self.D-self.oldD, self.name+"_Diff_")

        if self.check_point:
            pickle.dump(self, open(self.name+".pkl", "w"), protocol=-1)

    def imgs_D(self, D):
        if D is None:
            D = self.D
        W=int(np.sqrt(D.shape[1]))
        H=int(np.ceil(D.shape[1]/float(W)))

        dx = (self.Xb+1)
        dy = (self.Yb+1)
        imgs = []
        for j in range(self.num_time):
            img = np.zeros((W*dx, H*dy, 3)) + np.max(D)
            for i in range(D.shape[1]):
                x=i / H
                y=i % H
                if self.num_color == 6:
                    img6=D[:, i].reshape((self.Xb, self.Yb, self.num_color, self.num_time))[:, :, :, j]
                    img[x*dx:(x+1)*dx-1, y*dy:(y+1)*dy-1, :] = img6[:, :, 0:3] - img6[:, :, 3:6]
                else:
                    img[x*dx:(x+1)*dx-1, y*dy:(y+1)*dy-1, :] = D[:, i].reshape((self.Xb, self.Yb, self.num_color, self.num_time))[:, :, :, j]
            img /= np.maximum(np.abs(np.min(D)),np.max(D))
            img += 1.0
            img /= 2.0/255
            img = img.astype(np.uint8)
            imgs.append(img)

        return imgs

    def show(self, D=None, prefix=None):
        if prefix is None:
            prefix = self.name+"_"
        if D is None:
            D = self.D
        imgs = self.imgs_D(D)
        for i, img in enumerate(imgs):
            cv2.imwrite("%s%d_%d.png" % (prefix, self.step, i), cv2.resize(img, (img.shape[0]*3, img.shape[1]*3), interpolation=cv2.INTER_NEAREST))
