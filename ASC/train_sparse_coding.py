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

import sys
import numpy as np
import cPickle as pickle
from ASC.sparse_manager import SparseManager
from ASC.frame_reader import FrameReader


def run(prefix, video_list, im_shape=(80,80), profiling=False, **kwargs):

    SM = SparseManager(prefix, im_shape=im_shape)
    SM.create_all(**kwargs)

    FR = FrameReader(video_list=video_list, step=1, randomize=False, shape=(SM.im_shape[0], SM.im_shape[1]))
    for i in range(3000000):
        SM.feed(FR.read())
        if profiling and i==15000:
            break
    SM.save()
    
    IS=SM.Vs[0]
    for i in range(0,IS.K,IS.K/20):
        inds = np.argsort(-IS.C[0:IS.K,i])
        IS.show((IS.D*np.abs(IS.C[0:IS.K,i]))[:,inds],prefix+"_C_"+str(i)+"_")
        IS.show(IS.D[:,inds[0:16]],prefix+"_C16_"+str(i)+"_")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        video_list = ('DARPA/TrainingData/Parrots.pkl', 'DARPA/TrainingData/AmazingLife.pkl', 'DARPA/TrainingData/HoneyBadgers.pkl')
        max_epoch_size=300000  # how many frames before the stimulus repeats
    else:
        video_list = [sys.argv[i] for i in range(1, len(sys.argv)-1)]
        max_epoch_size=int(sys.argv[-1])  # how many frames before the stimulus repeats

    profiling = False
    num_levels = 4  # how many levels to create
    K = 400  # number of units in Simple and Complex layers shared across all levels
    target_active_cnt = 70
    Xb = 10  # tile width in pixels
    Yb = 10  # tile height in pixels
    im_shape = (Xb*(2**(num_levels-1)), Yb*(2**(num_levels-1)))  # input image size
    positive_input = False  # when True generate a 6 channel retina instead of 3
    use_neighborhood = 5  # can be False for not lateral, 1 for only receiving input from previous Complex, or 5 to receive input from the 5 neighborhood
    use_feedback = 1  # allow top-down connections, can be False, 1 or 4
    num_time = 1  # set to higher than 1 to learn motion
    prefix = "ASC_tiles%dx%d_px%dx%d_V%d_K%d" % (im_shape[0]/Xb, im_shape[1]/Yb, Xb, Yb, num_levels, K)

    if use_neighborhood is not None:
        prefix = prefix + "_N%d" % (use_neighborhood)
    if use_feedback:
        prefix = prefix + "_feedback"

    if not profiling:
        run(prefix=prefix, im_shape=im_shape, video_list=video_list, K=K, num_time=num_time, target_active_cnt=target_active_cnt, Xb=Xb, Yb=Yb, positive_input=positive_input, 
            num_levels=num_levels, use_neighborhood=use_neighborhood, use_feedback=use_feedback, max_epoch_size=max_epoch_size)
    else:
        import line_profiler
        SM = SparseManager()
        SM.create_V1(K=K)  # need to create a dummy V1 so we can get a handle to the sparsify method

        lp = line_profiler.LineProfiler(SM.Vs[0].sparsify, SM.feed)
        lp.run('run(profiling=True, prefix="profiling", im_shape=im_shape, video_list=video_list, K=K, num_time=num_time, target_active_cnt=target_active_cnt, Xb=Xb, Yb=Yb, positive_input=positive_input, num_levels=num_levels, use_neighborhood=use_neighborhood, use_feedback=use_feedback, max_epoch_size=max_epoch_size)')
        lp.print_stats()
