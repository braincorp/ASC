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
import random


class FrameReader(object):
    def __init__(self, video_list, step=1, cut_front=0000, cut_back=0000, randomize=False, shape=(100, 100)):
        self.shape = shape
        self.loaded = False
        self.randomize = randomize
        self.frames = []
        self._frame_gen = self.load_frames(video_list, step, cut_front, cut_back)
        if randomize:
            for img in self._frame_gen:
                self.frames.append(img)
            random.shuffle(self.frames)
            self.loaded = True
            
        self.cnt = 0
    
    def read(self):
        if not self.loaded:
            try:
                img = next(self._frame_gen)
                self.frames.append(img)
                self.cnt += 1
                return img
            except StopIteration:
                self.loaded = True
        img = self.frames[self.cnt % len(self.frames)]
            
        self.cnt += 1
        return img

    def load_frames(self, video_list, step=1, cut_front=5000, cut_back=5000):
        frames = []
        for file in video_list:
            try:
                from tracker_tools.tsartifacts import TSArtifacts
                ts = TSArtifacts()
                full_file = ts.get(file)
                if full_file == None:
                    full_file = file
            except:
                print "Could not download training video", file, "using local copy instead."
                full_file = file
            if full_file[-4:] == '.pkl':
                from tracker_base.labeled_movie import FrameCollection
                fc = FrameCollection()
                
                fc.load_from_file(full_file)
                for i in xrange(cut_front/step, len(fc)/step-cut_back/step):
                    img = fc.Frame(i*step).get_image()
                    img = cv2.resize(img, dsize=(self.shape), interpolation=cv2.INTER_AREA)
                    yield img
            else:
                if cut_back > 0:
                    print "Warning, ignoring cut_back while reading non-pkl videos."
                    
                assert os.path.exists(full_file), "Error, file not found: "+full_file
                # use opencv to read everything else
                cap = cv2.VideoCapture(full_file)
                while(cap.isOpened()):
                    for j in range(cut_front):
                        ret, img = cap.read()
                    for j in range(step):
                        ret, img = cap.read()
                    if not ret:
                        break
                    img = cv2.resize(img, dsize=(self.shape), interpolation=cv2.INTER_AREA)
                    yield img
