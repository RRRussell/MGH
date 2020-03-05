import torch
import numpy as np
'''
Since the original CT image is too large to be fully input into the network, we cut it into small cubes to input.
The split function is used to cut the original CT image.
The combine function is used to stitch, and the predicted bbox is mapped back to the original CT image.
'''
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        # We cut the original CT image into nz*nh*nw small cubes

        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        # For each cube, their side coincidence length is 2*margin, and their side length is side_len+2*margin
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if np.all(nzhw==None):
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1 * np.ones((
            int(nz * side_len),
            int(nh * side_len),
            int(nw * side_len),
            int(splits[0].shape[3]),
            int(splits[0].shape[4])), np.float32)

        idx = 0
        # The prediction results of each small cube are stitched together in the order of all CT images 
        # to generate the final global result
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][int(margin):int(margin + side_len), int(margin):int(margin + side_len), int(margin):int(margin + side_len)]
                    output[int(sz):int(ez), int(sh):int(eh), int(sw):int(ew)] = split
                    idx += 1

        return output                                                                             