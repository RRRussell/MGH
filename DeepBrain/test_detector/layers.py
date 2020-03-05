import numpy as np

import torch
from torch import nn
import math

def hard_mining(neg_output, neg_labels, num_hard):
    # print("in hard_mining")
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train=True):

        batch_size = labels.size(0)

        output = output.view(-1, 5)
        labels = labels.view(-1, 5)
        
        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)

        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]
        
        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        
        # neg_output = neg_output.float()
        # neg_labels = neg_labels.float()

        neg_prob = self.sigmoid(neg_output)
        
        # print("neg_prob:",neg_prob.shape)
        # print("neg_label:",neg_labels.shape)

        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])

            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]
            '''
            print("\npz:",pz.cpu().detach().numpy(),"\nlz:",lz.cpu().detach().numpy(),
                  "\nph:",ph.cpu().detach().numpy(),"\nlh:",lh.cpu().detach().numpy(),
                  "\npw:",pw.cpu().detach().numpy(),"\nlw:",lw.cpu().detach().numpy(),
                  "\npd:",pd.cpu().detach().numpy(),"\nld:",ld.cpu().detach().numpy())
            '''
            # print("pos_prob:",pos_prob)
            # print("pos_label:",pos_labels[:, 0])
            
            #prob = self.sigmoid(output[:, 0])
            #pz, ph, pw, pd = output[:, 1], output[:, 2], output[:, 3], output[:, 4]
            #lz, lh, lw, ld = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]

            aplha = 1  
            
            regress_losses = [
                aplha*self.regress_loss(pz, lz),
                aplha*self.regress_loss(ph, lh),
                aplha*self.regress_loss(pw, lw),
                aplha*self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            
            classify_loss = 0.5 * self.classify_loss(
            pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)
        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        
        classify_loss_data = classify_loss.item()
        
        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss
            
        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)
        # print("loss",loss)
        # print("class loss",classify_loss_data)
        # print("regress loss",regress_losses_data)
        # return [loss, classify_loss_data] + regress_losses_data
        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]

class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])
        self.nms_th = 0.1

    def __call__(self, output,thresh = 0.12, ismask=False, nzhw=[4,6,6]):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        def sigmoid_fun(x):
            return 1/(1+np.exp(-x))

        output[:, :, :, :, 0] = sigmoid_fun(output[:, :, :, :, 0])
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1)) #np.exp

        pbb = np.copy(output)
        crop_size = 24
        splits = []
	
        for z in range(nzhw[0]):
            for h in range(nzhw[1]):
                for w in range(nzhw[2]):
                    split = pbb[z*crop_size:(z+1)*crop_size,h*crop_size:(h+1)*crop_size,
                                w*crop_size:(w+1)*crop_size,:,:]
                    split = split.reshape(-1,5)
		    #print(split.shape)
                    split = nms(split,0.001)
                    if len(split)!=0:
                        splits.append(split)
	
        if len(splits)!=0:
            splits = np.concatenate(splits,axis=0)
	    print(splits.shape)
            print("-------------")
            #print("final nms")
            splits = nms(splits,0.001)
            return splits
        else:
            return np.array([])
        
def nms(output, nms_th):
    # print("before cut",len(output))
    # print(np.max(output[:,0]))
    # print(np.min(output[:,0]))
    output = output[output[:,0]>=0.5]
    # print(output)
    # print("after cut",len(output))
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    # if len(output)>200:
    #     output = output[:200] # get top 1000

    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    # print("after nms: ",len(bboxes))
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    #print("overlap",overlap)
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def acc(pbb, lbb, conf_th, nms_th, detect_th=0.1):
    # print("before cut:",pbb.shape)
    # pbb = pbb[pbb[:, 0] >= conf_th] 
    # print("before nms:",pbb.shape)
    # pbb = dzh_nms(pbb, nms_th)
    # print("after nms:",pbb.shape)
    
    tp = []
    fp = []
    fn = []
   
    l_flag = np.zeros((len(lbb),), np.int32)

    for p in pbb:
        flag = 0
        bestscore = 0

        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)  # iou between prediction and label

            if score>bestscore:
                bestscore = score
                besti = i

        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, lbb, pbb

def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp)+len(fp)<topk:
        conf_th = conf_th-0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th<-3:
            break
    tp = np.array(tp).reshape([len(tp),6])
    fp = np.array(fp).reshape([len(fp),6])
    fn = np.array(fn).reshape([len(fn),5])
    allp  = np.concatenate([tp,fp],0)
    sorting = np.argsort(allp[:,0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk,len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
#     print(fp_in_topk)
    fn_i =       np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i)>0:
        fn = np.concatenate([fn,tp[fn_i,:5]])
    else:
        fn = fn
    if len(tp_in_topk)>0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk)>0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp , fn

'''
def temp_nms(pbb, nms_th, step=1000):
    
    final_pbb = []
    
    for i in range(0,pbb.shape[0],step):
        temp_pbb = pbb[i:(i+step)]
        final_pbb.append(nms(temp_pbb,nms_th))
    
    return np.concatenate(final_pbb,axis=0)

def dzh_nms(pbb, nms_th, step=1000):
    
    pbb = pbb[pbb[:,0]>=0.8]
    
    if len(pbb) == 0:
        return pbb
    
    pbb = temp_nms(pbb,nms_th,step)

    return pbb

def dzh_iou(pbb, lbb):
    
    delta = 10
    r0 = pbb[3] / 2
    s0 = pbb[:3] - r0
    e0 = pbb[:3] + r0

    r1 = lbb[3] / 2
    s1 = lbb[:3] - r1
    e1 = lbb[:3] + r1
 
    flag = 0
    for i in range(3):
        if (pbb[i]<e1[i]+delta)&(pbb[i]>s1[i]-delta):
            #print(i," yes")
            flag += 1
            
    if flag == 3:
        return 1
    else:
        return 0

def dzh_acc(pbb, lbb, conf_th, nms_th, detect_th=0.1):
    
    if len(pbb)==0:
        return [],[],lbb,lbb,pbb
    print("before cut:",pbb.shape)
    pbb = pbb[pbb[:, 0] >= conf_th] 
    print("before nms:",pbb.shape)
    pbb = nms(pbb, nms_th)
    print("after nms:",pbb.shape)
    
    tp = []
    fp = []
    fn = []
   
    l_flag = np.zeros((len(lbb),), np.int32)
    
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = dzh_iou(p[1:5], l)  # iou between prediction and label
            if score == 1:
                print("in acc lbb",lbb)
                print("in acc pbb",p[1:5])
                print("in acc probability",p[0])
                print("i:",i,"score:",score)
            if score>bestscore:
                bestscore = score
                besti = i

        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(dzh_iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))
    return tp, fp, fn, lbb, pbb

'''