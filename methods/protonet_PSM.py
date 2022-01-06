# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))


class ProtoNet_PSM(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet_PSM, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.conv1 = ConvBlock(25, 5, 1)
        self.conv2 = nn.Conv2d(5, 25, 1, stride=1, padding=0)

    def get_masks(self, a):
        a = a.unsqueeze(0)
        a = (a.max(3))[0]
        a = a.transpose(1, 3) 
        a = F.relu(self.conv1(a))
        a = self.conv2(a) 
        a = a.transpose(1, 3)
        a = F.sigmoid(a/0.025)
        return a

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()


        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        z_proto   = z_support.view(self.n_way, self.n_support, -1 ).view(self.n_way*self.n_support, -1)

        support_embedding = z_proto.view(self.n_way*self.n_support,64,-1)

        query_embedding = z_query.view(self.n_way* self.n_query, 64,-1)

        support_embedding_norm = F.normalize(support_embedding, p=2, dim=1, eps=1e-12)
        query_embedding_norm = F.normalize(query_embedding, p=2, dim=1, eps=1e-12)
        
        
        support_embedding_norm = support_embedding_norm.transpose(1, 2).unsqueeze(1) 
        query_embedding_norm = query_embedding_norm.unsqueeze(0)


        semantic_correlations = torch.matmul(support_embedding_norm, query_embedding_norm) 

        query_guided_masks = self.get_masks(semantic_correlations)

        refined_support_embedding = support_embedding.unsqueeze(0).unsqueeze(2)* query_guided_masks.unsqueeze(3)

        refined_support_embedding = refined_support_embedding.squeeze(0).transpose(0,1).reshape(self.n_way* self.n_query, self.n_way,self.n_support, -1).mean(2)

        refined_support_embedding = F.normalize(refined_support_embedding, p=2, dim=-1, eps=1e-12)

        dists = euclidean_dist(z_query, refined_support_embedding)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(1)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
