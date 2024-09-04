import torch
import torch.nn.functional as F 
import torch.nn as nn
import pandas as pd
import numpy as np
from Model.inter_temporal_embedding import interpolation

T=52
m=450
Sliding_window = 4
class SPM(torch.nn.Module):

    def __init__(self,P,C,d):
        super(SPM,self).__init__()
        self.P,self.C,self.d = P,C,d
        self.TwE = nn.Parameter(torch.randn(Sliding_window,d))
        self.PE = nn.Parameter(torch.randn(P,d))
        self.CE = nn.Parameter(torch.randn(C,d))
        self.mE = nn.Parameter(torch.randn(m,d))
        self.d = d
        self.extractor_MLP = nn.Sequential(
                        nn.Linear(m, m),
                        nn.ReLU(),  
                        nn.Linear(m, m))
        self.dec = nn.Sequential(
                        nn.Linear(m, m),
                        nn.ReLU(),  
                        nn.Linear(m, (4*P*C)))
    
    def forward(self,x,t):
        extractor = torch.einsum('ti,pi,ci,mi->tpcm',self.TwE,self.PE,self.CE,self.mE)
        extracted_feature = torch.einsum('tpcm,tpc->m',extractor,x[-Sliding_window:])
        extracted_feature = self.extractor_MLP(extracted_feature)
        res = self.dec(extracted_feature).reshape(4,self.P,self.C)
        return res

class STCM(torch.nn.Module):

    def __init__(self,P,C,d,external_path,group,group_index,device):
        super(STCM,self).__init__()
        self.P,self.C,self.d,self.group_index = P,C,d,group_index.to(device)
        self.group_embedding = nn.Embedding(num_embeddings=len(group), embedding_dim=d)  
        self.offset_embedding = nn.Parameter(torch.randn(P, 10))
        self.offset_linear_transform = nn.Linear(10, d)
        self.linear_Q = nn.Linear(d, 10)
        self.linear_K = nn.Linear(d, 10)
        self.pos_enc = nn.Sequential(
                        nn.Linear(P, P),
                        nn.ReLU(),  
                        nn.Linear(P, int(d/2)))
        self.city_enc = nn.Sequential(
                        nn.Linear(C, C),
                        nn.ReLU(),  
                        nn.Linear(C, int(d/2)))
        self.temporal_embedding = nn.Embedding(53,d)
        # self.external_mask = torch.tensor(torch.load(external_path)).to(device)
        # self.external_factors = nn.Embedding(378,5)
        # self.external_factors_projector = nn.Linear(5,d)
        self.CP_fusion = nn.Linear(int(d*d/4), int(d))
        self.pos_enc_V = nn.Sequential(
                        nn.Linear(P, int(d/2)),
                        nn.ReLU(),  
                        nn.Linear(int(d/2), P))
        self.city_enc_V = nn.Sequential(
                        nn.Linear(C, int(d/2)),
                        nn.ReLU(),  
                        nn.Linear(int(d/2), C))
        self.ffn = nn.Sequential(
                        nn.Linear(T, T),
                        nn.ReLU(),  
                        nn.Linear(T, T))
        self.predictor = nn.Linear(T, 4)
        self.SPM = SPM(P,C,10)

    
    def forward(self,x,t):
        ## city-GCN
        all_base_embedding = self.group_embedding(self.group_index)
        offset_embeddings = self.offset_linear_transform(self.offset_embedding)
        final_embeddings = all_base_embedding+offset_embeddings
        final_embeddings = F.normalize(final_embeddings , 2, -1)
        final_embeddings = F.relu(final_embeddings)
        embeddings_Q = self.linear_Q(final_embeddings)
        embeddings_K = self.linear_K(final_embeddings)
        GCN_Q = torch.mm(embeddings_Q,embeddings_Q.T)
        GCN_K = torch.mm(embeddings_Q,embeddings_Q.T)
        GCN_Q = F.softmax(GCN_Q,dim=0)
        GCN_K = F.softmax(GCN_K,dim=0)
        x_shape = x.shape
        x2 = x.reshape(-1,x.shape[-1])
        Q = torch.mm(x2,GCN_Q)
        K = torch.mm(x2,GCN_K)
        Q = Q.reshape(*x_shape)
        K = K.reshape(*x_shape)
        Q = self.city_enc(Q).transpose(1,2)
        K = self.city_enc(K).transpose(1,2)
        Q = self.pos_enc(Q)
        K = self.pos_enc(K)
        Q = Q.reshape(x_shape[0],-1)
        K = K.reshape(x_shape[0],-1)
        Q = self.CP_fusion(Q)
        K = self.CP_fusion(K)
        # print(Q.shape,K.shape)
        ## temporal embedding
        inter_embeddings = interpolation(self.temporal_embedding,next(self.parameters()).device)[t]
        # external_embeddings = self.external_factors_projector(self.external_factors(t)*self.external_mask[t].unsqueeze(1))
        # inter_embeddings = inter_embeddings + external_embeddings
        # print(inter_embeddings.shape)
        Q = Q + inter_embeddings
        K = K + inter_embeddings
        attention_score = F.softmax(torch.mm(Q,K.T)/self.d**0.5,dim=1)
        # print(attention_score.shape)
        V = self.city_enc_V(x).transpose(1,2) ##T,P,C -> T,C,P
        V = self.pos_enc_V(V).transpose(1,2) ## TCP TPC
        V = torch.einsum("ij,jkl->ikl",attention_score,V)+x
        V = self.predictor(self.ffn(V.transpose(0,2))).transpose(0,2)
        res = self.SPM(x,t)
        return V+res