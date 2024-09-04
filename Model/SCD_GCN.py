import argparse
import os
import torch
import random
import numpy as np
import json

import torch
import numpy as np

class aggregating:
    def __init__(self, input_org):
        ### clustering C-P
        city = torch.mean(input_org, axis=1)
        position = torch.mean(input_org, axis=0)
        city_cor = np.corrcoef(np.array(city).T)
        pos_cor = np.corrcoef(np.array(position).T)
        index = np.where(city_cor > 0.98)

        class UnionFindSet:
            def __init__(self, n):
                self.parent = [i for i in range(n)]
                self.rank = [0] * n

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)

                if root_x != root_y:
                    if self.rank[root_x] < self.rank[root_y]:
                        self.parent[root_x] = root_y
                    elif self.rank[root_x] > self.rank[root_y]:
                        self.parent[root_y] = root_x
                    else:
                        self.parent[root_y] = root_x
                        self.rank[root_x] += 1

        ufs = UnionFindSet(city.shape[1])
        for i in range(len(index[0])):
            ufs.union(index[0][i], index[1][i])
        group = np.unique(ufs.parent)
        group_dic = {}
        for i in range(len(group)):
            group_dic[group[i]] = i
        group_index = torch.tensor([group_dic[i] for i in ufs.parent])

        self.group = group
        self.group_index = group_index

    def get_results(self):
        return self.group, self.group_index


