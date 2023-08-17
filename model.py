import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, lambda_1, dropout, batch_user, device, args):
        super(LightGCN,self).__init__()
        self.device = device
        self.args = args

        self.n_u, self.n_i = n_u, n_i
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)).cuda(torch.device(device)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)).cuda(torch.device(device)))
        #self.E_u_0 = nn.Parameter(nn.init.normal_(torch.empty(n_u,d), std=0.1).cuda(torch.device(device)))
        #self.E_i_0 = nn.Parameter(nn.init.normal_(torch.empty(n_i,d), std=0.1).cuda(torch.device(device)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm # (U)

        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_list = [None] * (l+1)

        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.Z_list = [None] * (l+1)

        self.lambda_1 = lambda_1
        self.dropout = dropout
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

    def forward(self, uids, pos, neg, csr=None, test=False):
        if test==True:  # testing phase
            if self.args.tten:
                preds = F.normalize(sum(self.E_u_list))[uids] @ F.normalize(sum(self.E_i_list)).T
                if self.args.pred_ratio+1:
                    i_divide = torch.pow(torch.norm(sum(self.E_i_list), dim=1), self.args.pred_ratio).unsqueeze(1)
                    preds = sum(self.E_u_list)[uids] @ (sum(self.E_i_list) / i_divide).T
            else:
                preds = sum(self.E_u_list)[uids] @ sum(self.E_i_list).T

            # mask train set
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask

            # mask test set during validation or valid set during test time
            mask = csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask

            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                # UXI * IXd
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                # IXU * UXd
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # aggregate
                # UXd
                self.E_u_list[layer] = self.Z_u_list[layer]
                # IXd
                self.E_i_list[layer] = self.Z_i_list[layer]

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]

            # bpr loss
            if self.args.loss_type == 'bpr':
                pos_scores = (u_emb * pos_emb).sum(-1)
                neg_scores = (u_emb * neg_emb).sum(-1)
                loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # ssm loss
            elif self.args.loss_type == 'ssm':
                u_emb = F.normalize(u_emb, dim = -1)
                pos_emb = F.normalize(pos_emb, dim = -1)
                neg_emb = F.normalize(neg_emb, dim = -1)

                ratings = torch.matmul(u_emb.unsqueeze(1), pos_emb.T).squeeze(dim=1) # BXB
                pos_ratings = torch.diag(ratings)
    
                numerator = torch.exp(pos_ratings / self.args.ssm_temp)
                denominator = torch.exp(ratings / self.args.ssm_temp).sum(dim=-1)
                loss_r = torch.mean(-torch.log(numerator/denominator))

            # reg loss
            loss_reg = 0
            #for param in self.parameters():
            #    loss_reg += param.norm(2).square()
            #loss_reg *= self.lambda_2
            if self.args.loss_type == 'ssm':
                loss_reg += 0.5 * (self.E_u_0[uids].norm(2).pow(2) + self.E_i_0[pos].norm(2).pow(2))/float(len(uids))
                loss_reg *= self.lambda_1            
            else:
                loss_reg += 0.5 * (self.E_u_list[0][uids].norm(2).pow(2) + self.E_i_list[0][pos].norm(2).pow(2) + self.E_i_0[neg].norm(2).pow(2))/float(len(uids))
                loss_reg *= self.lambda_1            

            # total loss
            loss = loss_r  + loss_reg 

            return loss, loss_r