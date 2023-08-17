import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.sparse as sp
import scipy

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

def group_metrics(uids, predictions, topk, test_labels, item_grp, popularity):
    predictions = torch.LongTensor(predictions)
    prediction = predictions[:,:topk] # BX20

    num_classes = len(torch.unique(item_grp))
    top_K_items_grp = item_grp[prediction] # BX20, item groups that are recommended as top K
    item_num_lst_i, hit_lst_i, recall_lst_i, ndcg_lst_i = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    p_rsp = np.zeros(num_classes)
    p_reo = np.zeros(num_classes)

    # get recall and ndcg in each item popularity group
    for j in item_grp.unique().int():
        pred_partial = prediction.clone() # BX20
        pred_partial[top_K_items_grp != j] = -100
        _, hit, test_item_num, hr, recall, ndcg = metrics(uids, pred_partial, topk, test_labels, grp=j, item_grps=item_grp)
        item_num_lst_i[j] = test_item_num
        hit_lst_i[j] = hit
        recall_lst_i[j] = recall
        ndcg_lst_i[j] = ndcg

        df_ranking_group = (top_K_items_grp==j).sum()
        df_group = (item_grp==j).sum()
        p_rsp[j] = float(df_ranking_group / df_group)
        
        label_grp = [1 if j in item_grp[test_label_u] else 0 for test_label_u in test_labels]
        df_positive_ranking_group = hr # the number of users who interated with item group j and got corrected
        df_positive_group = sum(label_grp) # the number of users who interated with item group j
        p_reo[j] = float(df_positive_ranking_group / df_positive_group)

    # C_Ratio
    c_ratio = C_Ratio(top_K_items_grp)
    return item_num_lst_i, hit_lst_i, recall_lst_i, ndcg_lst_i, c_ratio, p_rsp, p_reo

def metrics(uids, predictions, topk, test_labels, grp=None, item_grps=None):
    user_num = 0
    test_item_num = 0
    all_hits = 0
    all_hitrate = 0
    all_recall = 0
    all_ndcg = 0
    for i, uid in enumerate(uids):
        prediction = list(predictions[i][:topk])  # user u의 prediction (item 번호)
        label = test_labels[uid] # user u 의 ground truth
        if grp is not None:
            label = np.array(label)
            test_item_num += len(label[item_grps[label].numpy()==grp.item()]) # test item 중 그룹 i의 수
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label) # num_hit_item / total_gt
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
            if hit:
                all_hitrate = all_hitrate + 1 # hit(1) or not(0)
            all_hits += hit
    if user_num==0:
        return user_num, test_item_num, 0, 0, 0, 0
    return user_num, all_hits, test_item_num, all_hitrate, all_recall, all_ndcg

def get_item_attr(rating_mat, num_classes=5):
    """
    Get item popularity
    """
    popularity = get_item_popularity(rating_mat)
    val, indices = torch.topk(popularity, len(popularity))
    pop_grp = torch.zeros(popularity.shape)
    num_grp = np.zeros(num_classes)
    num_grp_item = len(popularity) // num_classes
    for i in range(num_classes):
        if i ==num_classes-1:
            pop_grp[indices[num_grp_item*i:]] = num_classes - i - 1
            num_grp[i] = len(indices[num_grp_item*i:])
        pop_grp[indices[num_grp_item*i:num_grp_item*(i+1)]] = num_classes - i - 1
        num_grp[i] = num_grp_item
    return popularity, pop_grp, num_classes, num_grp

def get_item_popularity(dense_rating, normalize=False):
    dense_rating = scipy_sparse_mat_to_torch_sparse_tensor(dense_rating)
    popularity =  torch.sparse.sum(dense_rating, 0).to_dense()
    if normalize == 'standard':
        #popularity = F.normalize(popularity, dim=0)
        popularity = (popularity - torch.mean(popularity)) / torch.std(popularity)
    elif normalize == 'minmax':
        popularity = (popularity - torch.min(popularity)) / (torch.max(popularity) - torch.min(popularity))
    return popularity

def C_Ratio(top_K_items_grp,num_grp=5):
    c_ratio = []
    for i in range(num_grp):
        k = top_K_items_grp.shape[1]
        c_ratio.append(torch.sum((top_K_items_grp == i).sum(axis=1) / k))
    return np.array(c_ratio)

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        
    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                # normal sampling
                i_neg = np.random.randint(self.dokmat.shape[1])                
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
