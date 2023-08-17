import numpy as np
import torch
import pickle
from model import LightGCN
from utils import *
import pandas as pd
from parser import args
from tqdm import tqdm
import torch.utils.data as data
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
import os

np.set_printoptions(suppress=True)
os.makedirs('saved_checkpoint', exist_ok=True)

if args.gpu_id != '-1':
    device = 'cuda:' + args.gpu_id
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.wandb:
    import wandb

    wandb.init(project="TTEN", entity="YOUR_ENTITIY")
    wandb.run.name = '{}_{}_{}_{}'.format(args.dataset, args.gnn_layer, args.temp, args.tten)
    wandb.config.update(args)

# hyperparameters
d = args.d
l = args.gnn_layer
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
dropout = args.dropout
lr = args.lr

best_epoch = 0
best_valrecall = 0
checkpoint_buffer = []

############# load data
# load train set
path = 'data/' + args.dataset + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
n_u, n_i=train.shape[0], train.shape[1]
train_csr = (train!=0).astype(np.float32)

# load test set
f = open(path+'tstMat.pkl','rb')
test_raw = pickle.load(f)
print('Data loaded.')

# load valid set
if os.path.exists(path+'valMat.pkl'):
    f = open(path+'valMat.pkl', 'rb')
    valid = pickle.load(f)
    test = test_raw
else:
    # if valid not exists, split from test set.
    print("Split valid and test")
    # split train and valid
    num_interactions = len(test_raw.data)
    test_indices = [i for i in range(num_interactions)]
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
    valid = coo_matrix((test_raw.data[val_indices], (test_raw.row[val_indices], test_raw.col[val_indices])), shape=(n_u, n_i))
    test = coo_matrix((test_raw.data[test_indices], (test_raw.row[test_indices], test_raw.col[test_indices])), shape=(n_u, n_i))

valid_csr = (valid!=0).astype(np.float32)
test_csr = (test!=0).astype(np.float32)

# process valid set
val_labels = [[] for i in range(valid.shape[0])]
for i in range(len(valid.data)):
    row = valid.row[i]
    col = valid.col[i]
    val_labels[row].append(col)
print('Valid data processed.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

print('user_num:',train.shape[0],'item_num:',train.shape[1])

############## data preprocessing
# get item popularity
popularity, item_grp, i_num_classes, item_grp_num = get_item_attr(train)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))

print('Adj matrix normalized.')

model = LightGCN(n_u, n_i, d, train_csr, adj_norm, l, lambda_1, dropout, batch_user, device, args)
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)

# construct data loader
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

for epoch in range(epoch_no):
    epoch_loss = 0
    epoch_loss_r = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # get loss
        optimizer.zero_grad()
        loss, loss_r = model(uids, pos, neg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r)
    if args.wandb:
        wandb.log({
        'Epoch':epoch,
        'Loss':epoch_loss,
        'Loss_r':epoch_loss_r
        })

    if epoch % 3 == 0:  # validate every 3 epochs
        val_uids = np.array([i for i in range(n_u)])
        batch_no = int(np.ceil(len(val_uids)/batch_user))

        all_user_num = 0
        all_hitrate_20 = 0
        all_recall_20 = 0
        all_ndcg_20 = 0
        all_hitrate_40 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        all_c_ratio = np.zeros(5)

        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(val_uids))

            val_uids_input = torch.LongTensor(val_uids[start:end]).cuda(torch.device(device))
            predictions = model(val_uids_input, None, None, csr=test_csr, test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            user_num, _, _, hitrate_20, recall_20, ndcg_20 = metrics(val_uids[start:end],predictions,20,val_labels)
            #top@40
            user_num, _, _, hitrate_40, recall_40, ndcg_40 = metrics(val_uids[start:end],predictions,40,val_labels)
                        
            # C_ratio
            top_K_items_grp = item_grp[predictions[:,:20]] # BX20, item groups that are recommended as top K
            c_ratio = C_Ratio(top_K_items_grp)

            all_user_num += user_num
            all_hitrate_20+=hitrate_20
            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_hitrate_40+=hitrate_40
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            all_c_ratio+=c_ratio

        print('-------------------------------------------')
        print('Test of epoch',epoch, ':','HR@20:',all_hitrate_20/all_user_num,'Recall@20:',all_recall_20/all_user_num,'Ndcg@20:',all_ndcg_20/all_user_num,'HR@40:',all_hitrate_40/all_user_num,'Recall@40:',all_recall_40/all_user_num,'Ndcg@40:',all_ndcg_40/all_user_num)
        print('Test of epoch',epoch, ':','C_ratio:',all_c_ratio/n_u) #'ConfRate:',all_conf_rate/batch_no)
        if args.wandb:
            wandb.log({
            'Epoch':epoch,
            'Valid_HR': all_hitrate_20/all_user_num,
            'Valid_Recall':all_recall_20/all_user_num,
            'Valid_Ndcg':all_ndcg_20/all_user_num,
            'Valid_C_ratio':all_c_ratio[-1].item()/n_u
            })
        
        if best_valrecall < all_recall_20/all_user_num:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            best_epoch = epoch
            best_valrecall = all_recall_20/all_user_num
            filename = os.path.join('saved_checkpoint', '{}_{}_{}_{}_epoch={}.checkpoint.pth.tar'.format(args.dataset, args.loss_type, args.gnn_layer, args.tten, best_epoch))
            torch.save(state, filename)
            checkpoint_buffer.append(filename)
            if len(checkpoint_buffer)>10:
                os.remove(checkpoint_buffer[0])
                del(checkpoint_buffer[0])

        if args.patience and epoch > 50:
            early_stopping(all_recall_20/all_user_num, model)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break    

filename = os.path.join('saved_checkpoint', '{}_{}_{}_{}_epoch={}.checkpoint.pth.tar'.format(args.dataset, args.loss_type, args.gnn_layer, args.tten, best_epoch))
print("Loading from checkpoint {}".format(filename))

checkpoint = torch.load(filename, map_location = str(device))

model.load_state_dict(checkpoint['state_dict'])
print("=> Successfully restored checkpoint (trained for {} epochs)"
        .format(checkpoint['epoch']))

if args.wandb:
    wandb.log({
        'Best epoch':best_epoch,
        'Best_valrecall': best_valrecall,
    })         

# run 1 epoch to update E_u_list
train_loader.dataset.neg_sampling()
for i, batch in enumerate(tqdm(train_loader)):
    uids, pos, neg = batch
    uids = uids.long().cuda(torch.device(device))
    pos = pos.long().cuda(torch.device(device))
    neg = neg.long().cuda(torch.device(device))
    iids = torch.concat([pos, neg], dim=0)
    # get loss
    loss, loss_r = model(uids, pos, neg)

# test
test_uids = np.array([i for i in range(n_u)])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_user_num = 0
all_hitrate_20 = 0
all_recall_20 = 0
all_ndcg_20 = 0
all_hitrate_40 = 0
all_recall_40 = 0
all_ndcg_40 = 0
all_c_ratio = np.zeros(5)
all_conf_rate = 0
all_rsp, all_reo = np.zeros(5), np.zeros(5)
recall_lst_i, ndcg_lst_i, hit_lst_i, test_item_num_lst_i = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)

for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input, None, None, csr=valid_csr, test=True)
    predictions = np.array(predictions.cpu())

    #top@20
    user_num, _, _, hitrate_20, recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    user_num, _, _, hitrate_40, recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    test_item_num_i, hit_i, recall_i, ndcg_i, c_ratio, rsp, reo = group_metrics(test_uids[start:end],predictions,20,test_labels, item_grp, popularity)
    
    all_user_num += user_num
    all_hitrate_20+=hitrate_20
    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_hitrate_40+=hitrate_40
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40

    test_item_num_lst_i += test_item_num_i
    hit_lst_i += hit_i
    recall_lst_i += recall_i
    ndcg_lst_i += ndcg_i
    all_c_ratio += c_ratio
    all_rsp += rsp
    all_reo += reo
    
all_rsp /= n_u

rsp = float(np.std(all_rsp) / np.mean(all_rsp))
reo = float(np.std(all_reo) / np.mean(all_reo))


print('-------------------------------------------')
print('Final test:','HR@20:',all_hitrate_20/all_user_num, 'Recall@20:',all_recall_20/all_user_num,'Ndcg@20:',all_ndcg_20/all_user_num,'HR@40:',all_hitrate_40/all_user_num,'Recall@40:',all_recall_40/all_user_num,'Ndcg@40:',all_ndcg_40/all_user_num)
print('Final test:','C_ratio:',all_c_ratio/n_u, 'rsp:', np.round(rsp, 4), 'reo:', np.round(reo, 4))

print('Item popularity:', np.round(recall_lst_i/all_user_num, 4), np.round(ndcg_lst_i/item_grp_num, 4))
print('Item_popularity:', np.round(hit_lst_i/test_item_num_lst_i, 4))

if args.wandb:
    wandb.log({
        'Epoch':epoch,
        'Test_HR': all_hitrate_20/all_user_num,
        'Test_Recall':all_recall_20/all_user_num,
        'Test_Ndcg':all_ndcg_20/all_user_num,
        'Test_C_ratio':all_c_ratio[-1].item()/n_u, 
        'Test_ConfRate':all_conf_rate/all_user_num,
        'Test_rsp': rsp,
        'Test_reo': reo,
        'Test_recall_cv': float(np.std(recall_lst_i) / np.mean(recall_lst_i))
        })
