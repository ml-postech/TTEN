import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--dataset', default='fair_gowalla', type=str, help='name of dataset')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size for inference')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size for training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--patience', default=5, type=int)
    
    parser.add_argument('--loss_type', default='bpr', type=str, choices=['bpr', 'ssm'])
    parser.add_argument('--lambda1', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--ssm_temp', default=0.1, type=float, help='temperature in ssm loss')    
    parser.add_argument('--gpu_id', default='0', type=str, help='the gpu to use')

    #parser.add_argument('--pop_type', default='z_dist', type=str, choices=['uniform', 'pareto', 'z_dist'])
    
    parser.add_argument('--pred_ratio', default=-1, type=float, help='normalization strength')    
    parser.add_argument('--tten', action='store_true', help='activate tten, which normalizes embedding during inference')
    parser.add_argument('--wandb', action='store_true')

    return parser.parse_args()
args = parse_args()
