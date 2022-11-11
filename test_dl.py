
import torch, os, argparse
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, default = "both")
    parser.add_argument("--file_marker", type = str, default = "forInfection")

    parser.add_argument("--hid_dim", type = int, default = 16)
    parser.add_argument("--nlayer", type = int, default = 1)
    parser.add_argument("--drop", type = float, default = 0)

    parser.add_argument("--device_id", type = int, default = 0)

    args = parser.parse_args()
    task = args.task

    torch.cuda.set_device(args.device_id)
    met_path = "data/nobe/met_{}.csv".format(args.file_marker)
    pro_path = "data/nobe/pro_{}.csv".format(args.file_marker)
    model_dir = "models/clfs/dl_{}_{}".format(args.file_marker, task)

    ids, labels, boards, data = load_data(met_path, pro_path, task)
    labels = np.expand_dims(labels, axis = -1)
    train_val_ids, train_val_ys, train_val_boards, train_val_xs = train_test_split(ids, labels, boards, data, "train")
    test_ids, test_ys, test_boards, test_xs = train_test_split(ids, labels, boards, data, "test")
    print("Train & Val: {}, Test: {}".format(train_val_xs.shape, test_xs.shape))
    test_xs, train_val_xs = tensor(test_xs), tensor(train_val_xs)
    test_ys, train_val_ys = tensor(test_ys), tensor(train_val_ys)

    model = MLP(train_xs.shape[1], 1, hid_dim = args.hid_dim, nlayer = args.nlayer, drop = args.drop).cuda()
    model = load_model(model, model_dir, "model.tar")
    model.eval()

    train_val_preds = model(train_val_xs)
    train_val_auc = cal_auc_tensor(train_val_preds, train_val_ys)
    test_preds = model(test_xs)
    test_auc = cal_auc_tensor(test_preds, test_ys)
    write_preds(train_val_preds, train_val_ys, train_val_ids, model_dir, "train_val.csv")
    write_preds(test_preds, test_ys, test_ids, model_dir, "test.csv")
    print(train_val_auc, test_auc)
