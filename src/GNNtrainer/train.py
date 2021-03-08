import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs_trainer

import src.utils.math_utils as math_utils
import src.utils.io_utils as io_utils
import src.utils.featgen as featgen
import src.utils.graph_utils as graph_utils

import src.GNNtrainer.models as models


#############################
#
# Prepare Data
#
#############################
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs, normalize=False, max_num_nodes=max_nodes, features=args.feature_type
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim,
    )


#############################
#
# Training
#
#############################
def train(
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
    device="cuda",
):

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            else:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).to(device)
            h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
            label = Variable(data["label"].long()).to(device)
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).to(device)

            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)

            predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(
            dataset, model, args, name="Train", max_num_examples=100, device=device
        )
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(
                val_dataset, model, args, name="Validation", device=device
            )
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(
                test_dataset, model, args, name="Test", device=device
            )
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    # MODIFIED HERE (COMMENTED)
    # if test_dataset is not None:
    #    plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
    #    plt.legend(["train", "val", "test"])
    # else:
    #    plt.plot(best_val_epochs, best_val_accs, "bo")
    #    plt.legend(["train", "val"])
    # plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    # plt.close()
    # matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


#############################
#
# Evaluate Trained Model
#
#############################
def evaluate(
    dataset, model, args, name="Validation", max_num_examples=None, device="cuda"
):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).to(device)
        h0 = Variable(data["feats"].float()).to(device)
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(data["assign_feats"].float(), requires_grad=False).to(
            device
        )

        ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


#############################
#
# Run Experiments
#
#############################


def benchmark_task(args, writer=None, feat="node-label", device="cuda"):
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )
    print(max([G.graph["label"] for G in graphs]))

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
                # make it -1/1 instead of 0/1
                # feat = np.array(G.nodes[u]['label'])
                # G.nodes[u]['feat'] = feat * 2 - 1
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        max_num_nodes,
        input_dim,
        assign_input_dim,
    ) = prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        )
    else:
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        )
    model = model.to(device)

    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
        device=device,
    )
    evaluate(test_dataset, model, args, "Validation", device=device)


def main():
    prog_args = configs_trainer.trainer_arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
        device = "cuda"
    else:
        print("Using CPU")
        device = "cpu"

    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    if prog_args.bmname is not None:
        benchmark_task(prog_args, writer=writer, device=device)

    writer.close()


if __name__ == "__main__":
    main()
