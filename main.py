
from utils.utils import *
from utils.triplet_loss import *
from utils.metric import AverageNonzeroTripletsMetric
from model.GAT import GAT

import time
from time import localtime, strftime
import torch.optim as optim
import torch.nn as nn
import json

import argparse


# Inference(prediction)
def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(embedding_save_path, args.data_path,
                                                                                 data_split, train_i, i, args, args.lang,
                                                                                 args.Tealang)
    # record the time spent in seconds on direct prediction
    time_predict = []
    # Directly predict
    message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    start = time.time()
    # Infer the representations of all tweets
    extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels, args, labels.device)
    test_nmi = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, args.metrics, False)
    seconds_spent = time.time() - start
    message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    time_predict.append(seconds_spent)
    np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))
    return model



def mutual_infer(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn, metrics,
                 model1, model2, device):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, test_indices1 = getdata(embedding_save_path1,
                                                                                       args.data_path1, data_split1,
                                                                                       train_i, i, args, args.lang1,
                                                                                       args.lang2)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, test_indices2 = getdata(embedding_save_path2,
                                                                                       args.data_path2, data_split2,
                                                                                       train_i, i, args, args.lang2,
                                                                                       args.lang1)

    # model1
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g1, model1, model2, args.lang1,
                                                                               args.lang2,
                                                                               len(labels1), labels1, args, device)
    test_value = evaluate(extract_features, extract_labels, test_indices1, -1, num_isolated_nodes2,
                          save_path_i1, args.metrics, False)

    # model2
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2, args, device)

    test_value = evaluate(extract_features, extract_labels, test_indices2, -1, num_isolated_nodes2,
                          save_path_i2, args.metrics, False)
    return model1, model2


def mutual_train(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn, metrics,
                 device):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, train_indices1, validation_indices1, test_indices1 = getdata(
        embedding_save_path1, args.data_path1, data_split1, train_i, i, args, args.lang1, args.lang2)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, train_indices2, validation_indices2, test_indices2 = getdata(
        embedding_save_path2, args.data_path2, data_split2, train_i, i, args, args.lang2, args.lang1)

    model1 = GAT(in_feats1, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    model2 = GAT(in_feats2, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

    # Optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
    model1_data = {'opt': optimizer1, 'best_value': 1e-9, 'best_epoch': 0,
                   'model': model1, 'peer': model2, 'src': args.lang1, 'tgt': args.lang2,
                   'save_path_i': save_path_i1, 'num_iso_nodes': num_isolated_nodes1, 'g': g1, 'labels': labels1,
                   'train_indices': train_indices1, 'vali_indices': validation_indices1, 'test_indices': test_indices1,
                   'all_vali_nmi': [], 'seconds_train_batches': []}

    model2_data = {'opt': optimizer2, 'best_value': 1e-9, 'best_epoch': 0,
                   'model': model2, 'peer': model1, 'src': args.lang2, 'tgt': args.lang1,
                   'save_path_i': save_path_i2, 'num_iso_nodes': num_isolated_nodes2, 'g': g2, 'labels': labels2,
                   'train_indices': train_indices2, 'vali_indices': validation_indices2, 'test_indices': test_indices2,
                   'all_vali_nmi': [], 'seconds_train_batches': []}
    print("\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n")

    if args.use_cuda:
        model1.to(device)
        model2.to(device)

    for epoch in range(args.n_epochs):

        for model_data in [model1_data, model2_data]:
            losses = []
            total_loss = 0
            for metric in metrics:
                metric.reset()

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                model_data['g'], model_data['train_indices'], sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
            )
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time.time()
                model_data['model'].train()
                model_data['peer'].eval()

                blocks = [b.to(device) for b in blocks]
                # forward
                pred = model_data['model'](blocks,args)
                batch_nids = blocks[-1].dstdata[dgl.NID].to(device=device, dtype=torch.long)
                batch_labels = model_data['labels'].to(device)[batch_nids]
                peerpred = None

                if args.mode == 2 and epoch >= args.e:
                    if args.add_mapping:
                        peerpred = model_data['peer'](blocks, args, trans=True, src=model_data['src'], tgt=model_data['tgt'])
                    else:
                        peerpred = model_data['peer'](blocks, args)
                    peerpred = peerpred.to(device)

                if args.mode == 4 and epoch >= args.e:
                    peerpred = model_data['peer'](blocks, args, trans=True)
                    peerpred = peerpred.to(device)

                loss_outputs = loss_fn(pred, batch_labels, args.rd, peerpred)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                if (args.mode == 2 or args.mode == 4) and epoch >= args.e:
                    l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                    lkd = l(pred, peerpred.to(device))
                    message = "    ".join(["add KD loss", str(loss), str(lkd)])
                    loss = loss + args.mt * lkd
                    print(message)
                    with open(save_path_i1 + '/log.txt', 'a') as f:
                        f.write(message)

                losses.append(loss.item())
                total_loss += loss.item()
                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)
                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices1.shape[0],
                        100. * batch_id / ((train_indices1.shape[0] // args.batch_size) + 1), np.mean(losses))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i1 + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                model_data['opt'].zero_grad()
                loss.backward()
                model_data['opt'].step()
                batch_seconds_spent = time.time() - start_batch
                model_data['seconds_train_batches'].append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            message += '\n'
            print(message)
            with open(model_data['save_path_i'] + '/log.txt', 'a') as f:
                f.write(message)

            for b in blocks:
                del b
            del pred
            del input_nodes
            del output_nodes
            if peerpred != None:
                del peerpred

            # Validation
            extract_nids, extract_features, extract_labels = extract_embeddings(model_data['g'], model_data['model'],
                                                                                len(model_data['labels']),
                                                                                model_data['labels'],
                                                                                args,
                                                                                device)
            validation_value = evaluate(extract_features, extract_labels, model_data['vali_indices'], epoch,
                                        model_data['num_iso_nodes'], model_data['save_path_i'], args.metrics, True)

            model_data['all_vali_nmi'].append(validation_value)
            if validation_value > model_data['best_value']:
                model_data['best_value'] = validation_value
                model_data['best_epoch'] = epoch
                # Save model
                model_path = model_data['save_path_i'] + '/models'
                if not os.path.isdir(model_path):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model_data['model'].state_dict(), p)
                print(model_data['src'], ':', 'Best model was at epoch ', str(model_data['best_epoch']))

            for metric in metrics:
                metric.reset()

    with open(save_path_i1 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch ' + str(model1_data['best_epoch'])
        f.write(message)
    with open(save_path_i2 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch ' + str(model2_data['best_epoch'])
        f.write(message)
    # Save all validation nmi
    np.save(save_path_i1 + '/all_vali_nmi.npy', np.asarray(model1_data['all_vali_nmi']))
    np.save(save_path_i2 + '/all_vali_nmi.npy', np.asarray(model2_data['all_vali_nmi']))
    # save all seconds_train
    np.save(save_path_i1 + '/seconds_train_batches.npy', np.asarray(model1_data['seconds_train_batches']))
    np.save(save_path_i2 + '/seconds_train_batches.npy', np.asarray(model2_data['seconds_train_batches']))


    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g1, model1, model2, args.lang1,
                                                                               args.lang2,
                                                                               len(labels1), labels1, args, device)
    test_value = evaluate(extract_features, extract_labels, test_indices1, -1, num_isolated_nodes1,
                          save_path_i1, args.metrics, False)

    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2, args, device)
    test_value = evaluate(extract_features, extract_labels, test_indices2, -1, num_isolated_nodes2,
                          save_path_i2, args.metrics, False)

    return model1, model2


# Train on initial/maintenance graphs
def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, args.data_path, data_split, train_i, i, args, args.lang, args.Tealang)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        if args.use_cuda:
            model.cuda()

    if args.mode == 2 or args.mode == 4:
        Tmodel = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        Tmodel_path = args.Tmodel_path + '/block_' + str(train_i) + '/models/best.pt'
        Tmodel.load_state_dict(torch.load(Tmodel_path))
        if args.use_cuda:
            Tmodel.cuda()
        Tmodel.eval()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Start training
    message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        Tpred = None
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            start_batch = time.time()
            model.train()
            # forward
            blocks = [b.to(train_indices.device) for b in blocks]
            pred = model(blocks,args)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            if args.mode == 2:
                if args.add_mapping:
                    Tpred = Tmodel(blocks, args, trans=True, src=args.lang, tgt=args.Tealang)
                else:
                    Tpred = Tmodel(blocks,args)
            if args.mode == 4:
                Tpred = Tmodel(blocks, args, trans=True)

            batch_nids = blocks[-1].dstdata[dgl.NID].to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss_outputs = loss_fn(pred, batch_labels, args.rd, Tpred)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            if args.mode == 2 or args.mode == 4:
                # p = torch.matmul(pred,pred.T)
                # Tp = torch.matmul(Tpred,Tpred.T)
                # kl = F.kl_div(p.softmax(dim=-1).log(), Tp.softmax(dim=-1), reduction='sum')
                l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                # l = torch.nn.MSELoss(reduce=True, size_average=True)
                lkd = l(pred, Tpred)
                message = "    ".join(["add KD loss", str(loss), str(lkd)])
                print(message)
                loss = loss + args.mt * lkd
            losses.append(loss.item())
            total_loss += loss.item()

            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)

            if batch_id % args.log_interval == 0:
                message += 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_id * args.batch_size, train_indices.shape[0],
                    100. * batch_id / ((train_indices.shape[0] // args.batch_size) + 1), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch
            del pred
            if args.mode != 0:
                del Tpred
            for b in blocks:
                del b

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels, args,
                                                                            labels.device)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        validation_nmi = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args.metrics, True)
        all_vali_nmi.append(validation_nmi)

        # Early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model saved after epoch ', str(epoch))
        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    # Load the best model of the current block

    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")
    return model



def main(args):
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpuid)
        device = torch.device("cuda:{}".format(args.gpuid))
    else:
        device = torch.device('cpu')

    # online situation with knowledge distillation
    if args.mutual:
        print("args.mutual is true")
        path1 = args.data_path1 + "/{}mode".format(args.mode)
        path2 = args.data_path2 + "/{}mode".format(args.mode)
        if not os.path.exists(path1):
            os.mkdir(path1)
        if not os.path.exists(path2):
            os.mkdir(path2)
        embedding_save_path1 = path1 + '/embeddings_' + \
                               strftime("%m%d%H%M%S", localtime()) + '-' + str(args.mode) + '-' + args.lang2
        embedding_save_path2 = path2 + '/embeddings_' + \
                               strftime("%m%d%H%M%S", localtime()) + '-' + str(args.mode) + '-' + args.lang1
        if not args.add_mapping and (args.mode == 1 or args.mode == 2) or args.mode == 0:
            embedding_save_path1 = embedding_save_path1 + "-nomap"
            embedding_save_path2 = embedding_save_path2 + "-nomap"
        else:
            embedding_save_path1 = embedding_save_path1 + "-map"
            embedding_save_path2 = embedding_save_path2 + "-map"
        os.mkdir(embedding_save_path1)
        os.mkdir(embedding_save_path2)
        print("embedding_save_path1 and embedding_save_path2: ", embedding_save_path1, embedding_save_path2)
        with open(embedding_save_path1 + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(embedding_save_path2 + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        data_split1 = np.load(args.data_path1 + '/data_split.npy')
        data_split2 = np.load(args.data_path2 + '/data_split.npy')
        print("data_split1:", data_split1, 'data_split2:', data_split2)


    else:
        if not args.is_incremental:  # offline situation
            # make dirs and save args
            embedding_dir = args.data_path + '/{}mode'.format(args.mode)
            if not os.path.exists(embedding_dir):
                os.mkdir(embedding_dir)
            embedding_save_path = embedding_dir + '/embeddings_' + strftime("%m%d%H%M%S", localtime()) + '-' + str(
                args.mode) + '-' + args.Tealang
            if not args.add_mapping and (args.mode == 1 or args.mode == 2) or args.mode == 0:
                embedding_save_path = embedding_save_path + "-nomap"
            else:
                embedding_save_path = embedding_save_path + "-map"
        else:  # online situation without knowledge distillation
            embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
        os.mkdir(embedding_save_path)
        print("embedding_save_path: ", embedding_save_path)
        with open(embedding_save_path + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        data_split = np.load(args.data_path + '/data_split.npy')

    # Loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # Metrics
    metrics = [AverageNonzeroTripletsMetric()]  # Counts average number of nonzero triplets found in minibatches
    train_i = 0  # Initially, only use block 0 as training set (with explicit labels)

    # online situation with knowledge distillation
    if args.mutual:
        model1, model2 = mutual_train(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, 0,
                                      loss_fn, metrics, device)
        if args.is_incremental:
            for i in range(1, min(data_split1.shape[0], data_split2.shape[0])):
                print("enter i ", str(i))
                model1, model2 = mutual_infer(embedding_save_path1, embedding_save_path2, data_split1, data_split2,
                                              train_i, i, loss_fn, metrics, model1, model2, device)
                if i % args.window_size == 0:
                    train_i = i
                    model1, model2 = mutual_train(
                        embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn,
                        metrics, device)

    else:
        model = initial_maintain(train_i, 0, data_split, metrics, embedding_save_path, loss_fn, None)
        if args.is_incremental:
            for i in range(1, data_split.shape[0]):
                print("incremental setting")
                print("enter i ", str(i))
                # Inference (prediction)
                model = infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model)
                # Maintain
                if i % args.window_size == 0:
                    train_i = i
                    model = initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--n_epochs', default=1, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2000, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=32, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--validation_percent', default=0.1, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--test_percent', default=0.2, type=float,
                        help="Percentage of test nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--metrics', type=str, default='ami')
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--gpuid', type=int, default=2)
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    # offline or online situation
    parser.add_argument('--is_incremental', action='store_true', default=False,
                        help="static or incremental")
    # Teacher-Student structure or Mutual-Learning structure
    parser.add_argument('--mutual', action='store_true', default=False)
    # mode==2, add linear cross-lingual knowledge ditillation; mode == 4, add non-linear cross-lingual knowledge transformation
    # mode==0, no knowledge distillation
    # mode==1,directly input student attribute features to teacher model
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--add_mapping', action='store_true', default=False)
    parser.add_argument('--data_path', default='/data/renjiaqian/CLKD/datasets/318_ALL_French',
                        type=str, help="Path of features, labels and edges")
    # offline situation Teacher-Student structure
    parser.add_argument('--Tmodel_path',
                        default='/data/renjiaqian/CLKD/datasets/318_ALL_English/embeddings_0401225602/',#'803_hash_static-8-English/0mode/embeddings_0227165510-0-English-nomap',
                        type=str,
                        help="File path that contains the pre-trained teacher model.")
    parser.add_argument('--lang', type=str, default="French")
    parser.add_argument('--Tealang', type=str, default='English')
    parser.add_argument('--t', type=int, default=1)

    # Mutual-Learning structure
    parser.add_argument('--data_path1', default='/data/renjiaqian/CLKD/datasets/318_ALL_English',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--data_path2', default='/data/renjiaqian/CLKD/datasets/318_ALL_French',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--lang1', type=str, default="English")
    parser.add_argument('--lang2', type=str, default="French")
    parser.add_argument('--e', type=int, default=0)
    parser.add_argument('--mt', type=float, default=0.5)
    parser.add_argument('--rd', type=float, default=0.1)

    args = parser.parse_args()

    # args.mutual = True
    # args.mode = 2
    # args.use_cuda = True
    # args.is_incremental = True
    # args.add_mapping = False
    # args.gpuid = 2

    main(args)

