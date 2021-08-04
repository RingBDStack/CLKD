import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from metric import AverageNonzeroTripletsMetric
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics


global NMI
global AMI
global ARI
NMI =0
AMI = 0
ARI = 0


# Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]


def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes

def generateMasks(length, data_split, train_i, i, validation_percent=0.2, test_percent=0.2, save_path=None):
    # verify total number of nodes
    print(length,data_split[i])
    assert length == data_split[i]
    # If is in initial/maintenance epochs, generate train、test and validation indices
    if train_i == i:
        # randomly suffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices,save_path+'/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path+'/test_indices.pt')
        return train_indices, validation_indices, test_indices
    # If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices

# Utility function, finds the indices of the values' elements in tensor
def find(tensor, values):
    return torch.nonzero(tensor.cpu()[..., None] == values.cpu())


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, nf, layer_id):
        h = nf.layers[layer_id].data['h']
        # equation (1)
        z = self.fc(h)
        nf.layers[layer_id].data['z'] = z
        # print("test test test")
        A = nf.layer_parent_nid(layer_id)
        # print(A)
        # print(A.shape)
        A = A.unsqueeze(-1)
        B = nf.layer_parent_nid(layer_id + 1)
        # print(B)
        # print(B.shape)
        B = B.unsqueeze(0)
        _, indices = torch.topk((A == B).int(), 1, 0)
        # print(indices)
        # print(indices.shape)
        # indices = np.asarray(indices)
        indices = indices.cpu().data.numpy()

        nf.layers[layer_id + 1].data['z'] = z[indices]
        # print(nf.layers[layer_id+1].data['z'].shape)
        # equation (2)
        nf.apply_block(layer_id, self.edge_attention)
        # equation (3) & (4)
        nf.block_compute(layer_id,  # block_id – The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        nf.layers[layer_id].data.pop('z')
        nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z[indices] + nf.layers[layer_id + 1].data['h']  # residual connection
        return nf.layers[layer_id + 1].data['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, nf, layer_id):
        head_outs = [attn_head(nf, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, nf, trans = False, src=None,tgt=None):
        features = nf.layers[0].data['features']
        if trans:
            if args.mode == 4:
                features = nf.layers[0].data['tranfeatures']
                print("This is nonlinear trans!")
            if args.mode == 2:
                features = features.cpu()
                W = torch.from_numpy(torch.load('./datasets/LinearTranWeight/spacy_{}_{}/best_mapping.pth'.format(src, tgt)))
                if args.add_mapping:
                    print("This is linear trans!")
                    part1 = torch.index_select(features, 1, torch.tensor(range(0, args.word_embedding_dim)))
                    part1 = torch.matmul(part1, torch.FloatTensor(W))
                    part2 = torch.index_select(features, 1,
                                               torch.tensor(range(args.word_embedding_dim, features.size()[1])))
                    features = torch.cat((part1, part2), 1).cuda()

        nf.layers[0].data['h'] = features
        h = self.layer1(nf, 0)
        h = F.elu(h)
        # print(h.shape)
        nf.layers[1].data['h'] = h
        h = self.layer2(nf, 1)
        #h = F.normalize(h, p=2, dim=1)
        return h


# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        # print("testing, shape of logits: ", logits.size())
        return logits


class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        for batch_id, nf in enumerate(
                dgl.contrib.sampling.NeighborSampler(g,  # sample from the whole graph (contain unseen nodes)
                                                     num_all_samples,  # set batch size = the total number of nodes
                                                     1000,
                                                     # set the expand_factor (the number of neighbors sampled from the neighbor list of a vertex) to None: get error: non-int expand_factor not supported
                                                     neighbor_type='in',
                                                     shuffle=False,
                                                     num_workers=32,
                                                     num_hops=2)):
            nf.copy_from_parent()
            extract_features = model(nf)
            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features.device, dtype=torch.long)  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        assert (A == extract_nids).all()

    return (extract_nids, extract_features, extract_labels)


def mutual_extract_embeddings(g, model, peer,src,tgt,num_all_samples, labels):
    with torch.no_grad():
        model.eval()
        model.eval()
        for batch_id, nf in enumerate(
                dgl.contrib.sampling.NeighborSampler(g,  # sample from the whole graph (contain unseen nodes)
                                                     num_all_samples,  # set batch size = the total number of nodes
                                                     1000,
                                                     # set the expand_factor (the number of neighbors sampled from the neighbor list of a vertex) to None: get error: non-int expand_factor not supported
                                                     neighbor_type='in',
                                                     shuffle=False,
                                                     num_workers=32,
                                                     num_hops=2)):
            nf.copy_from_parent()
            extract_features1 = model(nf)
            if (args.mode==2 and args.add_mapping):
                print("** add linear tran peer feature **",src,tgt)
                extract_features2 = peer(nf,True,src=src,tgt=tgt) # representations of all nodes
            elif args.mode==4:
                print("** add nonlinear tran peer feature **",src,tgt)
                extract_features2 = peer(nf,True)  # representations of all nodes
            else:
                print("** add feature **")
                extract_features2 = peer(nf)

            extract_nids = nf.layer_parent_nid(-1).to(device=extract_features1.device, dtype=torch.long)  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
        assert batch_id == 0
        extract_nids = extract_nids.data.cpu().numpy()
        extract_features = torch.cat((extract_features1,extract_features2),1)
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        assert (A == extract_nids).all()

    return (extract_nids, extract_features, extract_labels)


def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('user ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(extract_features, extract_labels, indices,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    return value


# Inference(prediction)
def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(embedding_save_path,data_split,train_i,i,args.lang,args.Tealang)
    # record the time spent in seconds on direct prediction
    time_predict = []
    # Directly predict
    message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    start = time.time()
    # Infer the representations of all tweets
    extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
    test_nmi = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, False)
    seconds_spent = time.time() - start
    message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    time_predict.append(seconds_spent)
    np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))
    return model


def mutual_infer(embedding_save_path1,embedding_save_path2,data_split1,data_split2,train_i,i,loss_fn,metrics,model1,model2):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, test_indices1 = getdata(embedding_save_path1, data_split1,
                                                                                 train_i, i, args.lang1, args.lang2)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, test_indices2 = getdata(embedding_save_path2, data_split2,
                                                                                       train_i, i, args.lang2,args.lang1)

    #model1
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g1, model1, model2, args.lang1,
                                                                               args.lang2,
                                                                               len(labels1), labels1)
    test_value = evaluate(extract_features, extract_labels, test_indices1, -1, num_isolated_nodes2,
                          save_path_i1, False)


    #model2
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2)

    test_value = evaluate(extract_features, extract_labels, test_indices2, -1, num_isolated_nodes2,
                          save_path_i2, False)
    return model1,model2

def getdata(embedding_save_path,data_split,train_i,i, src=None, tgt=None):
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)
    # load data
    data = SocialDataset(args.data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]  # feature dimension

    g = dgl.DGLGraph(data.matrix,
                     readonly=True)
    num_isolated_nodes = graph_statistics(g, save_path_i)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly(readonly_state=True)

    mask_path = save_path_i + '/masks'
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    if train_i == i:
        train_indices, validation_indices, test_indices = generateMasks(len(labels), data_split,train_i,i,
                                                                              args.validation_percent,
                                                                              args.test_percent,
                                                                              mask_path)
    else:
        test_indices = generateMasks(len(labels), data_split,train_i,i, args.validation_percent,
                                                                              args.test_percent,
                                                                              mask_path)
    if args.use_cuda:
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        if train_i == i:
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()

    g.ndata['features'] = features
    if args.mode == 4:
        tranfeatures = np.load(
            args.data_path + '/' + str(i) + '/' + "-".join([src, tgt, 'features']) + '.npy')
        tranfeatures = torch.FloatTensor(tranfeatures)
        if args.use_cuda:
            tranfeatures = tranfeatures.cuda()
        g.ndata['tranfeatures'] = tranfeatures

    if train_i == i:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices
    else:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices


def mutual_train(embedding_save_path1,embedding_save_path2,data_split1,data_split2,train_i,i,loss_fn,metrics):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, train_indices1, validation_indices1, test_indices1 = getdata(embedding_save_path1,data_split1,train_i,i)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, train_indices2, validation_indices2, test_indices2 = getdata(embedding_save_path2,data_split2,train_i,i)

    model1 = GAT(in_feats1, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    model2 = GAT(in_feats2, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model1.cuda()
        model2.cuda()
    # Optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
    model1_data = {'opt': optimizer1, 'best_value':1e-9, 'best_epoch':0,
                   'model': model1, 'peer': model2, 'src': args.lang1, 'tgt': args.lang2,
                   'save_path_i': save_path_i1, 'num_iso_nodes': num_isolated_nodes1, 'g': g1, 'labels': labels1,
                   'train_indices': train_indices1, 'vali_indices': validation_indices1, 'test_indices': test_indices1,
                   'all_vali_nmi':[],'seconds_train_batches':[]}

    model2_data = {'opt': optimizer2, 'best_value':1e-9, 'best_epoch':0,
                   'model': model2, 'peer': model1, 'src': args.lang2, 'tgt': args.lang1,
                   'save_path_i': save_path_i2, 'num_iso_nodes': num_isolated_nodes2, 'g': g2, 'labels': labels2,
                   'train_indices': train_indices2, 'vali_indices': validation_indices2, 'test_indices': test_indices2,
                   'all_vali_nmi':[],'seconds_train_batches':[]}
    print("\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n")

    for epoch in range(args.n_epochs):
        for model_data in [model1_data,model2_data]:
            losses = []
            total_loss = 0
            for metric in metrics:
                metric.reset()

            for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(model_data['g'],
                                                                               args.batch_size,
                                                                               args.n_neighbors,
                                                                               neighbor_type='in',
                                                                               shuffle=True,
                                                                               num_workers=32,
                                                                               num_hops=2,
                                                                               seed_nodes=model_data['train_indices'])):
                start_batch = time.time()
                nf.copy_from_parent()
                model_data['model'].train()
                model_data['peer'].eval()
                pred =  model_data['model'](nf)
                if args.mode==2 and epoch>args.e:
                    if args.add_mapping:
                        peerpred = model_data['peer'](nf, trans=True, src=model_data['src'], tgt=model_data['tgt'])
                    else:
                        peerpred = model_data['peer'](nf)
                if args.mode ==4 and epoch>args.e:
                    peerpred = model_data['peer'](nf, trans=True)

                batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
                batch_labels = model_data['labels'][batch_nids]
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                if (args.mode == 2 or args.mode == 4) and epoch>=args.e:
                    l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                    lkd = l(pred,peerpred)
                    message = "    ".join(["add KD loss",str(loss),str(lkd)])
                    loss = loss + args.mt*lkd
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

                model_data['optimizer'].zero_grad()
                loss.backward()
                model_data['optimizer'].step()
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

            # Validation
            extract_nids, extract_features, extract_labels = extract_embeddings(model_data['g'], model_data['model'], len(model_data['labels']), model_data['labels'])
            validation_value = evaluate(extract_features, extract_labels,model_data['vali_indices'] , epoch, model_data['num_iso_nodes'],model_data['save_path_i'], True)
            #test_value = evaluate(extract_features, extract_labels, model_data['test_indices'], epoch, model_data['num_iso_nodes'],model_data['save_path_i'], False)

            model_data['all_vali_nmi'].append(validation_value)
            if validation_value > model_data['best_value']:
                model_data['best_value'] = validation_value
                model_data['best_epoch'] = epoch
                # Save model
                model_path = model_data['save_path_i'] + '/models'
                if (epoch == 0) and (not os.path.isdir(model_path)):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model_data['model'].state_dict(), p)
                print(model_data['src'],':','Best model was at epoch ', str(model_data['best_epoch']))

            for metric in metrics:
                metric.reset()

    with open(save_path_i1 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch '+ str(model1_data['best_epoch'])
        f.write(message)
    with open(save_path_i2 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch '+ str(model2_data['best_epoch'])
        f.write(message)
    # Save all validation nmi
    np.save(save_path_i1 + '/all_vali_nmi.npy', np.asarray(model1_data['all_vali_nmi']))
    np.save(save_path_i2 + '/all_vali_nmi.npy', np.asarray(model2_data['all_vali_nmi']))
    #save all seconds_train
    np.save(save_path_i1 + '/seconds_train_batches.npy', np.asarray(model1_data['seconds_train_batches']))
    np.save(save_path_i2 + '/seconds_train_batches.npy', np.asarray(model2_data['seconds_train_batches']))

    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2)
    test_value = evaluate(extract_features, extract_labels, test_indices2, epoch, num_isolated_nodes2,
                          save_path_i2, False)

    return model1,model2



# Train on initial/maintenance graphs
def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, data_split, train_i,i,args.lang,args.Tealang)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        if args.use_cuda:
            model.cuda()

    if args.mode == 2 or args.mode==4:
        Tmodel = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        if args.use_cuda:
            Tmodel.cuda()
        Tmodel_path = args.Tmodel_path + '/block_0' + '/models/best.pt'
        Tmodel.load_state_dict(torch.load(Tmodel_path))

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
        for batch_id, nf in enumerate(dgl.contrib.sampling.NeighborSampler(g,
                                                                           args.batch_size,
                                                                           args.n_neighbors,
                                                                           neighbor_type='in',
                                                                           shuffle=True,
                                                                           num_workers=32,
                                                                           num_hops=2,
                                                                           seed_nodes=train_indices)):
            start_batch = time.time()
            nf.copy_from_parent()
            model.train()
            # forward
            pred = model(nf)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            if args.mode == 2:
                if args.add_mapping:
                    Tpred = Tmodel(nf, trans=True, src=args.lang, tgt = args.Tealang)
                else:
                    Tpred = Tmodel(nf)
            if args.mode == 4:
                Tpred = Tmodel(nf, trans=True)

            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            if args.mode == 2 or args.mode == 4:
                # p = torch.matmul(pred,pred.T)
                # Tp = torch.matmul(Tpred,Tpred.T)
                #kl = F.kl_div(p.softmax(dim=-1).log(), Tp.softmax(dim=-1), reduction='sum')
                l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                #l = torch.nn.MSELoss(reduce=True, size_average=True)
                lkd = l(pred,Tpred)
                message = "    ".join(["add KD loss",str(loss),str(lkd)])
                print(message)
                loss = loss + args.t*lkd
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


        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        validation_nmi = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, True)
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
    # use_cuda = args.use_cuda and torch.cuda.is_available()
    use_cuda = True
    print("Using CUDA:", use_cuda)
    if use_cuda:
        torch.cuda.set_device(args.gpuid)

    #online situation with knowledge distillation
    if args.mutual:
        print("args.mutual is true")
        path1 = args.data_path1 + "/{}mode".format(args.mode)
        path2 = args.data_path2 + "/{}mode".format(args.mode)
        if not os.path.exists(path1):
            os.mkdir(path1)
            os.mkdir(path2)
        embedding_save_path1 = path1 + '/embeddings_' + \
                              strftime("%m%d%H%M%S",localtime()) + '-' + str(args.mode) + '-' + args.lang2
        embedding_save_path2 = path2 + '/embeddings_' + \
                               strftime("%m%d%H%M%S", localtime()) + '-' + str(args.mode) + '-' + args.lang1
        if not args.add_mapping and ( args.mode==1 or args.mode==2) or args.mode == 0:
            embedding_save_path1 = embedding_save_path1 + "-nomap"
            embedding_save_path2 = embedding_save_path2 + "-nomap"
        else:
            embedding_save_path1 = embedding_save_path1 + "-map"
            embedding_save_path2 = embedding_save_path2 + "-map"
        os.mkdir(embedding_save_path1)
        os.mkdir(embedding_save_path2)
        print("embedding_save_path1 and embedding_save_path2: ", embedding_save_path1,embedding_save_path2)
        with open(embedding_save_path1 + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(embedding_save_path2 + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        data_split1 = np.load(args.data_path1 + '/data_split.npy')
        data_split2 = np.load(args.data_path2 + '/data_split.npy')
        print("data_split1:",data_split1,'data_split2:',data_split2)


    else:
        if not args.is_incremental: #offline situation
            # make dirs and save args
            embedding_dir = args.data_path + '/{}mode'.format(args.mode)
            if not os.path.exists(embedding_dir):
                os.mkdir(embedding_dir)
            embedding_save_path = embedding_dir + '/embeddings_' + strftime("%m%d%H%M%S", localtime()) + '-' + str(args.mode) + '-' + args.Tealang
            if not args.add_mapping and ( args.mode==1 or args.mode==2) or args.mode == 0:
                embedding_save_path = embedding_save_path+"-nomap"
            else:
                embedding_save_path=embedding_save_path+"-map"
        else: #online situation without knowledge distillation
            embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S",localtime())
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

    #online situation with knowledge distillation
    if args.mutual:
        model1,model2 = mutual_train(embedding_save_path1,embedding_save_path2,data_split1,data_split2,train_i,0,loss_fn,metrics)
        if args.is_incremental:
            for i in range(1, min(data_split1.shape[0], data_split2.shape[0])):
                print("enter i ", str(i))        
                model1,model2 = mutual_infer(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn, metrics, model1, model2)   
                if i % args.window_size == 0:
                    train_i = i
                    train_indices1, train_indices2, indices_to_remove1, indices_to_remove2, model1, model2 = mutual_train(
                        embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i,i, loss_fn, metrics)

    else:
        model = initial_maintain(train_i, 0, data_split, metrics,embedding_save_path, loss_fn, None)
        if args.is_incremental:
            for i in range(1, data_split.shape[0]):
                print("incremental setting")
                print("enter i ",str(i))
                # Inference (prediction)
                model = infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model)
                # Maintain
                if i % args.window_size == 0:
                    train_i = i
                    train_indices, indices_to_remove, model = initial_maintain(train_i, i, data_split, metrics,
                                                                                embedding_save_path, loss_fn,
                                                                                model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--n_epochs', default=15, type=int,
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
    parser.add_argument('--metrics', type=str, default='nmi')
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    #offline or online situation
    parser.add_argument('--is_incremental', action='store_true', default=True,
                        help="static or incremental")
    #Teacher-Student structure or Mutual-Learning structure
    parser.add_argument('--mutual', action='store_true', default=False)
    #mode==2, add linear cross-lingual knowledge ditillation; mode == 4, add non-linear cross-lingual knowledge transformation
    #mode==0, no knowledge distillation
    #mode==1,directly input student attribute features to teacher model
    parser.add_argument('--mode',type=int,default=2)
    parser.add_argument('--add_mapping', action='store_true', default=False)
    parser.add_argument('--data_path', default='0730_ALL_French',
                        type=str, help="Path of features, labels and edges")
    #offline situation Teacher-Student structure
    parser.add_argument('--Tmodel_path', default='0730_ALL_English/0mode/embeddings_0730234530-0-English-nomap',
                        type=str,
                        help="File path that contains the pre-trained teacher model.")
    parser.add_argument('--lang', type=str, default="French")
    parser.add_argument('--Tealang', type=str, default='English')
    parser.add_argument('--t',type=int,default=1)

    #Mutual-Learning structure
    parser.add_argument('--data_path1', default='0517_ALL_English',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--data_path2', default='0517_ALL_French',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--lang1', type=str, default="English")
    parser.add_argument('--lang2', type=str, default="French")
    parser.add_argument('--e', type=int, default=0)
    parser.add_argument('--mt', type=float, default=0.1)

    args = parser.parse_args()
    
    args.mutual = False
    args.add_mapping = True
    main(args)






    # nmilist = []
    # amilist = []
    # arilist = []
    # for i in range(1,7):
    #     with open('0702-mix-CLWE-ALL/embeddings_0703134628/block_'+str(i)+'/evaluate.txt','r') as f:
    #         # print('%.4f'%float(f.readlines()[4].split(':')[1]),end='  ')
    #         all = f.readlines()[6].split()
    #         nmilist.append('%.4f'%float(all[1]))
    #         amilist.append('%.4f'%float(all[3]))
    #         arilist.append('%.4f'%float(all[5]))
    # print('nmi:',' '.join(nmilist))
    # print('ami:',' '.join(amilist))
    # print('ari:',' '.join(arilist))

