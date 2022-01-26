
import argparse
import sys
import random
import tensorflow
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model
import numpy as np
import pandas as pd
from stellargraph import StellarDiGraph
from utlis import KGTripleGenerator
from model import DistMult


def get_args():
    ## Argument and global variables
    parser = argparse.ArgumentParser('WSDM competition')
    parser.add_argument('--dataset', type=str, choices=["A", "B"], default='A', help='Dataset name')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument("--emb_dim", type=int, default=200, help="embedding size")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--negative_samples", type=int, default=5, help="number of negative samples")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


id_gap = 0

def construct_grapg(args):
    global id_gap

    edge_csv = pd.read_csv(f'train_csvs/edges_train_{args.dataset}.csv', header=None)
    input_initial = pd.read_csv(f'intermediate/input_{args.dataset}_initial2.csv', header=None)
    input_inter = pd.read_csv(f'intermediate/input_{args.dataset}.csv', header=None)

    if args.dataset == "A":
        input_final = pd.read_csv('finals/input_A_final.csv', header=None)
    else:
        input_final = pd.read_csv('finals/input_B.csv', header=None)


    edge_csv = edge_csv[[0, 1, 2]]
    input_initial = input_initial[[0, 1, 2]]
    input_inter = input_inter[[0, 1, 2]]
    input_final = input_final[[0, 1, 2]]

    edge_csv.drop_duplicates(inplace=True)


    if args.dataset == "B":
        id_gap = max(edge_csv[0].values) + 1
        edge_csv[1] = edge_csv[1] + id_gap

    square_edges = pd.DataFrame(
        {"source": edge_csv[0].tolist(), "target": edge_csv[1].tolist()}
    )
    square_edges_types = square_edges.assign(
        edge_type=edge_csv[2].tolist()
    )

    if args.dataset == "B":
        User = pd.DataFrame(index=list(set(edge_csv[0].tolist())
                                       | set(input_initial[0].tolist())
                                       | set(input_inter[0].tolist())
                                       ))
        input_initial[1] = input_initial[1] + id_gap
        input_inter[1] = input_inter[1] + id_gap
        Item = pd.DataFrame(index=list(set(edge_csv[1].tolist())
                                       | set(input_initial[1].tolist())
                                       | set(input_inter[1].tolist())
                                       ))
        graph = StellarDiGraph(
            {"User": User, "Item": Item},
            square_edges_types,
            edge_type_column="edge_type",
        )
    else:
        graph = StellarDiGraph(
            edges=square_edges_types,
            edge_type_column="edge_type",
        )

    return graph

def train(graph,args):

    graph_gen = KGTripleGenerator(
        graph, batch_size=args.batch_size
    )

    distmult = DistMult(
        graph_gen,
        embedding_dimension=args.emb_dim,
        embeddings_regularizer=regularizers.l2(1e-7),
    )

    model_inp, model_out = distmult.in_out_tensors()

    model = Model(inputs=model_inp, outputs=model_out)
    model.compile(
        # optimizer=optimizers.Adam(learning_rate=args.lr),
        optimizer=optimizers.Adam(learning_rate=tensorflow.keras.experimental.CosineDecayRestarts(args.lr, 3)),
        loss=losses.BinaryCrossentropy(from_logits=True),
        # metrics=[metrics.BinaryAccuracy(threshold=0.0)],
        metrics=[metrics.AUC(from_logits=True)],
    )

    # get train/valid/test_inter/test_final
    train_gen, valid_gen, test1_gen, test2_gen = get_splits(args, graph_gen)

    # training
    graph_es = callbacks.EarlyStopping(monitor="val_auc", patience=args.epochs, mode='max',
                                       restore_best_weights=False)
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=[graph_es],
        verbose=1,
    )

    prediction1 = model.predict(test1_gen)
    np.savetxt(f"output_{args.dataset}_inter.csv", prediction1, delimiter=',')

    prediction2 = model.predict(test2_gen)
    np.savetxt(f"output_{args.dataset}.csv", prediction2, delimiter=',')

    sub = pd.read_csv(f"output_{args.dataset}_inter.csv", names=['exist_prob'])
    print('inter sub shape:', sub.shape)

    sub = pd.read_csv(f"output_{args.dataset}.csv", names=['exist_prob'])
    print('final sub shape:', sub.shape)

def get_splits(args, graph_gen):
    global id_gap

    if args.dataset == "A":
        edge_csv = pd.read_csv(f'train_csvs/edges_train_A.csv', header=None)
        train_split = edge_csv[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()

        input_A_initial = pd.read_csv('intermediate/input_A_initial2.csv', header=None)
        valid_split = input_A_initial[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()

        input_A_inter = pd.read_csv('intermediate/input_A.csv', header=None)
        test_split1 = input_A_inter[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()

        input_A_final = pd.read_csv('finals/input_A_final.csv', header=None)
        test_split2 = input_A_final[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()

    else:
        edge_csv = pd.read_csv(f'train_csvs/edges_train_B.csv', header=None)
        train_split = edge_csv[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()
        train_split['target'] = train_split['target'] + id_gap

        input_B_initial = pd.read_csv('intermediate/input_B_initial2.csv', header=None)
        valid_split = input_B_initial[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()
        valid_split['target'] = valid_split['target'] + id_gap

        input_B_inter = pd.read_csv('intermediate/input_B.csv', header=None)
        test_split1 = input_B_inter[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()
        test_split1['target'] = test_split1['target'] + id_gap

        input_B_final = pd.read_csv('finals/input_B.csv', header=None)
        test_split2 = input_B_final[[0, 2, 1]].rename(columns={0: 'source', 2: 'label', 1: 'target'}).copy()
        test_split2['target'] = test_split2['target'] + id_gap

    train_gen = graph_gen.flow(train_split, negative_samples=args.negative_samples, shuffle=True)
    valid_gen = graph_gen.flow(valid_split, negative_samples=args.negative_samples)
    test1_gen = graph_gen.flow(test_split1, negative_samples=0)
    test2_gen = graph_gen.flow(test_split2, negative_samples=0)

    return train_gen, valid_gen, test1_gen, test2_gen


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tensorflow.keras.utils.set_random_seed(seed)

if __name__ == "__main__":
    setup_seed(2022)
    args = get_args()
    print(args)
    graph = construct_grapg(args)
    train(graph, args)
