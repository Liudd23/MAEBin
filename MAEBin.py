import logging
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd


from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_custom_dataset
from graphmae.evaluation import get_embeddings
from graphmae.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def save_embeddings_to_tsv(embeddings, output_file):
    """
    Save embeddings to a TSV file with an index column in the form 'contig_INDEX' and a header row.

    Parameters:
        embeddings (numpy.ndarray): The embeddings to save (shape: N x F).
        output_file (str): The path to the output TSV file.
    """
    # Validate input
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.cpu().detach().numpy()

    # Generate index column (e.g., contig_0, contig_1, ...)
    index = [f"contig_{i}" for i in range(embeddings.shape[0])]

    # Generate header for features (e.g., 0, 1, 2, ...)
    header = [str(i) for i in range(embeddings.shape[1])]

    # Create a DataFrame
    embeddings_df = pd.DataFrame(embeddings, index=index, columns=header)

    # Save to TSV file
    embeddings_df.to_csv(output_file, sep='\t', header=True, index=True)

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        # if (epoch + 1) % 200 == 0:
        #     node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model


def main(args):
    torch.cuda.set_device(args.device)
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, num_features = load_custom_dataset()
    print("====== DGL Graph Information ======")
    print(f"Number of nodes: {graph.nodes()}")
    print(f"Number of edges: {graph.edges()}")
    print(f"Node feature shape: {graph.ndata['feat'].shape if 'feat' in graph.ndata else 'No node features'}")

    args.num_features = num_features

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            #余弦调度器
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            #余弦加预热调度起
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        embeddings = get_embeddings(model, graph, x, device)
        output_file = "/home/zhaozhimiao/ldd/GraphMAENoWeight/MAEembeddings.tsv"
        save_embeddings_to_tsv(embeddings, output_file)
        
        if logger is not None:
            logger.finish()


    print("finish get embeddings!")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
