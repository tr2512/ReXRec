import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description="explainer")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="finetune", help="finetune or generate")
    parser.add_argument("--no-injection", action='store_true', help="Train model without injection")
    parser.add_argument("--no-profile", action='store_true', help="Train model without profiles")
    parser.add_argument("--out_name", type=str, required=True, help="Select a name which will be added to the name of output files.")
    parser.add_argument("--graph", type=str, default='light-gcn', help='GNN model types')
    parser.add_argument('--random-embeddings', action='store_true', help='Use random embedding instead of GNN intialized')
    parser.add_argument('--random-random', action='store_true', help='Even more randomness')
    return parser.parse_args()

args = parse_configure()