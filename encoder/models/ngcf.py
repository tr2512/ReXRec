from utils.parse import args
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class NGCF(nn.Module):
    def __init__(self, num_user, num_item, adj, dropout, device):
        super(NGCF, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = args.embedding_dim
        self.n_layers = args.n_layers

        # Convert the adjacency matrix to a PyTorch sparse tensor
        if isinstance(adj, sp.coo_matrix):
            # Check if adj is a user-item matrix
            if adj.shape == (num_user, num_item):
                # Create the bi-adj matrix [0 R; R^T 0]
                R = adj
                R_t = adj.transpose().tocsr()
                
                # Create top-right and bottom-left blocks
                top_right = R
                bottom_left = R_t
                
                # Create top-left and bottom-right blocks as empty
                top_left = sp.coo_matrix((num_user, num_user))
                bottom_right = sp.coo_matrix((num_item, num_item))
                
                # Combine into a full adjacency matrix
                A_full = sp.vstack([
                    sp.hstack([top_left, top_right]),
                    sp.hstack([bottom_left, bottom_right])
                ]).tocoo()
            elif adj.shape == (num_user + num_item, num_user + num_item):
                # adj is already a full adjacency matrix
                A_full = adj
            else:
                raise ValueError(f"Adjacency matrix shape {adj.shape} does not match expected shapes.")

            # Extract the indices and values from the COO matrix
            indices = torch.from_numpy(
                np.vstack((A_full.row, A_full.col)).astype(np.int64)
            )
            values = torch.from_numpy(A_full.data.astype(np.float32))
            shape = A_full.shape

            # Compute the degree of each node
            degree = np.array(A_full.sum(axis=1)).flatten()
            epsilon = 1e-10  # Small value to prevent division by zero

            # Compute D^(-0.5)
            d_inv_sqrt = np.power(degree + epsilon, -0.5)
            d_inv_sqrt = torch.tensor(d_inv_sqrt, dtype=torch.float32)

            # Normalize the adjacency matrix: D^(-0.5) * A * D^(-0.5)
            values = values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]

            # Create the normalized PyTorch sparse tensor
            adj_normalized = torch.sparse_coo_tensor(
                indices,
                values,
                shape,
                dtype=torch.float32
            )

        # Move the adjacency matrix to the specified device
        self.adj = adj_normalized.to(device)

        # Initialize user and item embeddings on the specified device
        self.user_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_user, self.embedding_dim, device=device))
        )
        self.item_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_item, self.embedding_dim, device=device))
        )

        # Define weight matrices for each layer
        self.weight = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            for _ in range(self.n_layers)
        ])

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Activation function
        self.activation = F.relu

    def forward(self, adj):
        # Concatenate user and item embeddings
        embeds = torch.cat([self.user_embeds, self.item_embeds], dim=0)
        all_embeddings = [embeds]

        for layer in range(self.n_layers):
            # Propagate embeddings
            agg_embeddings = torch.spmm(adj.cuda(), all_embeddings[-1])
            # Linear transformation
            transformed = self.weight[layer](agg_embeddings)
            # Apply activation and dropout
            transformed = self.activation(transformed)
            transformed = self.dropout(transformed)
            all_embeddings.append(transformed)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=1)  # Shape: (num_nodes, n_layers + 1, embed_dim)
        final_embeddings = torch.mean(all_embeddings, dim=1)  # Shape: (num_nodes, embed_dim)
        self.final_embeds = final_embeddings

        # Split back into user and item embeddings
        return final_embeddings[:self.num_user], final_embeddings[self.num_user:]

    def cal_loss(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Calculate the difference between the positive and negative item predictions
        pos_scores = torch.sum(anc_embeds * pos_embeds, dim=1)
        neg_scores = torch.sum(anc_embeds * neg_embeds, dim=1)
        diff_scores = pos_scores - neg_scores

        # Compute the BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(diff_scores)))

        # Regularization term (optional)
        reg_loss = args.weight_decay * (
            anc_embeds.norm(2).pow(2)
            + pos_embeds.norm(2).pow(2)
            + neg_embeds.norm(2).pow(2)
        )

        return loss + reg_loss

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        users, _ = batch_data
        pck_user_embeds = user_embeds[users]
        full_preds = pck_user_embeds @ item_embeds.T
        return full_preds