import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch

class GraphConditioningModule(nn.Module):
    def __init__(self, hidden_dim, out_dim, use_attention=False):
        """
        Graph-based conditioning module for extracting node embeddings as condition vectors.

        Args:
            hidden_dim (int): Hidden dimension of GNN layers
            out_dim (int): Dimension of the conditioning vector
            use_attention (bool): If True, uses GATConv; otherwise, uses GCNConv.
        """
        super(GraphConditioningModule, self).__init__()

        self.use_attention = use_attention
        
        if use_attention:
            self.gnn1 = GATConv(1, hidden_dim)
            self.gnn2 = GATConv(hidden_dim, hidden_dim)
        else:
            self.gnn1 = GCNConv(1, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, out_dim)
    # end init

    def forward(self, batch_graph, node_indices):
        """
        Args:
            batch_graph (Batch): Batched graph object from PyG
            node_indices (torch.Tensor): Shape (batch_size,), selected node per sample
        
        Returns:
            condition_vectors (torch.Tensor): Shape (batch_size, out_dim)
        """
        x = torch.ones((batch_graph.num_nodes, 1), device=batch_graph.edge_index.device)  # Dummy features

        x = F.relu(self.gnn1(x, batch_graph.edge_index))
        x = F.relu(self.gnn2(x, batch_graph.edge_index))
        
        node_embeddings = x[node_indices]  # Shape: (batch_size, hidden_dim)
        condition_vectors = self.fc(node_embeddings)  # Shape: (batch_size, out_dim)

        return condition_vectors
    # end forward
# end class GraphConditioningModule

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        BiLSTM encoder for sequential input data.
        
        Args:
            input_dim (int): Input feature dimension per timestep
            hidden_dim (int): Hidden state dimension
        """
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Project bidirectional output
    # end init

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            hidden_state (torch.Tensor): Shape (batch_size, hidden_dim)
        """
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)  # Concatenate bidirectional outputs
        return self.fc(h_n)  # Shape: (batch_size, hidden_dim)
    # end forward
# end class BiLSTMEncoder

class BiLSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        """
        BiLSTM decoder that reconstructs sequences from latent representations.

        Args:
            hidden_dim (int): Hidden dimension of LSTM
            output_dim (int): Output feature dimension per timestep
        """
        super(BiLSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    # end init

    def forward(self, z, seq_len):
        """
        Args:
            z (torch.Tensor): Latent variable (batch_size, hidden_dim)
            seq_len (int): Target sequence length
        
        Returns:
            recon_x (torch.Tensor): Shape (batch_size, seq_len, output_dim)
        """
        z = z.unsqueeze(1).repeat(1, seq_len, 1)  # Expand latent state across sequence
        output, _ = self.lstm(z)
        return self.fc(output)  # Shape: (batch_size, seq_len, output_dim)
    # end forward
# end class BiLSTMDecoder

class CVAE(nn.Module):
    def __init__(self, transformer_dim, **config):
        """
        CVAE model integrating BiLSTM encoder-decoder and GNN-based conditioning.

        Args:
            transformer_dim (int): Input and output feature dimension per timestep
            hidden_dim_LSTM (int): Hidden dimension for BiLSTM
            hidden_dim_GNN (int): Hidden dimension for GNN
            latent_dim (int): Dimension of the VAE latent space
            condition_dim (int): Dimension of the conditioning vector
            use_attention (bool): If True, uses GATConv; otherwise, uses GCNConv.
        """
        super(CVAE, self).__init__()

        hidden_dim_LSTM = 256
        hidden_dim_GNN = 256
        latent_dim = 256
        condition_dim = 128
        use_attention=False
        if 'hidden_dim_LSTM' in config.keys():
            hidden_dim_LSTM = config['hidden_dim_LSTM']
        if 'hidden_dim_GNN' in config.keys():
            hidden_dim_GNN = config['hidden_dim_GNN']
        if 'latent_dim' in config.keys():
            latent_dim = config['latent_dim']
        if 'condition_dim' in config.keys():
            condition_dim = config['condition_dim']
        if 'use_attention' in config.keys():
            use_attention = config['use_attention']
        
        self.lstm_encoder = BiLSTMEncoder(transformer_dim, hidden_dim_LSTM)
        self.lstm_decoder = BiLSTMDecoder(hidden_dim_LSTM, transformer_dim)

        self.graph_conditioning = GraphConditioningModule(hidden_dim_GNN, condition_dim, use_attention=use_attention)

        # Latent space transformations
        self.fc_mu = nn.Linear(hidden_dim_LSTM + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_LSTM + condition_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim + condition_dim, hidden_dim_LSTM)
    # end init

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    # end reparametrize

    def build_batch_graphs(self, markov_matrices):
        """
        Converts a batch of Markov transition matrices into a single batched PyTorch Geometric graph.

        Args:
            markov_matrices (torch.Tensor): (batch_size, num_nodes, num_nodes) tensor

        Returns:
            batch_graph (Batch): Batched PyG graph containing all transition matrices
            node_indices (torch.Tensor): (batch_size,) tensor containing a node index per sample
        """
        batch_size, num_nodes, _ = markov_matrices.shape
        graphs = []
        node_indices = []

        for b in range(batch_size):
            # Extract nonzero entries (source, target) where transition probability > 0
            source_nodes, target_nodes = torch.nonzero(markov_matrices[b], as_tuple=True)
            edge_probs = markov_matrices[b][source_nodes, target_nodes]  # Extract transition probabilities

            # Create edge_index
            edge_index = torch.stack([source_nodes, target_nodes], dim=0)  # Shape (2, num_edges)
            
            # Create graph data object
            graph = Data(edge_index=edge_index, edge_attr=edge_probs, num_nodes=num_nodes)
            graphs.append(graph)

            # Select a random node to condition on (or use a rule)
            node_indices.append(torch.randint(0, num_nodes, (1,)))

        # Batch all graphs into a single PyG Batch object
        batch_graph = Batch.from_data_list(graphs)
        node_indices = torch.cat(node_indices)  # Shape (batch_size,)

        return batch_graph, node_indices
    # end build_batch_graphs

    def forward(self, x, transitions):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
            transitions: markov matrix
        
        Returns:
            recon_x (torch.Tensor): Reconstructed sequence
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
        """
        h = self.lstm_encoder(x)  # Shape: (batch_size, hidden_dim)
        batch_graph, node_indices = self.build_batch_graphs( transitions )
        condition = self.graph_conditioning(batch_graph, node_indices)  # Shape: (batch_size, condition_dim)

        h_cond = torch.cat([h, condition], dim=-1)  # Shape: (batch_size, hidden_dim_LSTM + condition_dim)

        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, condition], dim=-1)
        z_hidden = self.fc_z(z_cond)  # Shape: (batch_size, hidden_dim_LSTM)

        recon_x = self.lstm_decoder(z_hidden, x.shape[1])  # Reconstruct sequence

        return recon_x, mu, logvar
    # end forward
# end CVAE

class TransGraphVAE(nn.Module):
    def __init__(self, transformer, device=torch.device('cpu'), tokenizer=None, **config):
        """
        TransGraphVAE model that involves a GNN-conditioned BiLSTM VAE between a pretrained
        frozen transformer encoder-decoder.

        Args:
            t_encoder: frozen encoder of the pretrained transformer
            t_decoder: frozen encoder of the pretrained transformer
            **config: arguments for the CVAE module
        """
        super(TransGraphVAE, self).__init__()
        self.transformer = transformer
        self.t_encoder = transformer.get_encoder()
        self.t_decoder = transformer.get_decoder()
        self.cvae = CVAE(self.transformer.config.d_model, **config)
        self.device = device
        self.tokenizer = tokenizer
    # end init

    def compute_loss(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss (Reconstruction Loss + KL Divergence).
        
        Args:
            recon_x (torch.Tensor): Reconstructed sequences (batch_size, seq_len, transformer_dim)
            x (torch.Tensor): Ground truth sequences (batch_size, seq_len, transformer_dim)
            mu (torch.Tensor): Mean of latent distribution (batch_size, latent_dim)
            logvar (torch.Tensor): Log variance of latent distribution (batch_size, latent_dim)
        
        Returns:
            loss (torch.Tensor): Combined loss
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss, recon_loss, kl_loss
    # end compute_loss

    def forward(self, x, transitions, encoder_attention=None, generate_max_tokens=-1, temperature=1.0):
        input_ids = x
        x = self.t_encoder(x).last_hidden_state
        recon_x, mu, logvar  = self.cvae(x, transitions)
        total_loss, recon_loss, kl_loss = self.compute_loss(recon_x, x, mu, logvar)
        g_recon = None
        g = None
        if generate_max_tokens > 0:
            # autoregressive process with temperature
            # output from reconstruction
            if encoder_attention is None:
                encoder_attention = (input_ids != self.transformer.config.pad_token_id)
            print('recon generation')
            g_recon = self.generate(recon_x, encoder_attention, generate_max_tokens, temperature)
            print('normal generation')
            g = self.generate(x, encoder_attention, generate_max_tokens, temperature)
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'x': x,
            'recon_x': recon_x,
            'generated_ids': g_recon,
            'generated_recon_ids': g
        }
    # end forward

    def generate(self, encoder_hidden_states, encoder_attention, max_length, temperature):
        batch_size = encoder_hidden_states.shape[0]
        print('batch_size:', batch_size)
        if self.tokenizer is None:
            bos_token_id = self.transformer.config.bos_token_id
            eos_token_id = self.transformer.config.eos_token_id
        else:
            bos_token_id = self.tokenizer.vocab[self.tokenizer.harmony_tokenizer.start_harmony_token]
            eos_token_id = self.transformer.config.eos_token_id
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(self.device)  # (batch_size, 1)
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # (batch_size,)
        print('decoder_input_ids:', decoder_input_ids)
        for _ in range(max_length):
            # Pass through the decoder
            decoder_outputs = self.t_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention,
            )

            # Get the logits of the last generated token
            logits = decoder_outputs.last_hidden_state[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature scaling and softmax
            probs = F.softmax(logits / temperature, dim=-1)  # (batch_size, vocab_size)

            # Sample next token
            next_token_ids = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Stop condition: mask finished sequences
            finished |= next_token_ids.squeeze(1) == eos_token_id

            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)  # (batch_size, seq_len)

            # If all sequences are finished, stop early
            if finished.all():
                break
        return decoder_input_ids
    # end generate
# end class TransGraphVAE