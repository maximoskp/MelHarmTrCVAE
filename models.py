import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from copy import deepcopy
import pickle
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class LSTMCVAE_RoBERTa(nn.Module):
    def __init__(self, hidden_dim, device=torch.device('cpu'), **config):
        super().__init__()
        lstm_dim = 256
        roberta_model = "roberta-base"
        latent_dim = 256
        freeze_roberta = True
        if 'lstm_dim' in config.keys():
            hidden_dim_LSTM = config['lstm_dim']
        if 'roberta_model' in config.keys():
            roberta_model = config['roberta_model']
        if 'latent_dim' in config.keys():
            latent_dim = config['latent_dim']
        if 'freeze_roberta' in config.keys():
            freeze_roberta = config['freeze_roberta']
        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim

        # Load RoBERTa
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False  # Keep it frozen

        roberta_dim = self.roberta.config.hidden_size  # Usually 768 for RoBERTa-base

        # LSTM for Processing RoBERTa Outputs
        self.condition_lstm = nn.LSTM(roberta_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.condition_fc = nn.Linear(lstm_dim, roberta_dim)  # Project back to RoBERTa's dimension

        # LSTM Encoder for CVAE
        self.lstm_encoder = nn.LSTM(hidden_dim + roberta_dim, lstm_dim, batch_first=True, bidirectional=True)

        # Latent Space
        self.fc_mu = nn.Linear(lstm_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(lstm_dim * 2, latent_dim)

        # LSTM Decoder
        self.lstm_decoder = nn.LSTM(latent_dim + roberta_dim, lstm_dim, batch_first=True, bidirectional=True)

        # Output Projection to Reconstruct Encoder Hidden States
        self.decoder_fc = nn.Linear(lstm_dim * 2, hidden_dim)
    # end init

    def reparameterize(self, mu, logvar):
        """ VAE Reparameterization Trick """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    # end reparameterize

    def encode_condition(self, texts):
        """ Tokenize and encode text conditions using RoBERTa and an LSTM """
        roberta_inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():  # Freeze RoBERTa (unless fine-tuning)
            roberta_outputs = self.roberta(**roberta_inputs)

        roberta_seq_output = roberta_outputs.last_hidden_state  # (batch, seq_len, roberta_dim)

        # Pass through LSTM to model sequential dependencies
        lstm_out, (h_n, _) = self.condition_lstm(roberta_seq_output)
        
        # Use the last hidden state from the LSTM
        condition_embedding = self.condition_fc(h_n[-1])  # (batch, roberta_dim)

        return condition_embedding
    # end encode_condition

    def forward(self, encoder_hidden_states, texts):
        batch_size, seq_len, _ = encoder_hidden_states.shape

        # Encode Condition with RoBERTa + LSTM
        roberta_embeddings = self.encode_condition(texts)

        # Expand RoBERTa embeddings across the sequence
        roberta_embeddings = roberta_embeddings.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, roberta_dim)

        # Concatenate Encoder Output + RoBERTa Condition
        lstm_input = torch.cat([encoder_hidden_states, roberta_embeddings], dim=-1)

        # Encode with LSTM
        lstm_output, _ = self.lstm_encoder(lstm_input)

        # Compute Latent Variables
        mu = self.fc_mu(lstm_output[:, -1, :])  # Last timestep hidden state
        logvar = self.fc_logvar(lstm_output[:, -1, :])
        z = self.reparameterize(mu, logvar).unsqueeze(1).expand(-1, seq_len, -1)  # Repeat across sequence

        # Concatenate Latent Vector + RoBERTa Condition
        z = torch.cat([z, roberta_embeddings], dim=-1)

        # Decode with LSTM
        lstm_decoded, _ = self.lstm_decoder(z)

        # Reconstruct Encoder Hidden States
        reconstructed = self.decoder_fc(lstm_decoded)

        return reconstructed, mu, logvar
    # end forward
# end class LSTMCVAE_RoBERTa

class TransTextVAE(nn.Module):
    def __init__(self, transformer, device=torch.device('cpu'), tokenizer=None, **config):
        """
        TransTextVAE model that involves a RoBERTa-conditioned BiLSTM VAE between a pretrained
        frozen transformer encoder-decoder.

        Args:
            t_encoder: frozen encoder of the pretrained transformer
            t_decoder: frozen encoder of the pretrained transformer
            **config: arguments for the CVAE module
        """
        super(TransTextVAE, self).__init__()
        self.transformer = transformer
        self.t_encoder = transformer.model.encoder
        self.t_decoder = transformer.model.decoder
        self.device = device
        self.cvae = LSTMCVAE_RoBERTa(self.transformer.config.d_model, device=self.device, **config)
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

    def forward(self, x, texts, encoder_attention=None, generate_max_tokens=-1, num_bars=-1, temperature=1.0):
        input_ids = x
        x = self.t_encoder(x).last_hidden_state
        recon_x, mu, logvar  = self.cvae(x, texts)
        total_loss, recon_loss, kl_loss = self.compute_loss(recon_x, x, mu, logvar)
        g_recon = None
        g = None
        if generate_max_tokens > 0:
            # autoregressive process with temperature
            # output from reconstruction
            if encoder_attention is None:
                encoder_attention = (input_ids != self.transformer.config.pad_token_id)
            print('recon generation')
            g_recon = self.generate(recon_x, encoder_attention, generate_max_tokens, num_bars, temperature)
            print('normal generation')
            g = self.generate(x, encoder_attention, generate_max_tokens, num_bars, temperature)
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'x': x,
            'recon_x': recon_x,
            'generated_ids': g,
            'generated_recon_ids': g_recon,
        }
    # end forward

    def generate(self, encoder_hidden_states, encoder_attention, max_length, num_bars, temperature):
        batch_size = encoder_hidden_states.shape[0]
        bos_token_id = self.transformer.config.bos_token_id
        eos_token_id = self.transformer.config.eos_token_id
        bar_token_id = -1
        if self.tokenizer is not None:
            bar_token_id = self.tokenizer.vocab['<bar>']
        bars_left = deepcopy(num_bars)
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(self.device)  # (batch_size, 1)
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # (batch_size,)
        for _ in range(max_length):
            # Pass through the decoder
            decoder_outputs = self.t_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention,
            )

            # Get the logits of the last generated token
            logits = self.transformer.lm_head(decoder_outputs.last_hidden_state[:, -1, :])  # (batch_size, vocab_size)
            print('bars_left:', bars_left)
            # For the batch that has some bars left, zero out the eos_token_id logit
            # For the batch that has 0 bars left, zero out the bar token
            if bars_left != -1 and bar_token_id != -1:
                logits[ bars_left[:,0] > 0 , eos_token_id ] = 0
                logits[ bars_left[:,0] <= 0 , bar_token_id ] = 0

            # Apply temperature scaling and softmax
            probs = F.softmax(logits / temperature, dim=-1)  # (batch_size, vocab_size)

            # Sample next token
            next_token_ids = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            if bars_left != -1 and bar_token_id != -1:
                bars_left[ next_token_ids == bar_token_id ] -= 1

            # Stop condition: mask finished sequences
            finished |= next_token_ids.squeeze(1) == eos_token_id

            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)  # (batch_size, seq_len)

            # If all sequences are finished, stop early
            if finished.all():
                break
        return decoder_input_ids
    # end generate
# end class TransTextVAE

# ===================================================
# GNN

with open('data/chord_node_features.pickle', 'rb') as f:
    chord_node_features = pickle.load(f)

class GraphConditioningModule(nn.Module):
    def __init__(self, hidden_dim, out_dim, use_attention=False, device=torch.device('cpu')):
        """
        Graph-based conditioning module for extracting node embeddings as condition vectors.

        Args:
            hidden_dim (int): Hidden dimension of GNN layers
            out_dim (int): Dimension of the conditioning vector
            use_attention (bool): If True, uses GATConv; otherwise, uses GCNConv.
        """
        super(GraphConditioningModule, self).__init__()

        self.use_attention = use_attention
        self.device = device

        self.chord_node_features = torch.Tensor(chord_node_features).to(self.device)
        
        if use_attention:
            self.gnn1 = GATConv(self.chord_node_features.shape[1], hidden_dim).to(self.device)
            self.gnn2 = GATConv(hidden_dim, out_dim).to(self.device)
        else:
            self.gnn1 = GCNConv(self.chord_node_features.shape[1], hidden_dim).to(self.device)
            self.gnn2 = GCNConv(hidden_dim, out_dim).to(self.device)
    # end init

    def forward(self, batch_graph):
        """
        Args:
            batch_graph (Batch): Batched graph object from PyG
            node_indices (torch.Tensor): Shape (batch_size,), selected node per sample
        
        Returns:
            condition_vectors (torch.Tensor): Shape (batch_size, out_dim)
        """
        edge_index = batch_graph.edge_index.to(self.device)
        edge_attr = batch_graph.edge_attr.to(self.device)
        node_indices = batch_graph.node_indices.to(self.device)
        batch_index = batch_graph.batch.to(self.device)  # Node-to-batch mapping
        # x = torch.ones((batch_graph.num_nodes, 1), device=batch_graph.edge_index.device)  # Dummy features
        x = self.chord_node_features[node_indices].to(self.device)  # Shape: (num_nodes_in_batch, 12 or whatever)

        x = F.relu(self.gnn1(x, edge_index, edge_attr)).to(self.device)
        x = F.relu(self.gnn2(x, edge_index, edge_attr)).to(self.device)

        # Aggregate node embeddings per batch
        x = global_mean_pool(x, batch_index).to(self.device)  # (batch_size, output_dim)
        return x
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
    def __init__(self, transformer_dim, device=torch.device('cpu'), **config):
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
        self.device = device
        
        self.lstm_encoder = BiLSTMEncoder(transformer_dim, hidden_dim_LSTM)
        self.lstm_decoder = BiLSTMDecoder(hidden_dim_LSTM, transformer_dim)

        self.graph_conditioning = GraphConditioningModule(hidden_dim_GNN, \
                            condition_dim, use_attention=use_attention, device=self.device)

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
        edge_indices = []
        edge_attrs = []
        node_indices = []
        batch_indices = []

        for batch_idx in range(batch_size):
            matrix = markov_matrices[batch_idx]

            # Get nonzero indices (edges) and weights (edge attributes)
            edges = torch.nonzero(matrix, as_tuple=False).T  # Shape: (2, num_edges)
            weights = matrix[edges[0], edges[1]]  # Edge weights

            # Append edges and attributes
            edge_indices.append(edges)
            edge_attrs.append(weights)

            # Node indices (all nodes in this graph)
            node_indices.append(torch.arange(num_nodes, dtype=torch.long))

            # Batch assignment for nodes
            batch_indices.append(torch.full((num_nodes,), batch_idx, dtype=torch.long))

        # Concatenate to form a single large batched graph
        edge_index = torch.cat(edge_indices, dim=1)  # Shape: (2, total_edges)
        edge_attr = torch.cat(edge_attrs, dim=0)  # Shape: (total_edges,)
        node_indices = torch.cat(node_indices, dim=0)  # Shape: (total_nodes,)
        batch_index = torch.cat(batch_indices, dim=0)  # Shape: (total_nodes,)

        # Create batched PyG Data object
        batched_graph = Data(edge_index=edge_index, edge_attr=edge_attr, node_indices=node_indices, batch=batch_index)

        return batched_graph
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
        batched_graph = self.build_batch_graphs( transitions )
        condition = self.graph_conditioning(batched_graph)  # Shape: (batch_size, condition_dim)

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
        self.t_encoder = transformer.model.encoder
        self.t_decoder = transformer.model.decoder
        self.device = device
        self.cvae = CVAE(self.transformer.config.d_model, device=self.device, **config)
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

    def forward(self, x, transitions, encoder_attention=None, generate_max_tokens=-1, num_bars=-1, temperature=1.0):
        input_ids = x
        x = self.t_encoder(x).last_hidden_state
        recon_x, mu, logvar  = self.cvae(x, transitions)
        total_loss, recon_loss, kl_loss = self.compute_loss(recon_x, x, mu, logvar)
        g_recon = None
        g = None
        generated_markov = None
        recon_markov = None
        if generate_max_tokens > 0:
            # autoregressive process with temperature
            # output from reconstruction
            if encoder_attention is None:
                encoder_attention = (input_ids != self.transformer.config.pad_token_id)
            print('recon generation')
            g_recon = self.generate(recon_x, encoder_attention, generate_max_tokens, num_bars, temperature)
            print('normal generation')
            g = self.generate(x, encoder_attention, generate_max_tokens, num_bars, temperature)
            if self.tokenizer is not None:
                generated_markov = self.tokenizer.make_markov_from_token_ids_tensor(g)
                recon_markov = self.tokenizer.make_markov_from_token_ids_tensor(g_recon)
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'x': x,
            'recon_x': recon_x,
            'generated_ids': g,
            'generated_recon_ids': g_recon,
            'generated_markov': generated_markov,
            'recon_markov': recon_markov
        }
    # end forward

    def generate(self, encoder_hidden_states, encoder_attention, max_length, num_bars, temperature):
        batch_size = encoder_hidden_states.shape[0]
        bos_token_id = self.transformer.config.bos_token_id
        eos_token_id = self.transformer.config.eos_token_id
        bar_token_id = -1
        if self.tokenizer is not None:
            bar_token_id = self.tokenizer.vocab['<bar>']
        bars_left = deepcopy(num_bars)
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(self.device)  # (batch_size, 1)
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # (batch_size,)
        for _ in range(max_length):
            # Pass through the decoder
            decoder_outputs = self.t_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention,
            )

            # Get the logits of the last generated token
            logits = self.transformer.lm_head(decoder_outputs.last_hidden_state[:, -1, :])  # (batch_size, vocab_size)
            print('bars_left:', bars_left)
            # For the batch that has some bars left, zero out the eos_token_id logit
            # For the batch that has 0 bars left, zero out the bar token
            if bars_left != -1 and bar_token_id != -1:
                logits[ bars_left[:,0] > 0 , eos_token_id ] = 0
                logits[ bars_left[:,0] <= 0 , bar_token_id ] = 0

            # Apply temperature scaling and softmax
            probs = F.softmax(logits / temperature, dim=-1)  # (batch_size, vocab_size)

            # Sample next token
            next_token_ids = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            if bars_left != -1 and bar_token_id != -1:
                bars_left[ next_token_ids == bar_token_id ] -= 1

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