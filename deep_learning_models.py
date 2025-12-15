"""Deep learning models for sequence-based variant calling in MRD detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import PipelineConfig
from .performance import get_ml_performance_tracker


class SequenceDataset(Dataset):
    """Dataset for sequence-based variant calling."""

    def __init__(self, sequences: list[str], labels: np.ndarray, max_length: int = 100):
        """Initialize sequence dataset.

        Args:
            sequences: List of DNA sequences
            labels: Target labels
            max_length: Maximum sequence length for padding
        """
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_length = max_length

        # DNA base to integer mapping
        self.base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sequence and label for given index."""
        seq = self.sequences[idx]

        # Convert sequence to one-hot encoding
        one_hot = self._sequence_to_onehot(seq)

        return one_hot, self.labels[idx]

    def _sequence_to_onehot(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to one-hot encoding."""
        # Pad or truncate sequence
        if len(sequence) < self.max_length:
            sequence = sequence + "N" * (self.max_length - len(sequence))
        else:
            sequence = sequence[: self.max_length]

        # Convert to one-hot
        one_hot = torch.zeros(self.max_length, 5)  # 5 channels for A, C, G, T, N

        for i, base in enumerate(sequence):
            if base in self.base_to_int:
                one_hot[i, self.base_to_int[base]] = 1.0
            else:
                one_hot[i, 4] = 1.0  # N (unknown)

        return one_hot


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on biologically relevant sequence regions."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """Initialize attention layer.

        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention layer."""
        batch_size = x.size(0)

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(
            batch_size,
            -1,
            self.num_heads,
            self.hidden_dim // self.num_heads,
        ).transpose(1, 2)
        K = K.view(
            batch_size,
            -1,
            self.num_heads,
            self.hidden_dim // self.num_heads,
        ).transpose(1, 2)
        V = V.view(
            batch_size,
            -1,
            self.num_heads,
            self.hidden_dim // self.num_heads,
        ).transpose(1, 2)

        # Attention computation
        scores = (
            torch.matmul(Q, K.transpose(-2, -1))
            / (self.hidden_dim // self.num_heads) ** 0.5
        )
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        )

        # Final linear layer
        output = self.fc_out(context)

        return output


class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for sequence-based variant calling with attention."""

    def __init__(
        self,
        input_channels: int = 5,
        sequence_length: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        """Initialize CNN-LSTM model.

        Args:
            input_channels: Number of input channels (5 for DNA bases + N)
            sequence_length: Length of input sequences
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        # CNN layers for local feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Output sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN-LSTM model."""
        # CNN feature extraction
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)

        # LSTM sequence modeling
        # x shape: (batch_size, input_channels, sequence_length) -> (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention if enabled
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            # Use attention-weighted representation
            x = attn_out.mean(dim=1)  # Global average pooling of attention output
        else:
            # Use the last hidden state
            x = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Output probability
        output = self.sigmoid(x)

        return output.squeeze()


class HybridModel(nn.Module):
    """Hybrid model combining statistical features with deep learning sequence features."""

    def __init__(
        self,
        sequence_model: CNNLSTMModel,
        statistical_features_dim: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        """Initialize hybrid model.

        Args:
            sequence_model: Pre-trained sequence model
            statistical_features_dim: Dimension of statistical features
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()

        self.sequence_model = sequence_model

        # Freeze sequence model parameters (transfer learning)
        for param in self.sequence_model.parameters():
            param.requires_grad = False

        # Statistical feature processing
        self.statistical_encoder = nn.Sequential(
            nn.Linear(statistical_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined processing
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 1, hidden_dim),  # +1 for sequence output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence_input: torch.Tensor,
        statistical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through hybrid model.

        Args:
            sequence_input: Sequence tensor for CNN-LSTM
            statistical_features: Statistical feature tensor

        Returns:
            Combined prediction probability
        """
        # Get sequence-based prediction
        sequence_output = self.sequence_model(sequence_input)

        # Process statistical features
        statistical_output = self.statistical_encoder(statistical_features)

        # Combine features
        combined = torch.cat([sequence_output.unsqueeze(1), statistical_output], dim=1)

        # Final prediction
        output = self.combined_fc(combined)

        return output.squeeze()


class DeepLearningVariantCaller:
    """Enhanced deep learning variant caller with multiple architectures."""

    def __init__(
        self,
        config: PipelineConfig,
        model_type: str = "hybrid",
        device: str = "auto",
    ):
        """Initialize enhanced deep learning variant caller.

        Args:
            config: Pipeline configuration
            model_type: Type of model ('cnn_lstm', 'hybrid', 'transformer')
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.config = config
        self.model_type = model_type

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = None
        self.trainer = None

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50

        # Performance tracking
        self.tracker = get_ml_performance_tracker()

    def prepare_data(
        self,
        collapsed_df: pd.DataFrame,
    ) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
        """Prepare data for deep learning training.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Tuple of (train_loader, val_loader, data_info)
        """
        print("  Preparing data for deep learning...")

        # Extract sequences (if available)
        if "consensus_sequence" in collapsed_df.columns:
            sequences = collapsed_df["consensus_sequence"].tolist()
        else:
            # Generate synthetic sequences based on variant status
            sequences = self._generate_synthetic_sequences(collapsed_df)

        # Extract labels
        if "is_variant" in collapsed_df.columns:
            labels = collapsed_df["is_variant"].values
        else:
            # Generate synthetic labels
            labels = self._generate_synthetic_labels(collapsed_df)

        # Split data
        n_samples = len(sequences)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_size = int(0.8 * n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create datasets
        train_sequences = [sequences[i] for i in train_indices]
        train_labels = labels[train_indices]

        val_sequences = [sequences[i] for i in val_indices]
        val_labels = labels[val_indices]

        # Create sequence datasets
        max_seq_length = max(len(seq) for seq in sequences) if sequences else 100

        train_dataset = SequenceDataset(train_sequences, train_labels, max_seq_length)
        val_dataset = SequenceDataset(val_sequences, val_labels, max_seq_length)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        data_info = {
            "n_samples": n_samples,
            "max_sequence_length": max_seq_length,
            "positive_ratio": np.mean(labels),
            "train_size": len(train_sequences),
            "val_size": len(val_sequences),
        }

        return train_loader, val_loader, data_info

    def _generate_synthetic_sequences(self, collapsed_df: pd.DataFrame) -> list[str]:
        """Generate synthetic DNA sequences for testing."""
        sequences = []

        for _, row in collapsed_df.iterrows():
            # Generate random sequence based on variant status
            length = 50  # Default length
            bases = ["A", "C", "G", "T"]

            if "is_variant" in row and row["is_variant"]:
                # Add some mutations for variant sequences
                seq = "".join(np.random.choice(bases) for _ in range(length))
            else:
                # Normal sequence
                seq = "".join(np.random.choice(bases) for _ in range(length))

            sequences.append(seq)

        return sequences

    def _generate_synthetic_labels(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Generate synthetic labels for training."""
        # Use allele fraction as primary signal with noise
        af_signal = collapsed_df.get(
            "allele_fraction",
            pd.Series([0.001] * len(collapsed_df)),
        )

        # Add noise based on quality
        quality_factor = (
            collapsed_df.get("quality_score", pd.Series([25] * len(collapsed_df)))
            / 50.0
        )

        # Combine signals
        probabilities = np.clip(af_signal * quality_factor * 100, 0, 1)

        # Add some randomness
        noise = np.random.normal(0, 0.1, len(probabilities))
        noisy_probabilities = np.clip(probabilities + noise, 0, 1)

        # Convert to binary labels
        return (noisy_probabilities > 0.5).astype(int)

    def _extract_statistical_features(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Extract statistical features for hybrid model."""
        features = []

        # Basic features
        basic_features = [
            "family_size",
            "quality_score",
            "consensus_agreement",
            "allele_fraction",
        ]

        for feature in basic_features:
            if feature in collapsed_df.columns:
                features.append(collapsed_df[feature].values.reshape(-1, 1))

        # Derived features
        if (
            "family_size" in collapsed_df.columns
            and "quality_score" in collapsed_df.columns
        ):
            family_quality_ratio = collapsed_df["family_size"] / (
                collapsed_df["quality_score"] + 1
            )
            features.append(family_quality_ratio.values.reshape(-1, 1))

        # Combine features
        if features:
            return np.hstack(features)
        else:
            return np.zeros((len(collapsed_df), 1))

    def _extract_statistical_features_batch(self, batch_df: pd.DataFrame) -> np.ndarray:
        """Extract statistical features for a batch."""
        return self._extract_statistical_features(batch_df)

    def get_model_summary(self) -> dict[str, Any]:
        """Get model architecture summary."""
        if self.model is None:
            return {}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_type": self.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_channels": 5,
            "max_sequence_length": 100,
        }


class DNATransformerModel(nn.Module):
    """Transformer-based model for DNA sequence analysis with transfer learning support."""

    def __init__(
        self,
        vocab_size: int = 5,
        max_length: int = 100,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_pretrained: bool = False,
    ):
        """Initialize DNA transformer model.

        Args:
            vocab_size: Vocabulary size (5 for DNA bases + N)
            max_length: Maximum sequence length
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            use_pretrained: Whether to use pre-trained DNA language model
        """
        super().__init__()

        self.use_pretrained = use_pretrained

        if use_pretrained:
            # Use pre-trained DNA BERT embeddings
            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    "zhihan1996/DNABERT-2-117M",
                )
                self.bert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")

                # Freeze BERT parameters for transfer learning
                for param in self.bert_model.parameters():
                    param.requires_grad = False

                # Use BERT hidden size
                bert_hidden = self.bert_model.config.hidden_size
                self.projection = nn.Linear(bert_hidden, d_model)

            except ImportError:
                print("Warning: transformers not available, using custom embeddings")
                use_pretrained = False

        if not use_pretrained:
            # Custom embedding layer
            self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def _create_positional_encoding(
        self,
        max_length: int,
        d_model: int,
    ) -> torch.Tensor:
        """Create positional encoding for transformer."""
        pos_encoding = torch.zeros(max_length, d_model)

        for pos in range(max_length):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(
                        pos / (10000 ** (2 * i / d_model)),
                    )

        return pos_encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer model."""
        batch_size, seq_length = x.size()

        if self.use_pretrained:
            # Use pre-trained DNA BERT
            # Convert token IDs back to sequences for BERT
            # This is a simplified implementation - in practice, you'd need proper tokenization
            x_str = [self._ids_to_sequence(seq) for seq in x.cpu().numpy()]

            # Get BERT embeddings (batch processing would be needed for efficiency)
            embeddings = []
            for seq in x_str:
                inputs = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                # Use CLS token representation
                embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(embedding)

            x = torch.stack(embeddings).to(x.device)
            x = self.projection(x)

        else:
            # Custom embedding + positional encoding
            x = self.embedding(x)
            x = x + self.pos_encoding[:, :seq_length, :].to(x.device)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        output = self.classifier(x)

        return output.squeeze()

    def _ids_to_sequence(self, token_ids: np.ndarray) -> str:
        """Convert token IDs back to DNA sequence."""
        # Simplified - in practice, you'd need the tokenizer's id_to_token mapping
        # For now, just return a placeholder
        return "ACGT" * (len(token_ids) // 4)  # Placeholder
