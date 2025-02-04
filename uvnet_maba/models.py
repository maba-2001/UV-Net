import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from . import encoders
from torchmetrics import AUROC
import random
import numpy as np

# Set all relevant seeds to 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

auroc = AUROC(num_classes=2)

class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

###############################################################################
# Regression model
###############################################################################

class UVNetRegressor(nn.Module):
    """
    UV-Net regression model to predict the number of bends.
    """

    def __init__(
        self,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net regressor model
        
        Args:
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear regressor. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        # Changed to a regressor that outputs a single continuous value (number of bends)
        self.regressor = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, 1)  # Single output value for regression
        )

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Predicted value (batch_size x 1) representing the number of bends
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to a continuous value for regression
        out = self.regressor(graph_emb)
        return out

class Regression(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the regressor.
    """

    def __init__(self):
        super().__init__()
        self.model = UVNetRegressor()
        # MAE (Mean Absolute Error) for regression tasks
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.mse_loss(logits.squeeze(), labels.float(), reduction="mean")  # MSE loss for regression
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_mae", self.train_mae(logits.squeeze(), labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.mse_loss(logits.squeeze(), labels.float(), reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_mae", self.val_mae(logits.squeeze(), labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        
        # Ensure both logits and labels are 1D (batch_size,)
        logits = logits.squeeze()  # Ensure logits are 1D
        labels = labels.squeeze()  # Ensure labels are 1D
        
        # Calculate loss
        loss = F.mse_loss(logits, labels.float(), reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=False, sync_dist=True)
        
        # Ensure predictions and targets are of the same shape before calculating MAE
        preds = logits
        self.log("test_mae", self.test_mae(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
###############################################################################
# Classification model
###############################################################################


class UVNetClassifier(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
        self,
        num_classes,
        crv_emb_dim=128,
        srf_emb_dim=128,
        graph_emb_dim=256,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)
        return out


class Classification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes, learning_rate=1e-3):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetClassifier(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(logits, dim=-1)
        auc = auroc(preds, labels)
        self.log('train_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)        
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(logits, dim=-1)
        auc = auroc(preds, labels)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=False, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        auc = auroc(preds, labels)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

###############################################################################
# No embedding model
###############################################################################

class UVNoEmbedClassification(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
        self,
        num_classes,
        crv_emb_dim=60,
        srf_emb_dim=700,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph, nfeat=None):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata['x']
        input_srf_feat = batched_graph.ndata['x']
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, input_srf_feat, input_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)
        return out
class NoEmbedClassification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes, learning_rate=1e-3):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNoEmbedClassification(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()    
        self.learning_rate = learning_rate
        
    def forward(self, batched_graph,nfeat=None):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(logits, dim=-1)
        auc = auroc(preds, labels)
        self.log('train_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        preds = F.softmax(logits, dim=-1)
        auc = auroc(preds, labels)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device).long()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        auc = auroc(preds, labels)
        self.log('test_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=False, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

###############################################################################
# Segmentation model
###############################################################################


class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """

    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = encoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        # A graph neural network that message passes face and edge features
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout
        )

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (total_nodes_in_batch x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Repeat the global graph embedding so that it can be
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, num_classes, crv_in_channels=6):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
        # Setting compute_on_step = False to compute "part IoU"
        # This is because we want to compute the IoU on the entire dataset
        # at the end to account for rare classes, rather than within each batch
        self.train_iou = torchmetrics.IoU(
            num_classes=num_classes, compute_on_step=False
        )
        self.val_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.test_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)

        self.train_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )
        self.val_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )
        self.test_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)

    def test_epoch_end(self, outs):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
