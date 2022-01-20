import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ArcMargineLoss(nn.Module):
    def __init__(self, num_classes: int, dim: int, mergin: float, scale: float) -> None:
        super().__init__()

        self.scale = scale
        self.mergin = mergin
        self.weight = nn.Parameter(torch.empty(num_classes, dim))  # [n, d]
        self.num_classes = num_classes
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
        norm_weight = F.normalize(self.weight)  # [n, d]
        norm_feature = F.normalize(feature)  # [N, d]
        cos = torch.einsum("Nd,nd->Nn", norm_feature, norm_weight)  # [N, n]
        theta = torch.acos(cos)

        m_exp = torch.exp(self.scale * torch.cos(theta + self.mergin))  # [N, n]
        exp = torch.exp(self.scale * cos)  # [N, n]
        one_hot = F.one_hot(label, self.num_classes)  # [N, n]

        loss = -torch.log(
            (one_hot * m_exp).sum(dim=1)
            / (one_hot * m_exp + (1 - one_hot) * exp).sum(dim=1)
        ).mean()

        return loss


class FeatureExtractor(nn.Sequential):
    def __init__(self, dim: int, binarize: bool = True) -> None:
        base_model = torchvision.models.resnet18(pretrained=True)
        feature = nn.Sequential(*list(base_model.children())[:-1])
        flatten = nn.Flatten()
        fc = nn.Linear(512, dim)
        if binarize:
            act = nn.Tanh()
        else:
            act = nn.Identity()

        super().__init__(feature, flatten, fc, act)

