import torch
import torch.nn as nn
import torch.nn.functional as F


class MCCLoss(nn.Module):

    def __init__(self, temperature: float = 2.5):
        super().__init__()
        self.temperature = temperature

    def entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def forward(self, target_output: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = target_output.shape
        predictions = F.softmax(target_output / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = self.entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions)  # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


class PeriodicMFDModel(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.pm = nn.Sequential(  # pattern matching
            nn.Conv1d(1, 32, kernel_size=21, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 32, kernel_size=21, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Flatten(),
        )
        in_after_conv = in_features // 4
        self.fc1 = nn.Sequential(
            nn.Linear(32 * in_after_conv, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32 * in_after_conv, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
        )
        self.cls1 = nn.Linear(100, out_features)
        self.cls2 = nn.Linear(100, out_features)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.dis_loss = nn.L1Loss()
        self.mcc_loss = MCCLoss()

    def forward(self, x_s, x_t=0, y_s=0, mark=0):
        if mark == 1:
            x_s = self.pm(x_s)
            fea_s = self.fc1(x_s)
            pred_s = self.cls1(fea_s)

            x_t = self.pm(x_t)
            fea_t = self.fc1(x_t)
            out_t1 = self.cls1(fea_t)
            out_t2 = self.cls2(self.fc2(x_t))

            ce = self.ce_loss(pred_s, y_s)
            J_dis = self.dis_loss(out_t1, out_t2)
            MCC = self.mcc_loss(out_t1)
            return ce, J_dis, MCC
        elif mark == 2:
            x_s = self.pm(x_s)
            fea_s = self.fc2(x_s)
            pred_s = self.cls2(fea_s)

            x_t = self.pm(x_t)
            fea_t = self.fc2(x_t)
            out_t2 = self.cls2(fea_t)
            out_t1 = self.cls1(self.fc1(x_t))

            ce = self.ce_loss(pred_s, y_s)
            J_dis = self.dis_loss(out_t2, out_t1)
            MCC = self.mcc_loss(out_t2)
            return ce, J_dis, MCC
        else:
            x_s = self.pm(x_s)
            out1 = self.cls1(self.fc1(x_s))
            out2 = self.cls2(self.fc2(x_s))
            return out1, out2
