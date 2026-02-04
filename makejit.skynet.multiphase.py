from typing import Any

import numpy as np
import torch
import torch.nn as nn

class Skynet(nn.Module):
    """
    Skynet is a phase picker with a U-Net architecture with a focus on regional phases and a long receptive field.
    The model offers first arrival pickers and multiphase pickers for Pn, Pg, Sn, and Sg phases.

    .. document_args:: seisbench.models Skynet
    """



    def __init__(
        self,
        in_channels=3,
        classes=5,
        phases="PSN",
        sampling_rate=100,
        norm="peak",
        **kwargs,
    ):
        citation = (
            "Aguilar Suarez, A. L., & Beroza, G. (2025). "
            "Picking Regional Seismic Phase Arrival Times with Deep Learning. "
            "Seismica, 4(1). "
            "https://doi.org/10.26443/seismica.v4i1.1431"
        )

        super().__init__()

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        kernel_size = 7
        stride = 4

        pad_size = int(kernel_size / 2)

        self.conv1 = nn.Conv1d(
            in_channels, 8, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=kernel_size, stride=1, padding="same")
        self.bn2 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv3 = nn.Conv1d(
            8, 8, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn3 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.conv4 = nn.Conv1d(8, 11, kernel_size=kernel_size, stride=1, padding="same")
        self.bn4 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.conv5 = nn.Conv1d(
            11, 11, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn5 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.conv6 = nn.Conv1d(
            11, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn6 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.conv7 = nn.Conv1d(
            16, 16, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn7 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.conv8 = nn.Conv1d(
            16, 22, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn8 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.conv9 = nn.Conv1d(
            22, 22, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn9 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.conv10 = nn.Conv1d(
            22, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn10 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        # extra from original UNet
        self.conv11 = nn.Conv1d(
            32, 32, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bn11 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        self.conv12 = nn.Conv1d(
            32, 40, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bn12 = nn.BatchNorm1d(num_features=40, eps=1e-3)
        self.dconv0 = nn.ConvTranspose1d(
            40, 32, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bnd0 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        self.dconv01 = nn.Conv1d(
            64, 32, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd01 = nn.BatchNorm1d(num_features=32, eps=1e-3)
        #
        self.dconv1 = nn.ConvTranspose1d(
            32, 22, kernel_size=kernel_size, stride=stride, padding=pad_size
        )
        self.bnd1 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.dconv2 = nn.Conv1d(
            44, 22, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd2 = nn.BatchNorm1d(num_features=22, eps=1e-3)
        self.dconv3 = nn.ConvTranspose1d(
            22, 16, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd3 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.dconv4 = nn.Conv1d(
            32, 16, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd4 = nn.BatchNorm1d(num_features=16, eps=1e-3)
        self.dconv5 = nn.ConvTranspose1d(
            16, 11, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd5 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.dconv6 = nn.Conv1d(
            22, 11, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd6 = nn.BatchNorm1d(num_features=11, eps=1e-3)
        self.dconv7 = nn.ConvTranspose1d(
            11, 8, kernel_size=kernel_size, stride=stride, padding=pad_size - 1
        )
        self.bnd7 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.dconv8 = nn.Conv1d(
            16, 8, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.bnd8 = nn.BatchNorm1d(num_features=8, eps=1e-3)
        self.dconv9 = nn.Conv1d(
            8, classes, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X1 = torch.relu(self.bn1(self.conv1(X)))
        X2 = torch.relu(self.bn2(self.conv2(X1)))
        X3 = torch.relu(self.bn3(self.conv3(X2)))
        X4 = torch.relu(self.bn4(self.conv4(X3)))
        X5 = torch.relu(self.bn5(self.conv5(X4)))
        X6 = torch.relu(self.bn6(self.conv6(X5)))
        X7 = torch.relu(self.bn7(self.conv7(X6)))
        X8 = torch.relu(self.bn8(self.conv8(X7)))
        X9 = torch.relu(self.bn9(self.conv9(X8)))
        X10 = torch.relu(self.bn10(self.conv10(X9)))
        # extra from original UNet
        X10_a = torch.relu(self.bn11(self.conv11(X10)))
        X10_b = torch.relu(self.bn12(self.conv12(X10_a)))
        X10_c = torch.relu(self.bnd0(self.dconv0(X10_b)))
        X10_c = torch.cat(
            (
                X10_c,
                torch.zeros((X10_c.shape[0], X10_c.shape[1], 1), device=X10_c.device),
            ),
            dim=-1,
        )
        X10_c = torch.cat((X10, X10_c), dim=1)
        X10_d = torch.relu(self.bnd01(self.dconv01(X10_c)))
        X11 = torch.relu(self.bnd1(self.dconv1(X10_d)))
        X12 = torch.cat((X11, X8), dim=1)
        X12 = torch.relu(self.bnd2(self.dconv2(X12)))
        X13 = torch.relu(self.bnd3(self.dconv3(X12)))
        X14 = torch.relu(self.bnd4(self.dconv4(torch.cat((X13, X6), dim=1))))
        X15 = torch.relu(self.bnd5(self.dconv5(X14)))
        X15 = torch.cat(
            (X15, torch.zeros((X15.shape[0], X15.shape[1], 1), device=X15.device)),
            dim=2,
        )
        X16 = torch.relu(self.bnd6(self.dconv6(torch.cat((X15, X4), dim=1))))
        X17 = torch.relu(self.bnd7(self.dconv7(X16)))
        X17 = torch.cat(
            (X17, torch.zeros((X17.shape[0], X17.shape[1], 1), device=X17.device)),
            dim=2,
        )
        X18 = torch.relu(self.bnd8(self.dconv8(torch.cat((X17, X2), dim=1))))
        X19 = self.dconv9(X18)

        return self.softmax(X19)


import torch 

class Picker(Skynet):
    def __init__(self):
        super().__init__() 
   
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 30000 
            batchstride = seqlen - 30000 // 2
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            max, maxidx = torch.max(max, dim=1, keepdim=True)
            #max = torch.std(wave, dim=2, keepdim=True)
            wave /= (max + 1e-6)  
            X1 = torch.relu(self.bn1(self.conv1(wave)))
            X2 = torch.relu(self.bn2(self.conv2(X1)))
            X3 = torch.relu(self.bn3(self.conv3(X2)))
            X4 = torch.relu(self.bn4(self.conv4(X3)))
            X5 = torch.relu(self.bn5(self.conv5(X4)))
            X6 = torch.relu(self.bn6(self.conv6(X5)))
            X7 = torch.relu(self.bn7(self.conv7(X6)))
            X8 = torch.relu(self.bn8(self.conv8(X7)))
            X9 = torch.relu(self.bn9(self.conv9(X8)))
            X10 = torch.relu(self.bn10(self.conv10(X9)))
            # extra from original UNet
            X10_a = torch.relu(self.bn11(self.conv11(X10)))
            X10_b = torch.relu(self.bn12(self.conv12(X10_a)))
            X10_c = torch.relu(self.bnd0(self.dconv0(X10_b)))
            X10_c = torch.cat(
                (
                    X10_c,
                    torch.zeros((X10_c.shape[0], X10_c.shape[1], 1), device=X10_c.device),
                ),
                dim=-1,
            )
            X10_c = torch.cat((X10, X10_c), dim=1)
            X10_d = torch.relu(self.bnd01(self.dconv01(X10_c)))
            X11 = torch.relu(self.bnd1(self.dconv1(X10_d)))
            X12 = torch.cat((X11, X8), dim=1)
            X12 = torch.relu(self.bnd2(self.dconv2(X12)))
            X13 = torch.relu(self.bnd3(self.dconv3(X12)))
            X14 = torch.relu(self.bnd4(self.dconv4(torch.cat((X13, X6), dim=1))))
            X15 = torch.relu(self.bnd5(self.dconv5(X14)))
            X15 = torch.cat(
                (X15, torch.zeros((X15.shape[0], X15.shape[1], 1), device=X15.device)),
                dim=2,
            )
            X16 = torch.relu(self.bnd6(self.dconv6(torch.cat((X15, X4), dim=1))))
            X17 = torch.relu(self.bnd7(self.dconv7(X16)))
            X17 = torch.cat(
                (X17, torch.zeros((X17.shape[0], X17.shape[1], 1), device=X17.device)),
                dim=2,
            )
            X18 = torch.relu(self.bnd8(self.dconv8(torch.cat((X17, X2), dim=1))))
            X19 = self.dconv9(X18)

            oc = self.softmax(X19)
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * 1 + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            oc = oc[:, [4, 1, 3, 0, 2]]
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            # 接近非极大值抑制（NMS） 
            # .......P........S...... 
            oc = oc.cpu()
            ot = ot.cpu() 
            for itr in range(4):
                pc = oc[:, itr+1] 
                time_sel = torch.masked_select(ot, pc>0.5)
                score = torch.masked_select(pc, pc>0.5)
                _, order = score.sort(0, descending=True)    # 降序排列
                ntime = time_sel[order] 
                nprob = score[order]
                #print(batchstride, ntime, nprob)
                select = -torch.ones_like(order)
                selidx = torch.arange(0, order.numel(), 1, dtype=torch.long, device=device) 
                count = 0
                while True:
                    if nprob.numel()<1:
                        break 
                    ref = ntime[0]
                    idx = selidx[0]
                    select[idx] = 1 
                    count += 1 
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>1000)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>1000)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>1000)
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
            y = torch.cat(output, dim=0)
        return y 

model = Picker() 
ckpt = torch.load("ckpt/multiphase_skynet.pt", weights_only=False, map_location="cpu")

model.load_state_dict(ckpt)
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/skynet.multiphase.jit")
x = torch.randn([300000, 3])
y = model(x) 