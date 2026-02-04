import torch 
import torch.nn as nn 
from models.PhaseNetLight import PhaseNetLight 
class Picker(PhaseNetLight):
    def __init__(self):
        super().__init__()
        self.n_stride = 1 
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 3072 
            batchstride = seqlen - 3072 // 2
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            #x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            max = torch.std(wave, dim=2, keepdim=True)
            wave /= (max + 1e-6)  
            x_in = self.activation(self.in_bn(self.inc(wave)))
            x1 = self.activation(self.bnd1(self.conv1(x_in)))
            x2 = self.activation(self.bnd2(self.conv2(x1)))
            x3 = self.activation(self.bnd3(self.conv3(x2)))
            x4 = self.activation(self.bnd4(self.conv4(x3)))
            
            x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
            x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
            x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
            x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)
            #print(x.shape)
            x = self.out(x)

            oc = self.softmax(x) 
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            oc = oc[:, [2, 0, 1]]
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            # 接近非极大值抑制（NMS） 
            # .......P........S...... 
            oc = oc.cpu()
            ot = ot.cpu() 
            for itr in range(2):
                pc = oc[:, itr+1] 
                time_sel = torch.masked_select(ot, pc>0.2)
                score = torch.masked_select(pc, pc>0.2)
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
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>500)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>500)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>500)
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
            y = torch.cat(output, dim=0)
        return y 

model = Picker() 
ckpt = torch.load("ckpt/9_sc.pt", weights_only=False, map_location="cpu")
state = ckpt.state_dict()
for k in state:
    v = state[k]
    print(k, v.shape)
model.load_state_dict(state)
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/ustcpicker.9_sc.jit")
x = torch.randn([300000, 3])
y = model(x) 