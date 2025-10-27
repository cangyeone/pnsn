import torch 
import torch.nn as nn 
class Conv(nn.Module):
    def __init__(self, nin, nout, ks, stride, bias=False) -> None:
        super().__init__() 
        pad = (ks-1)//2
        self.conv = nn.Conv2d(nin, nout, [ks, 1], [stride, 1], padding=(pad, 0), bias=bias)
        self.bn = nn.BatchNorm2d(nout) 
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x) 
        y = self.bn(x) 
        y = self.relu(y) 
        return y 
class ConvT(nn.Module):
    def __init__(self, nin, nout, ks, stride, bias=False) -> None:
        super().__init__() 
        pad = (ks-1)//2 
        self.conv = nn.ConvTranspose2d(nin, nout, [ks, 1], [stride, 1], padding=(pad, 0), output_padding=(stride-1, 0), bias=bias)
        self.bn = nn.BatchNorm2d(nout) 
        self.relu = nn.ReLU()        
    def forward(self, x):
        x = self.conv(x) 
        y = self.bn(x) 
        y = self.relu(y) 
        return y         

class PhaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.input = Conv(3, 8, 7, 1, True)
        self.output = nn.Conv2d(8, 3, 1, 1)
        self.down = nn.ModuleDict() 
        pre = 8 
        for depth in range(0, 5):
            aft = int(2**(depth) * 8)
            self.down[f"conva_{depth}"] = Conv(pre, aft, 7, 1)
            if depth < 4:
                self.down[f"convb_{depth}"] = Conv(aft, aft, 7, 4)
            pre = aft
        self.up = nn.ModuleDict() 
        pre = 128 
        for depth in range(5-2, -1, -1):
            aft = int(2**(depth) * 8)
            self.up[f"conva_{depth}"] = ConvT(pre, aft, 7, 4)
            #if depth < 4:
            self.up[f"convb_{depth}"] = Conv(aft*2, aft, 7, 1)
            pre = aft 
    def forward(self, x):
        x = x.unsqueeze(3) 
        x = self.input(x) 
        x0a = self.down["conva_0"](x) 
        x0b = self.down["convb_0"](x0a)
        x1a = self.down["conva_1"](x0b) 
        x1b = self.down["convb_1"](x1a)
        x2a = self.down["conva_2"](x1b) 
        x2b = self.down["convb_2"](x2a)
        x3a = self.down["conva_3"](x2b) 
        x3b = self.down["convb_3"](x3a)
        x4a = self.down["conva_4"](x3b) 
        
        y3b = self.up["conva_3"](x4a)
        y3a = self.up["convb_3"](torch.cat([x3a, y3b], dim=1))

        y2b = self.up["conva_2"](y3a)
        y2a = self.up["convb_2"](torch.cat([x2a, y2b], dim=1))

        y1b = self.up["conva_1"](y2a)
        y1a = self.up["convb_1"](torch.cat([x1a, y1b], dim=1))

        y0b = self.up["conva_0"](y1a)
        y0a = self.up["convb_0"](torch.cat([x0a, y0b], dim=1))
        
        y = self.output(y0a) 
        y = y.softmax(dim=1)
        y = y.squeeze(dim=3)
        return y 

class Picker(PhaseNet):
    def __init__(self):
        super().__init__()
        self.n_stride = 1 
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 3072 
            batchstride = seqlen - 1536
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()

            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            #max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            max = torch.std(wave, dim=2, keepdim=True)
            wave /= (max + 1e-6)  
            x = wave.unsqueeze(3)
            x = self.input(x) 
            x0a = self.down["conva_0"](x) 
            x0b = self.down["convb_0"](x0a)
            x1a = self.down["conva_1"](x0b) 
            x1b = self.down["convb_1"](x1a)
            x2a = self.down["conva_2"](x1b) 
            x2b = self.down["convb_2"](x2a)
            x3a = self.down["conva_3"](x2b) 
            x3b = self.down["convb_3"](x3a)
            x4a = self.down["conva_4"](x3b) 
            
            y3b = self.up["conva_3"](x4a)
            y3a = self.up["convb_3"](torch.cat([x3a, y3b], dim=1))

            y2b = self.up["conva_2"](y3a)
            y2a = self.up["convb_2"](torch.cat([x2a, y2b], dim=1))

            y1b = self.up["conva_1"](y2a)
            y1a = self.up["convb_1"](torch.cat([x1a, y1b], dim=1))

            y0b = self.up["conva_0"](y1a)
            y0a = self.up["convb_0"](torch.cat([x0a, y0b], dim=1))
            
            x10 = self.output(y0a) 
            x10 = x10.softmax(dim=1)
            oc = x10.squeeze(dim=3)
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
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
                time_sel = torch.masked_select(ot, pc>0.3)
                score = torch.masked_select(pc, pc>0.3)
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
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>300)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>300)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>300)
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
            y = torch.cat(output, dim=0)
        return y 

model = Picker() 
model.load_state_dict(torch.load("phasenet.pt", map_location="cpu"))
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/phasenet.jit")
x = torch.randn([300000, 3])
y = model(x) 