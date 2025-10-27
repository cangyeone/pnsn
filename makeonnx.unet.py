from models.UNet import PhaseNetLight 
import torch 

class Picker(PhaseNetLight):
    def __init__(self):
        super().__init__() 
   
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
            x = x.to(device)
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
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * 1 + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            oc = oc[:, [2, 0, 1]]
            ot = tgrid.squeeze()
            ot = ot.reshape(-1) 
        return oc, ot   
model = Picker() 
model.eval()
ckpt = torch.load("model_list/9_sc.pt", weights_only=False, map_location="cpu")
state = ckpt.state_dict()
model.load_state_dict(state)
input_names = ["wave"]
output_names = ["prob", "time"]
#x = torch.randn([10, 3, 6144, 1])
x = torch.randn([500000, 3])
torch.onnx.export(model, x, 
"pickers/9_sc.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch"}, "prob":{0:"batch"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=11)