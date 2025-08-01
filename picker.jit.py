import numpy as np
import torch 
import obspy # pip install obspy 

mname = "pickers/pnsn.jit" # 
device = torch.device("cpu") #infer device 
sess = torch.jit.load(mname)
sess.eval() # inference mode 
sess.to(device)

# read data 
st1 = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")
st2 = obspy.read("data/waveform/X1.53085.01.BHN.D.20122080726235953.sac")
st3 = obspy.read("data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac")
data = [st1[0].data, st2[0].data, st3[0].data] 

x = np.stack(data, axis=1).astype(np.float32) #[N, 3]->一天 [8640000]100Hz 
with torch.no_grad():
    x = torch.tensor(x, dtype=torch.float32, device=device) 
    y = sess(x) 
    phase = y.cpu().numpy()# 后处理在文件中
import matplotlib.pyplot as plt 
plt.plot(x[:, 2], alpha=0.5) 
for pha in phase:
    if pha[0]==0:#Pg picking result 
        c = "r" 
    elif pha[0]==1:#Sg
        c = "b"
    elif pha[0]==2:#Pn 
        c = "g"
    else:          # Sn 
        c = "k"
    plt.axvline(pha[1], c=c)
plt.show()
