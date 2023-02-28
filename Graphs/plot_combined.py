import matplotlib.pyplot as plt
import numpy as np

X1 = np.arange(0, 56, 1)
loss1 = []
CER1 = []
WER1 = []
val_loss1 = []
val_CER1 = []
val_WER1 = []
with open('logs1.log', 'r') as file:
    
    for line in file:
        target = line.split(",")
        loss1.append(float(target[0]))
        CER1.append(float(target[1]))
        WER1.append(float(target[2]))
        val_loss1.append(float(target[3]))
        val_CER1.append(float(target[4]))
        val_WER1.append(float(target[5]))


X2 = np.arange(0, 65, 1)
loss2 = []
CER2 = []
WER2 = []
val_loss2 = []
val_CER2 = []
val_WER2 = []
with open('logs2.log', 'r') as file:
        
        for line in file:
            target = line.split(",")
            loss2.append(float(target[0]))
            CER2.append(float(target[1]))
            WER2.append(float(target[2]))
            val_loss2.append(float(target[3]))
            val_CER2.append(float(target[4]))
            val_WER2.append(float(target[5]))

#comparing loss
plt.plot(X1, loss1, label = "loss(ResNet)")
plt.plot(X1, val_loss1, label = "val_loss(ResNet)")
plt.plot(X2, loss2, label = "loss(CRNN)")
plt.plot(X2, val_loss2, label = "val_loss(CRNN)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.show()

#comparing CER
plt.plot(X1, CER1, label = "CER(ResNet)")
plt.plot(X1, val_CER1, label = "val_CER(ResNet)")
plt.plot(X2, CER2, label = "CER(CRNN)")
plt.plot(X2, val_CER2, label = "val_CER(CRNN)")
plt.xlabel('Epochs')
plt.ylabel('CER')
plt.title('CER')
plt.legend()
plt.show()

#comparing WER
plt.plot(X1, WER1, label = "WER(ResNet)")
plt.plot(X1, val_WER1, label = "val_WER(ResNet)")
plt.plot(X2, WER2, label = "WER(CRNN)")
plt.plot(X2, val_WER2, label = "val_WER(CRNN)")
plt.xlabel('Epochs')
plt.ylabel('WER')
plt.title('WER')
plt.legend()
plt.show()





            
