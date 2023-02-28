import matplotlib.pyplot as plt
import numpy as np

X1 = np.arange(0, 65, 1)
loss1 = []
CER1 = []
WER1 = []
val_loss1 = []
val_CER1 = []
val_WER1 = []
with open('logs2.log', 'r') as file:
    
    for line in file:
        target = line.split(",")
        loss1.append(float(target[0]))
        CER1.append(float(target[1]))
        WER1.append(float(target[2]))
        val_loss1.append(float(target[3]))
        val_CER1.append(float(target[4]))
        val_WER1.append(float(target[5]))

#loss and error in subplot
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('CRNN')
ax1.plot(X1, loss1, label = "loss")
ax1.plot(X1, val_loss1, label = "val_loss")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Loss')
ax1.legend()
ax2.plot(X1, CER1, label = "CER")
ax2.plot(X1, val_CER1, label = "val_CER")
ax2.plot(X1, WER1, label = "WER")
ax2.plot(X1, val_WER1, label = "val_WER")
ax2.set_xlabel('Epochs')
ax2.set_title('CER and WER')
ax2.legend()
plt.show()








