import matplotlib.pyplot as plt

def plot_pair(dat,target):
     fig = plt.figure()
     plt.gray()
     fig.add_subplot(1,2,1)
     plt.imshow(dat[0,0,:,:])
     fig.add_subplot(1,2,2)
     plt.imshow(target[0,0,:,:])
     plt.show(block=False)

def print_stats(loss, epoch, i, running_loss=0.0):
     running_loss += loss.item()
     
     if i % 5 == 4:    # print every 2000 mini-batches
          print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 5))
          running_loss = 0.0


