import CorectionNetwork1 as cn1
from CorectionUtilities import *
from torch import nn, optim
import matplotlib.pyplot as plt
import math

infile:str = "C:/Users/apesc/Downloads/pitch_train_data.csv"

def train(epocs, model, data, gt):
    # NN loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # NN training
    for epoch in range(epocs):
        model.train()           
        optimizer.zero_grad()         # gradients reset to 0
        outputs = model(data)      
        loss = criterion(outputs, gt)  # loss computation
        loss.backward()               # backpropagation
        optimizer.step()              # weights adjustement
    
        if (epoch + 1) % (epocs/10) == 0:
            print(f"Epoch {epoch+1}/{epocs}, Loss: {loss.item():.4f}") # display loss during training

    return model


def test1():
    inp,gt = readCSV_gt_evaled(infile)

    #model:nn.Module = cn1.default() # 1,2,3

    #model:nn.Module = cn1.CNN1([5, 15]) # 5, 6

    model:nn.Module = cn1.CNN1([5, 5]) # 7, 8

    #model:nn.Module = cn1.CNN1([5, 10, 10]) # 9

    #model:nn.Module = cn1.CNN1([5, 30]) # 11

    epocs = 8000  # training epocs
    err_ok = 0.05 # estimated pitch - maximum admited error

    model = train(1000, model, inp, gt)    
    #model = train(20000, model ,X, y)
    #model = train(16000, model ,X, y)


    # Model testing (Problem: only one dataset {X,y}, also for evaluation)
    model.eval()  
    predictions = model(inp)
    test_loss, correct = 0, 0
    initial_loss, inital_correct = 0, 0
    loss_fn = torch.nn.MSELoss(reduction='sum')
    size = inp.size()[0]
    with torch.no_grad():  
      for i in range(size):
        Xc = inp.tolist()[i]
        yc= gt.tolist()[i]
        pred = predictions[i]
        print(str(type(Xc)) + ' ' + str(type(yc)) + ' ' + str(type(pred)))
        test_loss += loss_fn(pred, gt[i]).item()
        correct += int(abs(pred.item() - yc[0]) <= err_ok)

        initial_pred = torch.tensor([Xc[-1]], requires_grad=True)

        initial_loss += loss_fn(initial_pred, gt[i]).item()
        inital_correct += int(abs(initial_pred.item() - yc[0]) <= err_ok)

    print(f"Test Error: \n Accuracy(diff < {err_ok:>0.3f}): {(100*correct/size):>0.1f}%, Avg loss: {math.sqrt(test_loss)/100:>8f} \n")
    print(f"Initial Error: \n Accuracy(diff < {err_ok:>0.3f}): {(100*inital_correct/size):>0.1f}%, Avg loss: {math.sqrt(initial_loss)/100:>8f} \n")

    #print([r[4] for r in df['OPWnd'].tolist()])
    vdf =  pd.DataFrame({'GT':gt.squeeze().tolist(), 'l2cs_net':[r[-1] for r in inp.squeeze().tolist()],'Err_corr':predictions.squeeze().tolist()})
    graph = vdf.plot(title='Err correction')
    plt.show()

test1()