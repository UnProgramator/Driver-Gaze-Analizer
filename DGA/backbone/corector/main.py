import CorectionNetwork1 as cn1
from CorectionUtilities import *
from torch import nn, optim
import matplotlib.pyplot as plt
import math

from losfucts import CustomLoss, CustomLoss_v

infile1:str = "C:/Users/apesc/Downloads/pitch_train_data.csv"

infile:str = "C:/Users/apesc/Downloads/Err_gaze.csv"

print_train = False

epocs = 8000  # training epocs
err_ok = 0.05 # estimated pitch - maximum admited error

def train(epocs, model, data, gt, criterion = nn.MSELoss()):
    # NN loss and optimizer
    #criterion = nn.MSELoss()  # Mean Squared Error Loss
    #criterion = CustomLoss(err_ok)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # NN training
    for epoch in range(epocs):
        model.train()           
        optimizer.zero_grad()         # gradients reset to 0
        outputs = model(data)      
        loss = criterion(outputs, gt)  # loss computation
        loss.backward()               # backpropagation
        optimizer.step()              # weights adjustement
    
        if print_train and (epoch + 1) % (epocs/10) == 0:
            print(f"Epoch {epoch+1}/{epocs}, Loss: {loss.item():.4f}") # display loss during training

    return model


def validate(model, invals, gtv, err_ok):
    # Model testing (Problem: only one dataset {X,y}, also for evaluation)
      
    predictions = model(invals)
    test_loss, correct = 0, 0
    initial_loss, inital_correct = 0, 0
    loss_fn = torch.nn.MSELoss(reduction='sum')
    size = invals.size()[0]
    with torch.no_grad():  
      for i in range(size):
        Xc = invals.tolist()[i]
        yc= gtv.tolist()[i]
        pred = predictions[i]
        # print(str(type(Xc)) + ' ' + str(type(yc)) + ' ' + str(type(pred)))
        test_loss += loss_fn(pred, gtv[i]).item()
        correct += int(abs(pred.item() - yc[0]) <= err_ok)

        initial_pred = torch.tensor([Xc[-1]], requires_grad=True)

        initial_loss += loss_fn(initial_pred, gtv[i]).item()
        inital_correct += int(abs(initial_pred.item() - yc[0]) <= err_ok)

    print(f"Test Error: \n Accuracy(diff < {err_ok:>0.3f}): {(100*correct/size):>0.1f}%, Avg loss: {math.sqrt(test_loss)/100:>8f} \n")
    print(f"Initial Error: \n Accuracy(diff < {err_ok:>0.3f}): {(100*inital_correct/size):>0.1f}%, Avg loss: {math.sqrt(initial_loss)/100:>8f} \n")

    #print([r[4] for r in df['OPWnd'].tolist()])
    vdf =  pd.DataFrame({'GT':gtv.squeeze().tolist(), 'l2cs_net':[r[-1] for r in invals.squeeze().tolist()],'Err_corr':predictions.squeeze().tolist()})
    graph = vdf.plot(title='Err correction')
    plt.show()

def test1(drv:int = 3):
    #inp,gt = readCSV_gt_evaled(infile1)

    inp, gt, intt, gttt = readCSV_gt_evaled_loo_drivface(infile, 5, drv)

    # print(inp)
    # print(gt)
    # print(intt)
    # print(gttt)

    #model:nn.Module = cn1.default() # 1,2,3

    #model:nn.Module = cn1.CNN1([5, 15]) # 5, 6

    #model:nn.Module = cn1.CNN1([5, 5]) # 7, 8

    model:nn.Module = cn1.CNN1([5, 120, 10]) # 9

    #model:nn.Module = cn1.CNN1([5, 30]) # 11

    

    #model = train(10000, model, inp, gt)    
    model = train(10000, model, inp, gt, CustomLoss(err_ok))    
    #model = train(20000, model ,inp, gt)
    #model = train(16000, model ,inp, gt)

    model.eval()

    print("pe datele de antrenare")

    validate(model, inp, gt, err_ok)

    print("pe datele de validare")

    validate(model, intt, gttt, err_ok)


test1(0)
  
# for i in range(4):
#     print(f'i = {i}')
#     test1(i)