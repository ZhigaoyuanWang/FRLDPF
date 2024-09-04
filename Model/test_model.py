import torch
def eval_epoch(model,dataset,l,r,criterion,device):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for i in range(l,r):
            x,t,y = dataset[i]
            output = model(x.to(device),t.to(device))
            loss = criterion(output,y.to(device))            
            eval_loss += (loss.detach().cpu().item())
            n += x.shape[0]
            
    return eval_loss/(r-l)