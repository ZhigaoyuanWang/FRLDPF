import time

def train_epoch(model,dataset,optimizer,l,r,criterion,device):
    model.train()
    train_loss = 0
    n = 0
    time_now = time.time()
    for i in range(l,r):
        x,t, y = dataset[i]
        optimizer.zero_grad()
        output = model(x.to(device),t.to(device))
        loss = criterion(output,y.to(device)) # not missing data
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item())
    print("Epoch: {} cost time: {}".format(i + 1, time.time() - time_now))
    time_now = time.time()
    return train_loss