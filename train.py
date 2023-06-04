import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch.utils
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
from modelSelector import train_model
# from Loss.Diceloss import * #ToDO: We can use Various losses to evaluate and tune our model
import torch.nn as nn
from Dataset.hubMapDataset import HubMapDataset as Dataset
import time
import Transforms as transforms

def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    setpu_seed(args.seed)

    model = train_model(args.model) #Select model
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    log_file = args.model+str(formatted_time)+".log"
    log_file = os.path.join(args.weights, log_file)#args.weights is the 'position' to store training results and logs
    logger = open(log_file, "w")
    model_name = args.model+str(formatted_time)
    model.to(device)
    # Log some initial information
    logger.write("Model: {}\n".format(str(model)))
    logger.write("Start time: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    if args.additional_info:
        print(args.additional_info)
        logger.write("Additional info: {}\n".format(args.additional_info))
    logger.flush()

    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = Dataset(split='Training',dataset_path=args.images,transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size , shuffle=True,num_workers=4)
    test_set = Dataset(split='Testing',dataset_path=args.images,transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    criterion = nn.BCELoss()
    best_validation_dsc = 0.0
    lr=args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_valid = []
    threshold = 0 #10个epoch没有更新就中止
    total_time = 0
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        start_time = time.time()
        for phase in ["train", "valid"]:

            if phase == "train":
                model.train()
                for batch_index,(input,mask) in enumerate(tqdm(train_loader)):
                    input = input.to(device)
                    mask = mask.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(input)
                        loss = criterion(y_pred,mask)
                        loss.backward()
                        optimizer.step()

            if phase == "valid":
                model.eval()
                for batch_index, (input, mask) in enumerate(tqdm(test_loader)):
                    input = input.to(device)
                    mask = mask.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(input)
                        dice = 1-criterion(y_pred, mask)#TODO: Change names to 'loss' rather than 'Dice'
                        loss_valid.append(dice.item())
                mean_dsc = np.mean(loss_valid)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(model.state_dict(), os.path.join(args.weights, model_name+".pt"))
                    best_epoch = epoch + 1
                    print("Current Best validation mean ACC: {:4f}\n".format(best_validation_dsc))
                    threshold = 0 #重新计数
                    #加个阶段
                else:
                    threshold+=1
                print("Current validation mean ACC: {:4f}\n".format(best_validation_dsc))
                loss_valid = []
        # Log some timing information
        epoch_time = time.time() - start_time
        total_time += epoch_time
        if threshold >=5:#TODO
           for group in optimizer.param_groups:
               if lr>=0.0005:
                    lr = lr/2
                    group['lr'] = lr
        if threshold ==20:
            break
    print("Final Best validation mean DICE: {:4f}".format(best_validation_dsc))
    # Log some final information
    logger.write("Training finished in {:.2f} seconds.\n".format(total_time))
    logger.write("Best validation dice: {:.4f} (epoch {})\n".format(best_validation_dsc, best_epoch))
    logger.flush()
    # Close the logger
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment instances of microvascular structures")
    parser.add_argument("--model",type=str,default="test",help="modelname")
    parser.add_argument("--batch_size",type=int,default=16,help="input batch size for training (default: 16)")
    parser.add_argument("--epochs",type=int,default=200,help="number of epochs to train (default: 200)")
    parser.add_argument("--lr",type=float,default=0.002,help="initial learning rate (default: 0.001)")
    parser.add_argument("--workers",type=int,default=2,help="number of workers for data loading (default: 4)")
    parser.add_argument("--weights", type=str, default="./Weight", help="folder to save weights")
    parser.add_argument( "--images", type=str, default="./archive/hubmap-hacking-the-human-vasculature", help="root folder with images")
    parser.add_argument('--cuda', default=True, type=bool,help='Use GPU calculating')
    parser.add_argument('--seed', default=2021,type=int,help="random seed to make sure get same result each time")
    parser.add_argument('--additional_info', type=str, help='Extra info you want to write in logs',
                        default="Test,随机种子是2021。数据集：hubMap，batch:16")
    args = parser.parse_args()
    main(args)