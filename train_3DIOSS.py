
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_Dataset_IOSS import *
from comparedmodel.THISNet import *
from losses_and_metrics_for_mesh import *
import pandas as pd
import time
import random
from tqdm import tqdm  # 导入 tqdm 库

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cross_entropy(logits, y):
    s = torch.exp(logits)
    logits = s / torch.sum(s, dim=1, keepdim=True)
    c = -(y * torch.log(logits)).sum(dim=-1)
    return torch.mean(c)

# 保存模型检查点和当前训练状态
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# 加载上次训练的检查点
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      

    epoch = checkpoint['epoch']-1
    return epoch


start = time.time()


if __name__ == '__main__':

    setup_seed(1111)

    ncells = 10000
    tooth_type = 'upper'
    kfold_nums = 5
    use_fold = 1

    train_list_path = '/home/root111/wu_code/DBGANet-main/3DIossing_data/train_upper_list.csv'
    test_list_path = '/home/root111/wu_code/DBGANet-main/3DIossing_data/test_upper_list.csv'
    labels_dir = "/home/root111/wu_code/DBGANet-main/3DIossing_data/output"

    model_path = './model/3DIOSS/DBGANet'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    segnet_name = 'DBGANet'
    num_classes = 15  # 国际编号中上牙和下牙类别数为15
    num_neighbor = 32  # k的值
    num_epochs = 200
    num_workers = 32
    train_batch_size = 6
    val_batch_size = 1
    num_batches_to_print = 500
    lr = 0.001

    save_model_threshold = 0.92

    start = time.time()

    model_name_path = '{}_{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr'
    model_name_path = model_name_path.format(segnet_name, num_classes, num_neighbor, num_epochs, ncells,
                                             train_batch_size, kfold_nums, use_fold, lr)

    model_path = os.path.join(model_path, model_name_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    checkpoint_name = 'latest_checkpoint.tar'
    train_loss_csv_file = 'losses_{}_{}classes_{}k_{}epochs_{}ncells_{}batchsize_{}kfold{}_{}lr.csv'
    train_loss_csv_file = train_loss_csv_file.format(segnet_name, num_classes, num_neighbor, num_epochs, ncells,
                                                     train_batch_size, kfold_nums, use_fold, lr)

    # set dataset

    training_dataset = Mesh_Dataset(data_list_path=train_list_path, labels_dir=labels_dir, num_classes=num_classes,
                                    patch_size=ncells)
    test_dataset = Mesh_Dataset(data_list_path=test_list_path, labels_dir=labels_dir, num_classes=num_classes,
                                patch_size=ncells)

    train_loader = DataLoader(dataset=training_dataset, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    model = My_Seg(num_classes=num_classes, num_neighbor=num_neighbor)

    # 加载模型，重写一下，加载一下之前的模型
    # path = "./model/test1/DBGANet_15classes_32k_100epochs_10000ncells_1batchsize_5kfold1_0.001lr/latest_checkpoint.tar"
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    print(torch.cuda.device_count())

    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    # 先检查是否有已保存的模型和训练状态
    checkpoint_path = 'model/3DIOSS/DBGANet/DBGANet_15classes_32k_100epochs_20000ncells_6batchsize_5kfold1_0.001lr/latest_checkpoint.tar'
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}...')
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    else:
        print(f'No checkpoint found at {checkpoint_path}, training from scratch...')
        start_epoch = 0
    losses, mdsc, msen, mppv = [], [], [], []
    test_losses, test_mdsc, test_msen, test_mppv = [], [], [], []

    best_test_dsc = 0.0

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    # for epoch in range(start_epoch,num_epochs):
    for epoch in range(start_epoch, num_epochs):
        # training
        model.train()

        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # 用 tqdm 包裹 train_loader 循环
        for i_batch, batched_sample in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            centroids = batched_sample['barycenter'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
            labels1 = batched_sample['barycenter_label'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs, cents, classes, weight = model(inputs)


            labels = labels.view(-1, 1)[:, 0]
            outputs1 = outputs.contiguous().view(-1, 15)


            seg_loss = F.nll_loss(outputs1, labels)
            # seg_loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            cent_loss = torch.nn.functional.smooth_l1_loss(cents, centroids)
            # class_loss = Generalized_Dice_Loss(classes, one_hot_labels, class_weights)

            class_loss = cross_entropy(classes, labels1[:, 0, :])
            loss = seg_loss + weight[0].item() * (cent_loss + class_loss * weight[1].item())
            # print(f'loss: {loss}', f'seg_loss:{seg_loss}', f'cent_loss:{cent_loss}', f'class_loss:{class_loss}')


            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            if i_batch % num_batches_to_print == num_batches_to_print - 1:
                print(f'[Epoch: {epoch + 1}/{num_epochs}, Batch: {i_batch + 1}/{len(train_loader)}] '
                      f'loss: {running_loss / num_batches_to_print}, dsc: {running_mdsc / num_batches_to_print}, '
                      f'sen: {running_msen / num_batches_to_print}, ppv: {running_mppv / num_batches_to_print}')

                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0

        losses.append(loss_epoch / len(train_loader))
        mdsc.append(mdsc_epoch / len(train_loader))
        msen.append(msen_epoch / len(train_loader))
        mppv.append(mppv_epoch / len(train_loader))
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # scheduler.step()
        # inputs = torch.tensor([])
        # outputs = torch.tensor([])
        # outputs1 = torch.tensor([])
        # labels = torch.tensor([])

        # 测试
        model.eval()
        with torch.no_grad():
            running_test_loss = 0.0
            running_test_mdsc = 0.0
            running_test_msen = 0.0
            running_test_mppv = 0.0
            test_loss_epoch = 0.0
            test_mdsc_epoch = 0.0
            test_msen_epoch = 0.0
            test_mppv_epoch = 0.0

            # 用 tqdm 包裹 test_loader 循环
            for i_batch, batched_test_sample in enumerate(tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{num_epochs}")):
                inputs = batched_test_sample['cells'].to(device, dtype=torch.float)
                labels = batched_test_sample['labels'].to(device, dtype=torch.long)
                centroids = batched_test_sample['barycenter'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
                labels1 = batched_test_sample['barycenter_label'].to(device, dtype=torch.long)



                outputs, cents, classes, weight = model(inputs)

                labels = labels.view(-1, 1)[:, 0]
                outputs1 = outputs.contiguous().view(-1, 15)

                # loss = F.nll_loss(outputs1, labels)

                seg_loss = F.nll_loss(outputs1, labels)
                cent_loss = torch.nn.functional.smooth_l1_loss(cents, centroids)

                class_loss = cross_entropy(classes, labels1[:, 0, :])
                loss = seg_loss + weight[0].item() * (cent_loss + class_loss * weight[1].item())
                # print(f'loss: {loss}', f'seg_loss:{seg_loss}', f'cent_loss:{cent_loss}', f'class_loss:{class_loss}')

                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_test_loss += loss.item()
                running_test_mdsc += dsc.item()
                running_test_msen += sen.item()
                running_test_mppv += ppv.item()
                test_loss_epoch += loss.item()
                test_mdsc_epoch += dsc.item()
                test_msen_epoch += sen.item()
                test_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print - 1:
                    print(f'[Epoch: {epoch + 1}/{num_epochs}, test batch: {i_batch + 1}/{len(test_loader)}] '
                          f'test_loss: {running_test_loss / num_batches_to_print}, test_dsc: {running_test_mdsc / num_batches_to_print}, '
                          f'test_sen: {running_test_msen / num_batches_to_print}, test_ppv: {running_test_mppv / num_batches_to_print}')

                    running_test_loss = 0.0
                    running_test_mdsc = 0.0
                    running_test_msen = 0.0
                    running_test_mppv = 0.0

            test_losses.append(test_loss_epoch / len(test_loader))
            test_mdsc.append(test_mdsc_epoch / len(test_loader))
            test_msen.append(test_msen_epoch / len(test_loader))
            test_mppv.append(test_mppv_epoch / len(test_loader))
            test_loss_epoch = 0.0
            test_mdsc_epoch = 0.0
            test_msen_epoch = 0.0
            test_mppv_epoch = 0.0

            print(
                f'*****\nEpoch: {epoch + 1}/{num_epochs}, loss: {losses[-1]}, dsc: {mdsc[-1]}, sen: {msen[-1]}, ppv: {mppv[-1]}\n'
                f'test_loss: {test_losses[-1]}, test_dsc: {test_mdsc[-1]}, test_sen: {test_msen[-1]}, test_ppv: {test_mppv[-1]}\n*****')

        # 保存模型检查点和统计信息
        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'test_losses': test_losses,
                    'test_mdsc': test_mdsc,
                    'test_msen': test_msen,
                    'test_mppv': test_mppv},
                   os.path.join(model_path, checkpoint_name))

        if test_mdsc[-1] >= save_model_threshold:
            best_test_dsc = test_mdsc[-1]
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'test_losses': test_losses,
                        'test_mdsc': test_mdsc,
                        'test_msen': test_msen,
                        'test_mppv': test_mppv},
                       os.path.join(model_path, format(best_test_dsc, '.4f') + "_" + str(epoch) + '.tar'))

        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'test_loss': test_losses,
                   'test_DSC': test_mdsc, 'test_SEN': test_msen, 'test_PPV': test_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(os.path.join(model_path, train_loss_csv_file))

        end = time.time()
        running_time = end - start
        print(f'Time cost: {running_time:.5f}s, {running_time / 3600:.2f} h\n')

    print(max(test_mdsc), weight[0].item(), weight[1].item())