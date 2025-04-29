# Train Function of Gaze Estimation (MPIIGAZE) with DMRF filtering
import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from l2cs import L2CS, select_device, Gaze360, Mpiigaze

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "mpiigaze", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=30, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=16, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152,'
                                    ' Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.module.conv1, model.module.bn1, model.module.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.module.fc_yaw_gaze, model.module.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

# Feature Extraction Function
def extract_fc_features(model, input_data):
    x = model.module.conv1(input_data)
    x = model.module.bn1(x)
    x = model.module.relu(x)
    x = model.module.maxpool(x)
    x = model.module.layer1(x)
    x = model.module.layer2(x)
    x = model.module.layer3(x)
    x = model.module.layer4(x)
    x = model.module.avgpool(x)
    features = x.view(x.size(0), -1).detach().cpu().numpy()
    return features

# Data Filtering by OSS Framework
def DMRF_DataFilter(data_train, data_test):
    x = data_train[:, :-1]
    y = data_train[:, -1].reshape(-1, 1)
    x_te = data_test[:, :-1]
    y_te = data_test[:, -1]
    n1, d = x.shape
    n = len(x_te)
    Ft1 = np.ones(n, dtype=int)
    CD = np.zeros(n)

    # 5 subsets prediction
    J = 5
    ys = np.zeros((n, J))
    shuffled_numbers = np.arange(n1)
    split_indices = np.array_split(shuffled_numbers,J)
    for i, part_in in enumerate(split_indices, start=1):
        indice_sub = part_in.tolist()
        x_tr = x[indice_sub]
        y_tr = y[indice_sub]
        gbr = RandomForestRegressor(random_state=42)
        gbr.fit(x_tr, y_tr.ravel())
        y_pred = gbr.predict(x_te)
        ys[:, i-1] = y_pred.flatten()

    for j in range(n):
        samples = ys[j, :]
        kde = gaussian_kde(samples)
        xi = np.linspace(samples.min() - 1, samples.max() + 1, 1000)
        f = kde(xi)
        y_map = xi[np.argmax(f)]
        CD[j] = abs(y_te[j] - y_map) # Label Noise Estimate by MAP

    CDs = np.sort(CD)[::-1]
    nfs = np.arange(1, n + 1)
    h = max(1, n // 2)
    eta = 0.05

    # OSS Filtering Framework
    with np.errstate(divide='ignore', invalid='ignore'):
        epDf = 1 / (1 - np.sqrt((h * (np.log(nfs / h) + 1) - np.log(eta)) / nfs))
        epDf[np.isnan(epDf)] = np.inf
        epD = 1 / (1 - np.sqrt((h * (np.log(n / h) + 1) - np.log(eta)) / n))
    EeD = np.mean(CDs ** 2)
    ErD = np.max(np.mean((ys - y_te[:, None]) ** 2, axis=1))
    C = ErD / EeD if EeD != 0 else 0
    EeDf = np.array([
        np.mean(CDs[:i] ** 2) if i > 0 else 0
        for i in range(1, n + 1)
    ])

    TrhoE = EeDf / EeD if EeD != 0 else 0
    BTrhoE = (epD / epDf) * (1 + C) - C
    objF = (BTrhoE - TrhoE) * epDf

    if np.all(objF == 0):
        nF = n
    else:
        nF = np.argmax(objF) + 1

    Ft1[CD > CDs[nF - 1]] = 0
    return Ft1, CD

# DMRF Main Function
def DMRF_main(data_all):
    tm = [0.0]
    tstart = time.time()

    Ft, _ = DMRF_DataFilter(data_all, data_all) # Warm-up Filtering
    D1 = data_all[Ft == 1]
    D2 = data_all[Ft == 0]
    x = D1[:, :-1]

    n, d = D1.shape
    if n == 0:
        raise ValueError("After the initial filtering, D1 is empty and training cannot continue")

    # Data Recall Process
    # 5 subsets prediction
    J = 5
    ys = np.zeros((len(D2), J))
    ys1 = np.zeros((len(D1), J))
    shuffled_numbers = np.arange(n)
    split_indices = np.array_split(shuffled_numbers, J)
    for i, part_in in enumerate(split_indices, start=1):
        indice_sub = part_in.tolist()
        x_tr = D1[indice_sub, :-1]
        y_tr = D1[indice_sub, -1].reshape(-1, 1)
        x_te = D2[:, :-1]

        gbr = RandomForestRegressor(random_state=42)
        gbr.fit(x_tr, y_tr.ravel())
        ys[:, i-1] = gbr.predict(x_te).flatten()
        ys1[:, i-1] = gbr.predict(x).flatten()

    y_map1 = []
    for j in range(len(D1)):
        samples = ys1[j, :]
        kde = gaussian_kde(samples)
        xi = np.linspace(samples.min(), samples.max(), 1000)
        y_map1.append(xi[np.argmax(kde(xi))])
    errorc = np.mean(abs(D1[:, -1] - y_map1))

    y_map = []
    for j in range(len(D2)):
        samples = ys[j, :]
        kde = gaussian_kde(samples)
        xi = np.linspace(samples.min(), samples.max(), 1000)
        y_map.append(xi[np.argmax(kde(xi))])
    CD = np.abs(D2[:, -1] - y_map)
    D3 = D2[CD <= errorc]

    index = []
    for row in D3:
        mask = np.all(np.isclose(data_all, row, atol=1e-8), axis=1)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            index.append(idx[0])
        else:
            raise ValueError("The sample matching failed. The data might have been modified")

    ft = Ft.copy()
    ft[index] = 1
    D1_3 = data_all[ft == 1]

    Ft1, CD3 = DMRF_DataFilter(D1_3, data_all) # Final Filtering

    tm.append(time.time() - tstart)
    return Ft1, tm, CD3


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    
    
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if data_set=="gaze360":
        model, pre_url = getArch_weights(args.arch, 90)
        if args.snapshot == '':
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        else:
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)
        
        
        model.cuda(gpu)
        dataset=Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
        

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)
       

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                
                iter_gaze += 1

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )
        
          
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )
            

   
    elif data_set=="mpiigaze":
        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        for fold in range(15):
            model, pre_url = getArch_weights(args.arch, 28)
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            model = nn.DataParallel(model)
            model.to(gpu)
            print('Loading data.')
            dataset=Mpiigaze(testlabelpathombined,args.gazeMpiimage_dir, transformations, True, 42, fold)

            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=8,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))
            

            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))

            
            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                {'params': get_ignored_params(model), 'lr': 0},
                {'params': get_non_ignored_params(model), 'lr': args.lr},
                {'params': get_fc_params(model), 'lr': args.lr}
            ], args.lr)

            

            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n" \
                            f" Start training dataset={data_set}, loader={len(train_loader_gaze)},\n " \
                            f"----------------------------fold={fold}----------------------------\n"
            print(configuration)

            # Feature Extraction
            feature_app = np.empty((0, 2048))
            labels_gaze_app = np.empty((0, 2))
            con_labels_gaze_app = np.empty((0, 2))
            index_chunks = np.empty(0)
            with torch.no_grad():
                for i, (images_gaze, labels_gaze, cont_labels_gaze, name, index) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)
                    features = extract_fc_features(model, images_gaze)
                    feature_app = np.append(feature_app, features, axis=0)
                    labels_gaze1 = labels_gaze.cpu().numpy()
                    labels_gaze_app = np.append(labels_gaze_app, labels_gaze1, axis=0)
                    cont_labels_gaze1 = cont_labels_gaze.cpu().numpy()
                    con_labels_gaze_app = np.append(con_labels_gaze_app, cont_labels_gaze1, axis=0)
                    index1 = index.cpu().numpy()
                    index_chunks = np.append(index_chunks, index1)

            # Dimensionality Reduction
            pca = PCA(n_components=128)
            feature_app_pca = pca.fit_transform(feature_app)

            # DMRF Filtering
            tstart = time.time()
            Ft1, _, _ = DMRF_main(np.concatenate((feature_app_pca, con_labels_gaze_app[:, 0:1]), axis=1)) # Pitch Filtering
            Ft2, _, _ = DMRF_main(np.concatenate((feature_app_pca, con_labels_gaze_app[:, 1].reshape(-1, 1)), axis=1)) # Yaw Filtering
            tend = time.time() - tstart
            print(tend)
            Ft = Ft1 | Ft2
            print(f"The ratio of retained data is: {sum(Ft)/len(Ft)}")
            index_filter = []
            for i in range(len(index_chunks)):
                if Ft[i] == 1:
                    index_filter.append(index_chunks[i])

            # Replace the original dataset with the filtered dataset
            filtered_dataset = dataset.get_subset_including_indices(np.array(index_filter))
            train_loader_gaze1 = DataLoader(
                dataset=filtered_dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=8,
                pin_memory=True)

            # Training process
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

                for i, (images_gaze, labels_gaze, cont_labels_gaze, name, index) in enumerate(train_loader_gaze1):
                    images_gaze = Variable(images_gaze).cuda(gpu)

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i + 1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                              'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                  epoch + 1,
                                  num_epochs,
                                  i + 1,
                                  len(dataset) // batch_size,
                                  sum_loss_pitch_gaze / iter_gaze,
                                  sum_loss_yaw_gaze / iter_gaze
                              )
                              )


                

                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    # add
                    if not os.path.exists(output):
                        os.makedirs(output)
                    if not os.path.exists(output + '/fold' + str(fold)):
                        os.makedirs(output + '/fold' + str(fold))
                    # add
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output+'/fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pkl')
                        )
                    
                    

   
