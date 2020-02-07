
from __future__ import print_function
import numpy as np
import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import pickle
import copy
import os


cwd = os.getcwd()


def model_list():
    vanilla_file = os.path.join(cwd, 'vanilla.txt') 
    mixup_file = os.path.join(cwd, 'mixup.txt') 
    umixup_file =os.path.join(cwd, 'umixup.txt') 


    
    #read list of models(list)
    with open(vanilla_file) as f:                               
        vanilla_models = f.readlines()    
    vanilla_models = [x.strip() for x in vanilla_models] 
    with open(mixup_file) as f:
        mixup_models = f.readlines()    
    mixup_models = [x.strip() for x in mixup_models] 
    with open(umixup_file) as f:
        umixup_models = f.readlines()    
    umixup_models = [x.strip() for x in umixup_models] 

    #pick 50 vanilla to generate adversarial data    
    targets = np.random.choice(100, 50, replace = False).tolist()
    a = np.arange(100)
    adv_generating_idx = [x for x in a if x not in targets]
    
    #pick the 50 models from table to directory
    vanilla_models = np.array(vanilla_models)               
    mixup_models = np.array(mixup_models)
    umixup_models = np.array(umixup_models)
    
    adv_generating_models = vanilla_models[adv_generating_idx].tolist()
    target_vanilla_models = vanilla_models[targets].tolist()
    target_mixup_models = mixup_models[targets].tolist()
    target_umixup_models = umixup_models[targets].tolist()
    return adv_generating_models, target_vanilla_models, target_mixup_models, target_umixup_models


def deepfool(im, net, lambda_fac=3., num_classes=10, overshoot=0.02, max_iter=50, device='cuda'):

    image = copy.deepcopy(im)
    input_shape = image.size()

    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    pert_image = copy.deepcopy(image)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = label
    loop_i = 0

    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r_tot = r_tot + r_i

        pert_image = pert_image + r_i

        check_fool = image + (1 + overshoot) * r_tot
        k_i = torch.argmax(net.forward(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    r_tot = lambda_fac * r_tot
    pert_image = image + r_tot

    return grad, pert_image










 
def Pgd(images, labels, model, criterion, eps=0.3, alpha= 2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    
    ori_images = images.data
    
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
#    torch.cuda.manual_seed(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print('#devices: %d' %(torch.cuda.device_count()))
    print('Using CUDA..')
else:
    device = torch.device("cpu")
    print('cpu')

#Data
dataset = 'cifar10'
num_classes = 10
side_length = 32
num_channels = 3
normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2612))
transform_test = transforms.Compose([   #generate adversial data from training set
        transforms.ToTensor(),
        normalize_transform,
    ])

testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                                   transform=transform_test)
#label embedding
label_dim = 300




#checkpoint = ME_DIR + '/checkpoint_1471264167.torch'
def model_loader(checkpoint):
    # Model
    net = models.__dict__['ResNet18'](num_channels, num_classes, isCosineLoss=False, labelDim = label_dim)  #1ResNet18
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net)


    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=1e-4)      


    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net_state_dicts'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    #torch_rng_state = checkpoint['torch_rng_state']
    torch_rng_state = checkpoint['rng_state']
    torch.set_rng_state(torch_rng_state)
    #numpy_rng_state = checkpoint['numpy_rng_state']
    #np.random.set_state(numpy_rng_state)
    return net


     
     
def testmodel(adv_generating_model, target_vanilla_model, target_mixup_model, target_umixup_model, test_size = 500):
    #test_loss = 0
    adv_generating_model.eval()
    target_vanilla_model.eval()
    target_mixup_model.eval()
    target_umixup_model.eval()
    
    correct_nat = np.zeros(4)
    correct_df = np.zeros(4)
    correct_pgd = np.zeros(4)
    correct_ctm_df = np.zeros(2) #for mixup and umixup
    correct_ctm_pgd = np.zeros(2) #for mixup and umixup
    
    total = 0
    criterion = nn.CrossEntropyLoss()   
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=True, num_workers=8) 
    for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx > test_size: break
            total += targets.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            
            #natural eval
            outputsA = adv_generating_model(inputs)
            outputsB = target_vanilla_model(inputs)
            outputsC = target_mixup_model(inputs)
            outputsD = target_umixup_model(inputs)
            _, predictedA = torch.max(outputsA.data, 1)
            _, predictedB = torch.max(outputsB.data, 1)
            _, predictedC = torch.max(outputsC.data, 1)
            _, predictedD = torch.max(outputsD.data, 1)
            correct_nat[0]  += predictedA.eq(targets.data).cpu().sum()
            correct_nat[1]  += predictedB.eq(targets.data).cpu().sum()
            correct_nat[2]  += predictedC.eq(targets.data).cpu().sum()
            correct_nat[3]  += predictedD.eq(targets.data).cpu().sum()
            
            #generating adv
            _, adv_data_df = deepfool(inputs, adv_generating_model, lambda_fac=3., num_classes=10, overshoot=0.02, max_iter=20)
            
            _, predictedA = torch.max(adv_generating_model(adv_data_df).data, 1)
            _, predictedB = torch.max(target_vanilla_model(adv_data_df).data, 1)
            _, predictedC = torch.max(target_mixup_model(adv_data_df).data, 1)
            _, predictedD = torch.max(target_umixup_model(adv_data_df).data, 1)
            correct_df[0] += predictedA.eq(targets.data).cpu().sum()
            correct_df[1] += predictedB.eq(targets.data).cpu().sum()
            correct_df[2] += predictedC.eq(targets.data).cpu().sum()
            correct_df[3] += predictedD.eq(targets.data).cpu().sum()
            
            adv_data_pgd = Pgd(inputs, targets, adv_generating_model, criterion, eps=0.02, iters = 20)
            _, predictedA = torch.max(adv_generating_model(adv_data_pgd).data, 1)
            _, predictedB = torch.max(target_vanilla_model(adv_data_pgd).data, 1)
            _, predictedC = torch.max(target_mixup_model(adv_data_pgd).data, 1)
            _, predictedD = torch.max(target_umixup_model(adv_data_pgd).data, 1)          
            correct_pgd[0] += predictedA.eq(targets.data).cpu().sum()
            correct_pgd[1] += predictedB.eq(targets.data).cpu().sum()
            correct_pgd[2] += predictedC.eq(targets.data).cpu().sum()
            correct_pgd[3] += predictedD.eq(targets.data).cpu().sum()
            
            #generate adv for mixup and umixup
            _, adv_data_df_mu = deepfool(inputs, target_mixup_model, lambda_fac=3., num_classes=10, overshoot=0.02, max_iter=20)
            _, adv_data_df_umu = deepfool(inputs, target_umixup_model, lambda_fac=3., num_classes=10, overshoot=0.02, max_iter=20)
            _, predictedA = torch.max(target_mixup_model(adv_data_df_mu).data, 1)
            _, predictedB = torch.max(target_umixup_model(adv_data_df_umu).data, 1)
            correct_ctm_df[0] += predictedA.eq(targets.data).cpu().sum()
            correct_ctm_df[1] += predictedB.eq(targets.data).cpu().sum()
            
            
            adv_data_pgd_mu = Pgd(inputs, targets, target_mixup_model, criterion, eps=0.02, iters = 20)
            adv_data_pgd_umu = Pgd(inputs, targets, target_umixup_model, criterion, eps=0.02, iters = 20)
            _, predictedA = torch.max(target_mixup_model(adv_data_pgd_mu).data, 1)
            _, predictedB = torch.max(target_umixup_model(adv_data_pgd_umu).data, 1)
            correct_ctm_pgd[0] += predictedA.eq(targets.data).cpu().sum()
            correct_ctm_pgd[1] += predictedB.eq(targets.data).cpu().sum()

            #test_loss += loss.data.item()
    acc_nat = correct_nat/total    
    acc_df = correct_df/total
    acc_pgd = correct_pgd/total
    acc_ctm_df = correct_ctm_df/total     
    acc_ctm_pgd = correct_ctm_pgd/total  
    print('     gen, van, mix, umix')        
    print('nat: %f, %f, %f, %f' %(acc_nat[0],acc_nat[1],acc_nat[2],acc_nat[3]))    
    print('df : %f, %f, %f, %f' %(acc_df[0],acc_df[1],acc_df[2],acc_df[3])) 
    print('pgd : %f, %f, %f, %f' %(acc_pgd[0],acc_pgd[1],acc_pgd[2],acc_pgd[3])) 
    print('df_ctm : %f, %f' %(acc_ctm_df[0],acc_ctm_df[1])) 
    print('pgd_ctm : %f, %f' %(acc_ctm_pgd[0],acc_ctm_pgd[1])) 
    return acc_nat, acc_df, acc_pgd, acc_ctm_df, acc_ctm_pgd

if __name__ == '__main__':

    generating_model_natural = []
    target_vanilla_natural = []
    target_mixup_natural= []
    target_umixup_natural = []
    generating_model_df= []
    target_vanilla_out_df = []
    target_mixup_out_df= []
    target_umixup_out_df = []
    generating_model_pgd= []
    target_vanilla_out_pgd = []
    target_mixup_out_pgd= []
    target_umixup_out_pgd = []
    target_mixup_self_df = []
    target_umixup_self_df = []
    target_mixup_self_pgd = []
    target_umixup_self_pgd = []
    log = []
    
    
    for i in range(2):
        adv_generating_models, target_vanilla_models, target_mixup_models, target_umixup_models = model_list()   
        
        for exp_idx in range(50):
            print('run: %d' %exp_idx)
            try:
                adv_generating_model = model_loader(adv_generating_models[exp_idx])
                target_vanilla_model = model_loader(target_vanilla_models[exp_idx])
                target_mixup_model = model_loader(target_mixup_models[exp_idx])
                target_umixup_model = model_loader(target_umixup_models[exp_idx])
            except:
                log.append(str(exp_idx) + ' loading fail')
                continue
    
            
            nat, df, pgd, ctm_df, ctm_pgd =testmodel(adv_generating_model, target_vanilla_model, target_mixup_model, target_umixup_model, test_size = 500)
            
            generating_model_natural.append(nat[0])
            target_vanilla_natural.append(nat[1]) 
            target_mixup_natural.append(nat[2])
            target_umixup_natural.append(nat[3]) 
            generating_model_df.append(df[0])
            target_vanilla_out_df.append(df[1]) 
            target_mixup_out_df.append(df[2])
            target_umixup_out_df.append(df[3]) 
            generating_model_pgd.append(pgd[0])
            target_vanilla_out_pgd.append(pgd[1]) 
            target_mixup_out_pgd.append(pgd[2])
            target_umixup_out_pgd.append(pgd[3]) 
            target_mixup_self_df.append(ctm_df[0]) 
            target_umixup_self_df.append(ctm_df[1]) 
            target_mixup_self_pgd.append(ctm_pgd[0]) 
            target_umixup_self_pgd.append(ctm_pgd[1])
            
            del adv_generating_model
            del target_vanilla_model
            del target_mixup_model
            del target_umixup_model
            
    kk =     [generating_model_natural ,  
    target_vanilla_natural ,  
    target_mixup_natural,  
    target_umixup_natural ,  
    generating_model_df,  
    target_vanilla_out_df ,  
    target_mixup_out_df,  
    target_umixup_out_df ,  
    generating_model_pgd,  
    target_vanilla_out_pgd ,  
    target_mixup_out_pgd,  
    target_umixup_out_pgd ,  
    target_mixup_self_df ,  
    target_umixup_self_df ,  
    target_mixup_self_pgd ,  
    target_umixup_self_pgd,
    log]
    
    
    pickle.dump(kk, open(cwd+'save.p', 'wb') )
    


  








