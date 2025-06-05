import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model


def labels_to_onehot(label, n_class=7, size=32, device='cpu'):
    """
    Converts a batch of class‐indices (shape [B]) into a one‐hot label map
    of shape [B, n_class, size, size].  Here n_class=7 (FedGAN was trained on 7 classes).
    Any “extra” channels beyond the true label remain zero.
    """
    B = label.size(0)
    onehot = torch.zeros(B, n_class, size, size, device=device)
    idx = label.view(B, 1, 1, 1).expand(B, 1, size, size)
    onehot.scatter_(1, idx, 1.0)
    return onehot


def main():
    ##################################################################
    # 1) Parse training‐time options so that things like lr, beta1, etc. exist
    ##################################################################
    opt = TrainOptions().parse()

    ##################################################################
    # 2) FORCE the fedGAN checkpoint‐directory and number of classes=7
    ##################################################################
    
    opt.n_class = 7               

    ##################################################################
    # 3) We still want netD to be built (so isTrain=True), but we will run in eval mode
    ##################################################################
    opt.phase   = 'test'   
    opt.isTrain = True     
    opt.eval    = True     

    ##################################################################
    # 4) Override any unwanted data‐augmentation / threading parameters
    ##################################################################
    opt.num_threads    = 0
    opt.batch_size     = 1
    opt.serial_batches = True
    opt.no_flip        = True
    opt.preprocess     = 'none'

    ##################################################################
    # 5) Build the model (netG + netD) and load FedGAN weights
    ##################################################################
    model = create_model(opt)
    model.setup(opt)  

    if opt.epoch != 'latest':
        print(f'Loading weights from epoch {opt.epoch}')
    else:
        print('Loading latest weights')

    ##################################################################
    # 6) Create “member” (train‐split) and “non‐member” (test‐split) datasets
    ##################################################################
    print('Loading member dataset (training data)')
    opt.phase = 'train'   
    member_dataset = create_dataset(opt)

    print('Loading non-member dataset (test data)')
    opt.phase = 'test'    
    non_member_dataset = create_dataset(opt)

    ##################################################################
    # 7) Switch D into evaluation mode and inspect its expected input‐channels
    ##################################################################
    device = model.device

    if isinstance(model.netD, list):
        for d in model.netD:
            d.eval()
        print("FedGAN: switched all clientDs to eval mode.")
        # Inspect only the first one’s conv_label channels
        print("netD[0].conv_label expects", model.netD[0].module.conv_label.in_channels, "channels")
    else:
        model.netD.eval()
        print("netD.conv_label expects", model.netD.module.conv_label.in_channels, "channels")

    ##################################################################
    # 8) Gather all labels so we can report “inferred number of classes”
    ##################################################################
    all_labels = []
    for d in member_dataset:
        all_labels.append(int(d['label'].item()))
    for d in non_member_dataset:
        all_labels.append(int(d['label'].item()))

    unique_labels    = set(all_labels)
    num_classes_data = max(unique_labels) + 1
    print(f"Inferred number of classes in data: {num_classes_data} (0..{num_classes_data-1}).")

    ##################################################################
    # 9) Compute “member” scores
    ##################################################################
    member_scores = []
    print('\nProcessing member samples...')
    for i, data in enumerate(member_dataset):
        img   = data['A'].to(device)       # [1,3,32,32]
        label = data['label'].to(device)   # [1]

        with torch.no_grad():
            # Build a one‐hot label map [1,7,32,32]
            label_img = labels_to_onehot(
                label, n_class=opt.n_class, size=opt.load_size, device=device
            )

            if isinstance(model.netD, list):
                scores_i = []
                for d in model.netD:
                    out_i = d(img, label_img)      # [1,1] logit
                    scores_i.append(torch.sigmoid(out_i).mean())
                score = torch.stack(scores_i).mean().item()
            else:
                out   = model.netD(img, label_img)
                score = torch.sigmoid(out).mean().item()

        member_scores.append(score)
        if (i + 1) % 100 == 0:
            print(f'  Processed {i+1}/{len(member_dataset)} member samples')

    ##################################################################
    # 10) Compute “non‐member” scores
    ##################################################################
    non_member_scores = []
    print('\nProcessing non-member samples...')
    for i, data in enumerate(non_member_dataset):
        img   = data['A'].to(device)
        label = data['label'].to(device)

        with torch.no_grad():
            label_img = labels_to_onehot(
                label, n_class=opt.n_class, size=opt.load_size, device=device
            )

            if isinstance(model.netD, list):
                scores_i = []
                for d in model.netD:
                    out_i = d(img, label_img)
                    scores_i.append(torch.sigmoid(out_i).mean())
                score = torch.stack(scores_i).mean().item()
            else:
                out   = model.netD(img, label_img)
                score = torch.sigmoid(out).mean().item()

        non_member_scores.append(score)
        if (i + 1) % 100 == 0:
            print(f'  Processed {i+1}/{len(non_member_dataset)} non-member samples')

    ##################################################################
    # 11) Build y_true / y_scores and compute AUC, ROC, threshold, accuracy
    ##################################################################
    y_true   = np.array([1] * len(member_scores) + [0] * len(non_member_scores))
    y_scores = np.array(member_scores + non_member_scores)

    auc_value  = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores      = tpr - fpr
    opt_idx       = np.argmax(j_scores)
    opt_threshold = thresholds[opt_idx]
    accuracy      = np.mean((y_scores >= opt_threshold) == y_true)

    ##################################################################
    # 12) Print final Membership Inference Attack Results
    ##################################################################
    print('\n' + '=' * 50)
    print('Membership Inference Attack Results')
    print('=' * 50)
    print(f'Member samples:     {len(member_scores)}')
    print(f'Non-member samples: {len(non_member_scores)}')
    print(f'AUC:                {auc_value:.4f}')
    print(f'Accuracy:           {accuracy:.4f}')
    print(f'Optimal Threshold:  {opt_threshold:.4f}')
    print(f'TPR:                {tpr[opt_idx]:.4f}')
    print(f'FPR:                {fpr[opt_idx]:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
