class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 4           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay
    resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    evaluate = False      # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = True  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    project = "cloud"     # project name used during save model
    dataset = "/data/zcj/dataset/detection/cloud/dataset/train/"  # dataset folder with train and val
    test_folder =  "/data/zcj/dataset/detection/cloud/dataset/test/"      # test images' folder
    checkpoints = "./checkpoints/%s/"%project        # path to save checkpoints
    log_dir = "./logs/%s/"%project                   # path to save log files
    pred_mask = "./pred_mask/"            # path to save pred test mask
    bs = 32               # batch size
    lr = 1e-4             # learning rate
    epochs = 40           # train epochs
    input_size = 384      # model input size or image resied
    num_classes = 2       # num of classes
    gpu_id = "0"          # default gpu id
    encoder = "resnet50"      # encoder model name
    encoder_weights = "imagenet"
    optim = "adam"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    loss_func = "BCEWithLogitsLoss" # see ./losses/README.md
    use_strong_aug = False  # use strong aug for common dataset
    lr_scheduler = "step" # "on_iou","on_dice","on_loss","cosine"
    activation = None 
    warmup = True          # use warmup for lr
    warmup_factor = 10
    warmup_epo = 1
configs = DefaultConfigs()
