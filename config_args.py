import os


def get_args(parser):
    parser.add_argument('--data_root', type=str, default='/home/amax/Public/data/postate')
    parser.add_argument('--dataset', type=str, choices=['postate158', 'picai'], default='picai')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results')

    parser.add_argument('--lr', type=float, default=0.00001) # 0.00001
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--choose_model', type=str, choices=['UNet', 'OurModel', 'SAM'], default='OurModel')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--inference', action='store_true', default=False)

    parser.add_argument('--name', type=str, default='test')

    args = parser.parse_args()

    if args.dataset == 'postate158':
        args.data_root = '/home/amax/Public/data/prostate158/images_pos'
    elif args.dataset == 'picai':
        args.data_root = '/home/amax/Public/data/picai/images_pos'
    else:
        print('no such dataset')
        exit(0)

    model_name = args.dataset

    model_name += '.'+args.choose_model

    if args.name != '':
        model_name += '.'+args.name
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    model_name = os.path.join(args.results_dir, model_name)
    
    args.model_name = model_name

    if args.inference:
        args.epochs = 1

    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    return args
