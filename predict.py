import torch
from torchvision import datasets, transforms, models
import PIL
import torch.nn.functional as nnf
from torch import nn
import pandas as pd, argparse, json
import matplotlib.pyplot as plt
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--data_directory', default='flowers/')
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--arch', default='vgg16', help='HINT: [vgg16 or alexnet]')
    args = parser.parse_args()

    predictor = DL_Predictor(args)
    print(predictor.predict())


class DL_Predictor:
    def __init__(self, args):
        self.args = args
        self.dataloaders_dict, self.image_datasets = self.get_transform_data()
        self.model = self.load_model(self.args.checkpoint)


    # Adapted From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, feature_extract, use_pretrained=True):
        model_ft = None
        input_size = 0
        if self.args.arch.lower() == 'vgg16':
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
        if self.args.arch.lower() == 'alexnet':
            model_ft = models.alexnet(pretrained=use_pretrained)
        self.set_parameter_requires_grad(model_ft, feature_extract)
        max_clf_idx = (len(model_ft.classifier) - 1)
        num_ftrs = model_ft.classifier[max_clf_idx].in_features
        num_classes = len(self.image_datasets['train'].classes)
        model_ft.classifier[max_clf_idx] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return model_ft, input_size

    def get_transform_data(self):
        data_dir = self.args.data_directory
        data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])
                                                        ]),
                           'test': transforms.Compose([transforms.Resize(255),
                                                       transforms.CenterCrop(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                            [0.229, 0.224, 0.225])
                                                       ]),
                           'valid': transforms.Compose([transforms.Resize(255),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])
                                                        ])
                           }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'test', 'valid']}

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=8, shuffle=True,
                                                           num_workers=4) for x in ['train', 'valid', 'test']}

        return dataloaders_dict, image_datasets

    def load_model(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model, input_size = self.initialize_model(feature_extract=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        return transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])(PIL.Image.open(self.args.image_path))

    def predict(self):
        # Transform
        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.args.gpu is True) else "cpu")
        cat_to_name = json.load(open(self.args.category_names, 'r'))
        actual_category = cat_to_name[Path(self.args.image_path).parent.name]
        inputs = self.process_image().unsqueeze(0)
        self.model.eval()
        model, inputs = self.model.to(device), inputs.to(device)
        prob = nnf.softmax(model(inputs), dim=1)
        class_to_idx = self.image_datasets['train'].class_to_idx
        idx_to_class = {val: key for key, val in class_to_idx.items()}
        top_p, top_class = prob.topk(self.args.top_k, dim=1)
        top_class = [idx_to_class[int(idx)] for idx in top_class[0]]
        top_p = [p.cpu().detach().numpy() for p in top_p[0]]
        class_name = [cat_to_name[t] for t in top_class]
        print('\n', '*' * 30, f'\nACTUAL FLOWER NAME: {actual_category}\n', '*' * 30)
        df = pd.DataFrame({'probability': top_p,
                           'category': top_class,
                           'class_name': class_name})
        df['probability'] = df['probability'].astype(float).round(5)
        return df

main()