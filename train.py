from workspace_utils import active_session
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os, copy, time, json, argparse
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--hidden_units', default=4096)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--arch', default='vgg16', help='HINT: [vgg16 or alexnet]')
    args = parser.parse_args()

    trainer = DL_Trainer(args)
    trainer.train_and_save()


class DL_Trainer:
    def __init__(self, args):
        self.args = args
        self.dataloaders_dict, self.image_datasets = self.get_transform_data()
        self.model = self.initialize_model(feature_extract=True)[0]

    # Adapted From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, feature_extract, use_pretrained=True):
        model_ft = None
        hidden_unit_size = self.args.hidden_units
        input_size = 0
        if self.args.arch.lower() == 'vgg16':
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
        if self.args.arch.lower() == 'alexnet':
            model_ft = models.alexnet(pretrained=use_pretrained)
        self.set_parameter_requires_grad(model_ft, feature_extract)
        num_classes = len(self.image_datasets['train'].classes)

        ##Adapted from Udacity Forums
        if hidden_unit_size != 4096:
            if self.args.arch.lower() == 'vgg16':
                in_feats = model_ft.classifier[0].in_features
            elif self.args.arch.lower() == 'alexnet':
                in_feats = model_ft.classifier[1].in_features

            hidden_units = [int(x) for x in [hidden_unit_size, hidden_unit_size]]
            hidden_units = [in_feats] + hidden_units
            hidden_units_pair = list(zip(hidden_units, hidden_units[1:]))
            hidden_units_pair.append((hidden_units[-1], num_classes))

            lcnt, rcnt, dcnt = 0, 0, 0
            layers = []
            for layer in enumerate(model_ft.classifier):
                if 'Linear' in str(layer):
                    layers.append((f'fc{lcnt}', nn.Linear(in_features=hidden_units_pair[lcnt][0],
                                                          out_features=hidden_units_pair[lcnt][1], bias=True)))
                    lcnt += 1
                elif 'ReLU' in str(layer):
                    layers.append((f'relu{rcnt}', nn.ReLU(inplace=True)))
                    rcnt += 1
                elif 'Drop' in str(layer):
                    layers.append((f'drop{dcnt}', nn.Dropout(p=0.5)))
                    dcnt += 1

            classifier = nn.Sequential(OrderedDict(layers))
            model_ft.classifier = classifier
            print(model_ft.classifier)

        max_clf_idx = (len(model_ft.classifier) - 1)
        num_ftrs = model_ft.classifier[max_clf_idx].in_features
        model_ft.classifier[max_clf_idx] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        return model_ft, input_size

    # Adapted From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def train_model(self, model, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.args.gpu is True) else "cpu")
        print(device)
        model.to(device)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders_dict[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

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

    # Adapted From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def train_and_save(self):
        with active_session():
            feature_extract = True
            # COMMENTED OUT PER REVIEWS INSTRUCTIONS
            # params_to_update = model.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                for name, param in self.model.named_parameters():
                    if param.requires_grad == True:
                        print("\t", name)

            optimizer_ft = optim.SGD(params_to_update, lr=self.args.learning_rate, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            self.model, hist = self.train_model(self.model,
                                                criterion=criterion,
                                                optimizer=optimizer_ft,
                                                num_epochs=self.args.epochs)

            torch.save(self.model.state_dict(),
                       f'{self.args.save_dir}/cli_checkpoint_{self.args.arch}.pth')


main()
