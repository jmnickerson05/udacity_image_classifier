import torch
from torchvision import transforms, models
import PIL
import torch.nn.functional as nnf
import pandas as pd, argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--data_directory', default='flowers/')
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    predictor = DL_Predictor(args)
    print(predictor.predict())


class DL_Predictor:
    def __init__(self, args):
        self.args = args
        self.checkpoint = self.set_checkpoint()
        self.model = self.load_model()

    def set_checkpoint(self):
        return torch.load(self.args.checkpoint)

    def load_model(self):
        model = None
        input_size = 0
        if 'vgg16' in self.args.checkpoint:
            model = models.vgg16_bn(pretrained=True)
        if 'alexnet' in self.args.checkpoint:
            model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.class_to_idx = self.checkpoint['class_to_idx']
        model.classifier = self.checkpoint['classifier']
        model.load_state_dict(self.checkpoint['state_dict'])
        model.eval()
        return model

    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array.
        '''
        return transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])(PIL.Image.open(self.args.image_path))

    def predict(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.args.gpu is True) else "cpu")
        cat_to_name = json.load(open(self.args.category_names, 'r'))
        actual_category = cat_to_name[Path(self.args.image_path).parent.name]
        inputs = self.process_image().unsqueeze(0)
        self.model.eval()
        model, inputs = self.model.to(device), inputs.to(device)
        prob = nnf.softmax(model(inputs), dim=1)
        # class_to_idx = self.image_datasets['train'].class_to_idx
        class_to_idx = self.checkpoint['class_to_idx']
        idx_to_class = {val: key for key, val in class_to_idx.items()}
        top_p, top_class = prob.topk(self.args.top_k, dim=1)
        top_class = [idx_to_class[int(idx)] for idx in top_class[0]]
        top_p = [p.cpu().detach().numpy() for p in top_p[0]]
        class_name = [cat_to_name[t] for t in top_class]
        print('\n', '*' * 30, f'\nACTUAL FLOWER NAME: {actual_category}\n', '*' * 30)
        df = pd.DataFrame({'class_name': class_name,
                           'probability': top_p,
                           'category': top_class,
                           })
        df['probability_rounded'] = df['probability'].astype(float).round(5)
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 150)
        return df[['class_name', 'probability', 'probability_rounded', 'category']]

main()