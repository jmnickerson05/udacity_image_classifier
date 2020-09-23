import torch
from torchvision import datasets, transforms, models
import PIL
import torch.nn.functional as nnf
import pandas as pd, argparse, json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('checkpoint', default='cli_checkpoint.pth')
    parser.add_argument('--top_k', default=5)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--show_plots', default=False)
    global args
    args = parser.parse_args()
    model = load_model()

    if not args.show_plots:
        predict(image_path=args.image_path, model=model, topk=args.top_k)
    else:
        predict_and_plot(model, args.image_path)


def load_model():
    model = torch.load('cli_checkpoint.pth', ((lambda storage, loc: storage.cuda())
            if (torch.cuda.is_available() and args.gpu is True) else 'cpu'))
    model.eval()
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
        ])(PIL.Image.open(image_path))

def predict(image_path, model, topk=5):
    # Transform
    cat_to_name = json.loads(open(args.category_names, 'r').read())
    actual_category = cat_to_name[image_path.split('/')[2]]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = process_image(image_path).unsqueeze(0)
    model.eval()
    model, inputs = model.to(device), inputs.to(device)
    prob = nnf.softmax(model(inputs), dim=1)
    top_p, top_class = prob.topk(topk, dim = 1)
    pred = pd.DataFrame({'probability': top_p.cpu().detach().numpy()[0],
                         'category': top_class.cpu().detach().numpy()[0]})
    ppred = pred.sort_values(by='probability', ascending=False)[
        ['category', 'probability']
    ]
    pred['name'] = pred.category.apply(
        lambda x: cat_to_name[str(int(x))]
    )
    pred['category'] = pred['category'].astype(int)
    print('\n','*'*30, f'\nACTUAL FLOWER NAME: {actual_category}\n', '*'*30)
    print(pred)
    return pred

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    return ax

def predict_and_plot(model, img_path):
    cat_to_name = json.loads(open(args.category_names, 'r').read())
    imshow(process_image(img_path))
    pred = predict(img_path, model)
    pred = pred.sort_values(by='probability', ascending=False)[
        ['category', 'probability']
    ]
    pred['name'] = pred.category.apply(
        lambda x: cat_to_name[str(int(x))]
    )
    pred['category'] = pred['category'].astype(int)
    print(pred)
    ax = (pred.sort_values(by='probability', ascending=False)
          .plot.barh(x='name', y='probability'))
    ax.invert_yaxis()

main()