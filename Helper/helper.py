import torchvision.transforms as transforms


def convert_to_yoloFormat(width, height, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]


# data agumentation

agumenteda_data  = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

