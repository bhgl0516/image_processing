import torch
import torchvision
import torchvision.transforms as transforms


class AddSPNoise(object):
    """
    检查要点2：严格实现 胡椒(0.1) 和 盐粒(0.1) 概率的噪声注入
    """

    def __init__(self, prob_salt=0.1, prob_pepper=0.1):
        self.prob_salt = prob_salt
        self.prob_pepper = prob_pepper

    def __call__(self, tensor):
        # tensor 此时为 ToTensor 后的 [0.0, 1.0]
        noisy = tensor.clone()
        _, h, w = tensor.shape
        # 生成单通道随机矩阵，确保 RGB 同步变黑或变白
        rand_mat = torch.rand(1, h, w)

        # 盐粒噪声 (白点: 1.0)
        noisy[:, rand_mat[0] < self.prob_salt] = 1.0
        # 胡椒噪声 (黑点: 0.0)
        noisy[:, rand_mat[0] > (1.0 - self.prob_pepper)] = 0.0
        return noisy


def get_dataloaders(data_dir, batch_size=64):
    """获取三种 DataLoader: 训练集、干净测试集、带噪测试集"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 检查要点2要求：注入指定概率噪声
    transform_test_noisy = transforms.Compose([
        transforms.ToTensor(),
        AddSPNoise(prob_salt=0.1, prob_pepper=0.1),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化必须在噪声之后
    ])

    # 自动处理 cifar-10-python.tar.gz 下载与解压
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset_clean = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                                 transform=transform_test_clean)
    testloader_clean = torch.utils.data.DataLoader(testset_clean, batch_size=batch_size, shuffle=False, num_workers=0)

    testset_noisy = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                                 transform=transform_test_noisy)
    testloader_noisy = torch.utils.data.DataLoader(testset_noisy, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    return trainloader, testloader_clean, testloader_noisy, classes