import torchvision.transforms as T
import torch
import os
from PIL import Image

# torch.Size([3, 3, 3, 512, 512]) 로 들어오는 문제를 방지하기 위한 함수
def collate_ft(batch):
    imgs_list, labels_list = zip(*batch)
    
    # 이미지는 기존대로 Stack 후 View
    imgs = torch.stack(imgs_list, dim=0)
    
    B,K,C,H,W = imgs.shape
    imgs = imgs.view(B*K,C,H,W) 
    
    # labels_list = (['cat', 'dog', 'wild'], ['cat', 'dog', 'wild'], ...)
    labels = [item for sublist in labels_list for item in sublist]
    
    return imgs, labels

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        super().__init__()
        
        BASE_PATH = "data/dataset/afhq/train"
        self.test = test
        self.basepath = BASE_PATH
        self.transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor()
            ])
        
        self.category = ["cat", "dog", "wild"]
        
        if self.test:
            self.file_paths = []
            
            for category in self.category:
                dir_path = os.path.join(BASE_PATH, category)
                file_list = sorted(os.listdir(dir_path))
                target_files = file_list[-1700:]

                for filename in target_files:
                    full_path = os.path.join(dir_path, filename)
                    self.file_paths.append((full_path, category))
                    
            self.len = len(self.file_paths) 
            
        else: 
            self.lists = {}
            min_len = 10000000 
            
            for item in self.category:
                dir_path = os.path.join(BASE_PATH, item)
                
                full_list = sorted(os.listdir(dir_path))
                train_list = full_list[:-1700] 
                
                self.lists[item] = train_list
                
                if len(train_list) < min_len:
                    min_len = len(train_list)
            
            self.len = min_len

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.test:
            path, label_str = self.file_paths[idx]
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, label_str
        else:
            imgs = []
            labels = []
            
            for category in self.category:
                file_list = self.lists[category]
                
                filename = file_list[idx] 
                path = os.path.join(self.basepath, category, filename)
                
                img = Image.open(path).convert("RGB")
                img = self.transform(img)
                
                imgs.append(img)     
                labels.append(category)

            
            return torch.stack(imgs, dim=0), labels
        
def test():
    data_test = CustomDataset(test=True)
    dataloader_test = torch.utils.data.DataLoader(data_test)
    print(len(dataloader_test))
    
    a,b = next(iter(dataloader_test))
    print(a.shape, b) 
    
#test()


#print("dataloader.py executed...")