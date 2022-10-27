#%%
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
from PIL import Image

from train_model import Net
from train_utils import Get_Device

#%%

if __name__ == "__main__":
    IMAGE_SIZE = 48

    train_transform = transforms.Compose([
                                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                        transforms.Normalize((0.5, ), (0.5, ))
                                        ])

    device = Get_Device()

    model_path = "weights/model.pth"

    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    emotion_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happy', 4 : 'Neutral', 5 : 'Sad', 6 : 'Surprise'}

    model.eval()
    with torch.no_grad():
            img = cv2.imread("img.jpg")
            img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            image_tensor = train_transform(img2).float()
            image_tensor = image_tensor.unsqueeze_(0).to(device)
            pred = model(image_tensor) # 進行預測
            output = F.softmax(pred).cpu().numpy()[0] # 轉換成每個項目的機率
            output_list = output.tolist()
            output_index = output_list.index(max(output_list)) # 取出機率最高的那一項

            # print("{} : {:0.3f} %".format(emotion_dict[output_index], output[output_index]*100))

            for i in range(len(emotion_dict)):
                print("{} : {:0.3f} %".format(emotion_dict[i], output[i]*100))

# %%