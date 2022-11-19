from models.fan_model import  FAN
from utils.imutils import im_to_numpy, im_to_torch
from utils.transforms import transform, transform_preds
from utils.evaluation import get_preds
from utils.misc import *
import torch
from PIL import Image

class LandmarkDetector73():

    def __init__(self, num_lmks, checkpoint_dir):

        self.num_lmks = num_lmks
        self.fan_model = FAN(2, 73)
        self.fan_model = torch.nn.DataParallel(self.fan_model).cuda()
        print(f"=> Loading checkpoint {checkpoint_dir}")
        self.checkpoint = torch.load(checkpoint_dir)
        self.fan_model.load_state_dict(self.checkpoint['state_dict'])
        print(f"=> Loaded checkpoint {checkpoint_dir} (epoch {self.checkpoint['epoch']})")


    def final_preds(self, output, center, scale, res,):
        if output.size(1) == 136:
            coords = output.view((output.szie(0), self.num_lmks, 2))
        else:
            coords = get_preds(output[:,:self.num_lmks,:,:])  # float type


        # output shape is batch, num_lmks, 64, 64
        # coords shape is batch, num_lmks, 2
        # pose-processing
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = output[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if px > 1 and px < res[0] and py > 1 and py < res[1]:
                    diff = torch.Tensor(
                        [hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                    coords[n][p] += diff.sign() * .25
        coords += 0.5
        preds = coords.clone()


        # Transform back
        for i in range(coords.size(0)):
            preds[i] = transform_preds(coords[i], center, scale, res)

        if preds.dim() < 3:
            preds = preds.view(1, preds.size())
        return preds


    def validate(self,inputs, model,center,scale):


        model.eval()
        input_var = torch.autograd.Variable(inputs.cuda())
        output = model(input_var)
        score_map = output[-1].data.cpu()

        preds = self.final_preds(score_map, center,scale, [64, 64])

        return preds



    def crop(self,img, center, scale, res=[256, 256], rot=0):
        img = im_to_numpy(img)

        # Preprocessing for efficient cropping
        ht, wd = img.shape[0], img.shape[1]
        sf = scale * 200.0 / res[0]
        if sf < 2:
            sf = 1
        else:
            new_size = int(np.math.floor(max(ht, wd) / sf))
            new_ht = int(np.math.floor(ht / sf))
            new_wd = int(np.math.floor(wd / sf))
            if new_size < 2:
                return torch.zeros(res[0], res[1], img.shape[2]) \
                    if len(img.shape) > 2 else torch.zeros(res[0], res[1])
            else:
                img = scipy.misc.imresize(img, [new_ht, new_wd])
                center = center * 1. / sf
                scale = scale / sf

        # Upper left point
        ul = np.array(transform([0, 0], center, scale, res, invert=1))
        # Bottom right point
        br = np.array(transform(res, center, scale, res, invert=1))

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        if not rot == 0:
            # Remove padding
            new_img = scipy.misc.imrotate(new_img, rot)
            new_img = new_img[pad:-pad, pad:-pad]

        new_img = np.array(new_img*255, dtype = np.uint8)

        new_img = im_to_torch(np.array(Image.fromarray(new_img).resize((256,256))))
        return new_img


    def detect_landmarks_custom(self, image):

        img = np.array(image)
        c = torch.Tensor((img.shape[0] / 2, img.shape[1] / 2))
        s = 1.8
        inp = self.crop(im_to_torch(img), c, s, [256, 256], rot=0)
        landmark = self.validate(inp[None, ...], self.fan_model, c , s)

        landmark = landmark[0]

        return landmark ##num_lmks X 2

















