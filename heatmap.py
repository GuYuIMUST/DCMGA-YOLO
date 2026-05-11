import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, copy
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image


# -------------------------------------------------------------------------------------------
# 工具函数：预处理图像
# -------------------------------------------------------------------------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (top, bottom, left, right)


# -------------------------------------------------------------------------------------------
# 核心类：提取激活值与梯度 (适配非 End2End 输出)
# -------------------------------------------------------------------------------------------
class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        # YOLO 输出通常为 [batch, 4+nc, anchors]，转置为 [anchors, 4+nc]
        result = result.squeeze(0).transpose(0, 1)
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        # 获取最大类别的置信度用于排序
        conf, _ = torch.max(logits_, dim=1)
        _, indices = torch.sort(conf, descending=True)
        return logits_[indices], boxes_[indices]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        # 即使模型内部有 end2end，由于我们初始化时关闭了它，这里拿到的是原始 Tensor
        post_result, pre_post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


# -------------------------------------------------------------------------------------------
# 目标函数类
# -------------------------------------------------------------------------------------------
class yolo_detect_target(torch.nn.Module):
    def __init__(self, output_type, conf, ratio):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        num_to_process = int(post_result.size(0) * self.ratio)
        for i in range(num_to_process):
            val = post_result[i].max()
            if float(val) < self.conf: break
            if self.output_type in ['class', 'all']:
                result.append(val)
            if self.output_type in ['box', 'all']:
                result.append(pre_post_boxes[i].sum())
        return sum(result) if result else torch.tensor(0.0, requires_grad=True)


# -------------------------------------------------------------------------------------------
# 主类：热力图生成引擎
# -------------------------------------------------------------------------------------------
class yolo_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, img_size):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.show_result = show_result

        # 加载模型
        self.model_yolo = YOLO(weight)
        model = copy.deepcopy(self.model_yolo.model).to(self.device)

        # --- 关键修改：强行关闭 end2end 以允许梯度回传 ---
        model.end2end = False
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        self.model = model

        # 设置目标层
        self.target_layers = [self.model.model[l] for l in layer]
        self.target = yolo_detect_target(backward_type, conf_threshold, ratio)

        # 初始化 CAM 方法
        self.method = eval(method)(model, self.target_layers)
        self.method.activations_and_grads = ActivationsAndGradients(model, self.target_layers, None)

    def process(self, img_path, save_path):
        img_raw = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        if img_raw is None: return
        h_orig, w_orig = img_raw.shape[:2]

        # 1. 预处理
        img_box, _, (top, bottom, left, right) = letterbox(img_raw, new_shape=(self.img_size, self.img_size),
                                                           auto=False)
        img_float = np.float32(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # --- 关键修改：开启输入张量的梯度追踪 ---
        tensor.requires_grad = True

        # 2. 生成热力图
        try:
            grayscale_cam = self.method(tensor, [self.target])[0, :]
        except Exception as e:
            print(f"Error at {img_path}: {e}")
            return

        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # 3. 叠加原图检测结果
        if self.show_result:
            # 预测时使用原始 YOLO 接口（它会自动处理 end2end）
            pred = self.model_yolo.predict(img_box, conf=self.conf_threshold, verbose=False)[0]
            cam_image = pred.plot(img=cam_image, labels=True)

        # 4. 裁剪补边并 Resize 回原图尺寸
        valid_h = cam_image.shape[0] - bottom
        valid_w = cam_image.shape[1] - right
        cam_image = cam_image[top:valid_h, left:valid_w]
        cam_image = cv2.resize(cam_image, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        Image.fromarray(cam_image).save(save_path)
        print(f"Success: {save_path} ({w_orig}x{h_orig})")

    def __call__(self, img_path, save_root):
        if os.path.exists(save_root): shutil.rmtree(save_root)
        os.makedirs(save_root, exist_ok=True)
        files = [os.path.join(img_path, f) for f in os.listdir(img_path) if
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            self.process(f, os.path.join(save_root, os.path.basename(f)))


# -------------------------------------------------------------------------------------------
# 启动配置
# -------------------------------------------------------------------------------------------
def get_params():
    return {
        'weight': r'D:\leo\ultralytics-main26\ultralytics-main\runs\detect\runs\end330\yolo262\weights\best.pt',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',  # 推荐 PlusPlus，梯度更平滑
        'layer': [16, 19, 22],  # YOLO26 核心检测头层
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_result': False,
        'img_size': 640,
    }


if __name__ == '__main__':
    params = get_params()
    engine = yolo_heatmap(**params)
    engine(r'D:\qileo\last\image', r'D:\leo\ultralytics-main26\result26')