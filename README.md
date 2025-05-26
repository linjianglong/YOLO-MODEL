The impact of floating debris in rivers on water environments has become increasingly severe with rapid urbanization and industrialization. To address the challenges of low detection accuracy and high false-negative rates in existing methods, especially under complex water surface backgrounds and multi-scale object recognition, this study proposes an enhanced river floating debris detection algorithm based on YOLOv8n. We integrate an Efficient Multi-scale Attention (EMA) mechanism into the C2f module to enhance object recognition in complex backgrounds. A Deep Cross-scale Feature Fusion (DCWF) module is designed to improve the model's adaptability to scale variations. An Inner Intersection over Union (Inner-IoU) loss function, based on auxiliary bounding boxes, is introduced to optimize bounding box regression and enhance localization accuracy. A diversified floating debris dataset, encompassing various environments and water surface conditions, is constructed to bolster the model's generalization ability.These enhancements significantly improve detection performance, offering new insights into the development of intelligent river waste monitoring technology.

If you want to train the model, add the EMA attention mechanism, the DCWF module, and the Inner-iou module to the YOLOv8n benchmark model for training，Here's how to add it：
1.First, the EMA Attention module is implemented
1.1 The EMA attention mechanism code is given in the document and can be added as follows：
    Modified C2f module to integrate EMA attention， The details are as follows：
    class C2f_EMA(nn.Module):
    """C2f module with EMA attention."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ema = EMA(self.c)  # Add EMA attention
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # Apply EMA attention to the last feature map
        y[-1] = self.ema(y[-1])
        
        return self.cv2(torch.cat(y, 1))
  Replace the original C2f module in YOLOv8n，The training is then validated using the methods described below
  from ultralytics import YOLO
# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')
#Replace all C2f modules in the model with C2f_EMA
replace_c2f_with_c2f_ema(model.model)
# View the model structure
print(model.model)
# Training or inference
results = model.train(data='data.yaml', epochs=100, imgsz=640)

2.Implement the DCWF module
2.1 The DCWF module code has been uploaded to a file and is implemented as follows：
    Modify the relevant modules in YOLOv8n，The following is the following to modify the code
    from ultralytics.nn.modules import Conv, C2f, Bottleneck
def replace_concat_with_dcwf(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for i, m in enumerate(module):
                if isinstance(m, (C2f, Bottleneck)):
                    replace_concat_in_c2f(m)
        elif hasattr(module, 'forward_concat'): 
            in_channels = module.forward_concat[0].in_channels * 2
            setattr(model, name, DCWF(in_channels))
        else:
            replace_concat_with_dcwf(module)
def replace_concat_in_c2f(c2f_module):
    if not hasattr(c2f_module, 'cv2'):
        return
    n = len(c2f_module.m)
    in_channels = (2 + n) * c2f_module.c
    c2f_module.cv2 = DCWF(in_channels, c2f_module.cv2.conv.out_channels)

  Modify the C2f module to support DCWF as follows：
  class C2f_DCWF(nn.Module):
    """C2f module with DCWF instead of concat"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = DCWF((2 + n) * self.c, c2) 
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(y) 
  The overall replacement method and verification command are as follows
 from ultralytics import YOLO
 import math
def apply_dcwf_to_yolov8n():
    # Load the model
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    # Replace all C2f modules with C2f_DCWF
    for name, module in model.model.named_children():
        if isinstance(module, C2f):
            # Get the parameters
            c1 = module.cv1.conv.in_channels
            c2 = module.cv2.conv.out_channels
            n = len(module.m)
            shortcut = module.m[0].shortcut
            g = module.m[0].cv2.conv.groups
            e = module.c / c2
  
            # Replace with C2f_DCWF
            setattr(model.model, name, C2f_DCWF(c1, c2, n, shortcut, g, e))
    
    # Replace the concat operation elsewhere
    replace_concat_with_dcwf(model.model)
    return model

# Example of use
model = apply_dcwf_to_yolov8n()
print(model.model)  # View the modified model structure
# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)

3.Code implementation using the Inner-IoU loss function in YOLOv8n
3.1 The code for the Inner-IoU loss function has been uploaded to the file, and the following describes how to do it
    Modified the loss calculation class of YOLOv8n
    class v8DetectionLossWithInnerIoU(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.box_loss = InnerIoULoss(iou_ratio=0.7, box_format='xywh')
    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
   Replace the loss function of the YOLOv8n model:
   from ultralytics import YOLO
def apply_inner_iou_to_yolov8n():
    # Load the model
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    # Replace the loss function
    model.model.loss = v8DetectionLossWithInnerIoU(model.model)
    return model
# Example of use
model = apply_inner_iou_to_yolov8n()
