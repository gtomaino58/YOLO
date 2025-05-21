# Version de YOLOV1 con Darknet-19 como backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from collections import OrderedDict # Todavía puede ser útil para la estructura de tu CNNBlock o si las usas internamente

# --- NUEVA IMPORTACIÓN DE HOLOCRON ---
import holocron.models as models # Importa el módulo models de holocron
from torchvision.transforms.functional import InterpolationMode # Para transformaciones, si se usan con holocron


# --- 1. FUNCIÓN INTERSECTION_OVER_UNION (IoU) ---
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError("box_format must be 'midpoint' or 'corners'")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    box1_area = box1_area.clamp(min=0)
    box2_area = box2_area.clamp(min=0)

    union = box1_area + box2_area - intersection + 1e-6

    return intersection / union

# --- 2. CLASE DE LA FUNCIÓN DE PÉRDIDA (YoloLoss) ---
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        box_preds = predictions[..., :5*self.B].reshape(-1, self.S, self.S, self.B, 5)
        class_preds = predictions[..., 5*self.B:]

        target_box = target[..., :5]
        target_class = target[..., 5*self.B:]

        obj_mask = target[..., 4] == 1
        noobj_mask = ~obj_mask       

        noobj_loss = 0
        for b_idx in range(self.B):
            noobj_pred_conf = box_preds[noobj_mask][:, b_idx, 4]
            noobj_target_conf = torch.zeros_like(noobj_pred_conf)
            noobj_loss += F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        noobj_loss *= self.lambda_noobj

        obj_box_preds = box_preds[obj_mask]
        obj_target_box = target_box[obj_mask]

        if obj_box_preds.numel() == 0:
            return noobj_loss

        batch_indices, i_indices, j_indices = torch.where(obj_mask)

        x_cell_preds_b = obj_box_preds[..., 0]
        y_cell_preds_b = obj_box_preds[..., 1]
        w_preds_b = obj_box_preds[..., 2]
        h_preds_b = obj_box_preds[..., 3]
        
        i_expanded = i_indices.unsqueeze(1).expand_as(x_cell_preds_b).to(obj_box_preds.device)
        j_expanded = j_indices.unsqueeze(1).expand_as(y_cell_preds_b).to(obj_box_preds.device)

        x_global_preds = (j_expanded.float() + x_cell_preds_b) / self.S
        y_global_preds = (i_expanded.float() + y_cell_preds_b) / self.S
        
        global_pred_boxes = torch.stack([x_global_preds, y_global_preds, w_preds_b, h_preds_b], dim=-1)

        x_cell_gt = obj_target_box[:, 0]
        y_cell_gt = obj_target_box[:, 1]
        w_gt = obj_target_box[:, 2]
        h_gt = obj_target_box[:, 3]

        x_global_gt = (j_indices.float() + x_cell_gt) / self.S
        y_global_gt = (i_indices.float() + y_cell_gt) / self.S
        
        global_gt_boxes = torch.stack([x_global_gt, y_global_gt, w_gt, h_gt], dim=-1)
        
        global_gt_boxes_expanded = global_gt_boxes.unsqueeze(1)
        
        ious = intersection_over_union(global_pred_boxes, global_gt_boxes_expanded, box_format="midpoint")

        best_box_iou, best_box_idx = torch.max(ious, dim=1)
        
        responsible_box_preds = obj_box_preds[torch.arange(obj_box_preds.shape[0]), best_box_idx]
        
        loss_xy = F.mse_loss(responsible_box_preds[:, :2], obj_target_box[:, :2], reduction='sum')
        loss_wh = F.mse_loss(
            torch.sign(responsible_box_preds[:, 2:4]) * torch.sqrt(torch.abs(responsible_box_preds[:, 2:4]) + 1e-6),
            torch.sqrt(obj_target_box[:, 2:4]),
            reduction='sum'
        )
        coord_loss = self.lambda_coord * (loss_xy + loss_wh)

        obj_pred_conf = responsible_box_preds[:, 4]
        obj_target_conf = best_box_iou.float().to(obj_pred_conf.device)
        
        obj_conf_loss = F.mse_loss(obj_pred_conf, obj_target_conf, reduction='sum')

        class_loss = F.mse_loss(class_preds[obj_mask], target_class[obj_mask], reduction='sum')

        total_loss = coord_loss + obj_conf_loss + noobj_loss + class_loss

        return total_loss

# --- 3. CLASE DEL DATASET (MyYOLOv1Dataset) ---
class MyYOLOv1Dataset(Dataset):
    def __init__(self, csv_file, img_dir, S=7, B=2, C=3, transform=None):
        self.raw_annotations = pd.read_csv(csv_file, header=0, names=['image_name', 'class_id', 'x', 'y', 'w', 'h'])
        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        self.grouped_annotations = self.raw_annotations.groupby('image_name').apply(
            lambda x: x[['class_id', 'x', 'y', 'w', 'h']].values.tolist(),
            include_groups=False
        ).to_dict()

        self.image_names = list(self.grouped_annotations.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        object_annotations = self.grouped_annotations[img_name]

        target = torch.zeros((self.S, self.S, self.B * 5 + self.C), dtype=torch.float32)

        for anno in object_annotations:
            class_id, x, y, w, h = anno

            i, j = int(self.S * y), int(self.S * x)

            if i >= self.S or j >= self.S or i < 0 or j < 0:
                continue
            
            x_cell = self.S * x - j
            y_cell = self.S * y - i

            if target[i, j, 4] == 0:
                target[i, j, 0:4] = torch.tensor([x_cell, y_cell, w, h], dtype=torch.float32)
                target[i, j, 4] = 1.0

                class_one_hot = torch.zeros(self.C, dtype=torch.float32)
                class_one_hot[int(class_id)] = 1.0
                target[i, j, self.B * 5 : self.B * 5 + self.C] = class_one_hot
            else:
                pass

        if self.transform:
            image = self.transform(image)

        return image, target

## 4. CLASE DEL MODELO YOLOv1 (con Backbone Darknet-19 de Holocron)

# Eliminamos la clase CNNBlock personalizada, ya que la de Holocron gestionará la estructura interna.
# Si la tenías definida en algún otro lugar, puedes borrarla de allí.

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, S=7, B=2, C=20, pretrained_holocron_model_name="darknet19", freeze_darknet=False):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C

        # Cargar el modelo Darknet19 pre-entrenado de Holocron models directamente
        # models.darknet19(pretrained=True) cargará la versión pre-entrenada en ImageNet.
        print(f"Cargando backbone Darknet19 de Holocron models: {pretrained_holocron_model_name}")
        holocron_darknet = models.darknet19(pretrained=True) # <-- Carga directa
        
        # Tomar solo la parte de características (backbone) del modelo de Holocron.
        # La convención en holocron es que el backbone convolucional está en el atributo 'features'.
        self.darknet = holocron_darknet.features 
        
        # Capas completamente conectadas (FCs) - estas se entrenan desde cero para tu tarea
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096), # La salida de Darknet-19 es 1024 canales en 7x7 grid
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

        # Congelar capas del Darknet si se especifica
        if freeze_darknet:
            for param in self.darknet.parameters():
                param.requires_grad = False
            print("Capas del Darknet (Holocron) congeladas (requires_grad=False).")
        
        # Asegúrate de que las capas FC siempre requieran gradiente (para fine-tuning de tu tarea)
        for param in self.fcs.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x)
        return x.reshape(-1, self.S, self.S, self.B * 5 + self.C)


## 5. CONFIGURACIÓN Y BUCLE DE ENTRENAMIENTO PRINCIPAL (con validación)

# --- Configuración de Hiperparámetros (¡Ajusta estos para tu entrenamiento!) ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"CUDA está disponible. Usando la GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA no está disponible. Usando la CPU.")

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 100
NUM_WORKERS = 4 
IMG_HEIGHT = 448
IMG_WIDTH = 448
S = 7 
B = 2 
C = 3 # TU NÚMERO DE CLASES (0, 1, 2 en tu caso)

# --- Rutas de datos (¡ACTUALIZA ESTAS RUTAS CON LAS TUYAS REALES!) ---
FULL_CSV_FILE = '/home/224F8578gianfranco/YOLO/annotations_normalized.csv'
IMG_DIR = '/home/224F8578gianfranco/YOLO/BCCD/'

# --- Proporción para la división de datos (si usas un solo CSV) ---
TRAIN_SPLIT_RATIO = 0.8 

# --- Transformaciones de datos ---
# Usamos las transformaciones estándar de ImageNet que son muy similares
# a las que Holocron espera. Se ha añadido PILToTensor y ConvertImageDtype explícitamente.
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.PILToTensor(), # Convierte PIL Image a PyTorch Tensor
    transforms.ConvertImageDtype(torch.float32), # Asegura que sea de tipo float32
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Función para un paso de entrenamiento por época ---
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        predictions = model(x)
        loss = loss_fn(predictions, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}] Pérdida: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    return avg_loss

# --- Función para un paso de evaluación (validación) ---
def eval_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss

# --- Función principal que orquesta el entrenamiento ---
def main():
    
    full_dataset = MyYOLOv1Dataset(
        csv_file=FULL_CSV_FILE,
        img_dir=IMG_DIR,
        S=S, B=B, C=C,
        transform=transform
    )
    
    print(f"Total de imágenes cargadas por el dataset: {len(full_dataset)}") # <-- Añade esto
    
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last= False
        #drop_last=True
    )

    # 2. Inicializar modelo con el backbone de Holocron
    # `pretrained_holocron_model_name` se pasa para cargar el modelo de Holocron
    # Cargamos Darknet19 de holocron.models directamente, con pretrained=True
    # `C` es el número de clases de TU dataset.
    model = YOLOv1(S=S, B=B, C=C, 
                pretrained_holocron_model_name="darknet19", # Nombre del modelo en holocron.models
                freeze_darknet=True).to(DEVICE) # Empieza congelando el backbone
    
    print(f"Modelo movido a: {next(model.parameters()).device}")
    
    # Después de inicializar el modelo, cuenta y muestra los parámetros entrenables/no entrenables
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print(f"Resumen del modelo YOLOv1:")
    print(f"Parámetros totales: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")
    print(f"Parámetros no entrenables: {non_trainable_params}")
    
    # El optimizador solo optimiza los parámetros que tienen requires_grad=True
    # Al principio, solo las capas FC si el backbone está congelado.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    loss_fn = YoloLoss(S=S, B=B, C=C)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    print(f"Iniciando entrenamiento en el dispositivo: {DEVICE}")
    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)} imágenes")
    print(f"Tamaño del dataset de validación: {len(val_dataset)} imágenes")


    # 3. Bucle de entrenamiento principal
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Época {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Lógica para descongelar capas (Opcional, pero recomendado en TL) ---
        # Descongelar el backbone después de 10 épocas y reducir la tasa de aprendizaje.
        if epoch == 10: 
            print("Descongelando capas del Darknet (Holocron) y ajustando la tasa de aprendizaje.")
            for param in model.darknet.parameters():
                param.requires_grad = True # Descongelar
            # Crea un nuevo optimizador para incluir ahora los parámetros descongelados
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


        current_train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        print(f"Pérdida de Entrenamiento de la Época {epoch+1}: {current_train_loss:.4f}")
        
        current_val_loss = eval_fn(val_loader, model, loss_fn, DEVICE)
        print(f"Pérdida de Validación de la Época {epoch+1}: {current_val_loss:.4f}")

        if scheduler:
            scheduler.step(current_val_loss)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"yolov1_epoch_{epoch+1}.pth")
            print(f"¡Modelo guardado en yolov1_epoch_{epoch+1}.pth!")

    print("\nEntrenamiento y Validación finalizados.")

# --- Ejecución del script ---
if __name__ == "__main__":
    main()