import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import pandas as pd
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import ast

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define constants
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_IMAGES_DIR = 'train_images'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCUMULATION_STEPS = 2
CHECKPOINT_PATH = 'product_image_model_checkpoint.pth'
MODEL_PATH = 'product_image_model_final.pth'
TRAIN_CSV_PATH = 'dataset/train.csv'
TEST_CSV_PATH = 'dataset/test.csv'
OUTPUT_CSV_PATH = 'dataset/test_out.csv'

# Load entity_unit_map
with open('src/constants.py', 'r') as f:
    exec(f.read())

# Preprocessing function
def preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    original_count = len(data)
    
    def is_valid_sample(row):
        try:
            value_str = row['entity_value']
            entity_name = row['entity_name']
            value, unit = value_str.rsplit(' ', 1)
            if unit == 'fluid':
                value, unit = value_str.rsplit(' ', 2)[0], ' '.join(value_str.rsplit(' ', 2)[1:])
            float(value)  # Check if value can be converted to float
            return unit in entity_unit_map.get(entity_name, [])
        except:
            return False

    filtered_data = data[data.apply(is_valid_sample, axis=1)]
    filtered_count = len(filtered_data)
    
    print(f"Original sample count: {original_count}")
    print(f"Filtered sample count: {filtered_count}")
    print(f"Removed {original_count - filtered_count} samples ({(original_count - filtered_count) / original_count:.2%})")
    
    return filtered_data

# Custom dataset
class ProductImageDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.entity_names = sorted(self.data['entity_name'].unique())
        self.entity_name_to_idx = {name: idx for idx, name in enumerate(self.entity_names)}
        self.entity_units = {entity: sorted(units) for entity, units in entity_unit_map.items()}
        self.max_units = max(len(units) for units in self.entity_units.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.data.iloc[idx]['image_link'])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        entity_name = self.data.iloc[idx]['entity_name']
        entity_value, entity_unit = self.parse_entity_value(self.data.iloc[idx]['entity_value'])
        
        entity_name_tensor = torch.zeros(len(self.entity_names), dtype=torch.float32)
        entity_name_tensor[self.entity_name_to_idx[entity_name]] = 1
        
        entity_unit_tensor = torch.zeros(self.max_units, dtype=torch.float32)
        unit_idx = self.entity_units[entity_name].index(entity_unit)
        entity_unit_tensor[unit_idx] = 1
        
        return img, entity_name_tensor, torch.tensor(entity_value, dtype=torch.float32), entity_unit_tensor, entity_name

    def parse_entity_value(self, value_str):
        value, unit = value_str.rsplit(' ', 1)
        if unit == 'fluid':
            value, unit = value_str.rsplit(' ', 2)[0], ' '.join(value_str.rsplit(' ', 2)[1:])
        return float(value), unit

# Custom model
class ProductImageModel(nn.Module):
    def __init__(self, num_entity_names, max_units):
        super(ProductImageModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.fc_entity_name = nn.Linear(num_ftrs, num_entity_names).float()
        self.fc_entity_value = nn.Linear(num_ftrs, 1).float()
        self.fc_entity_units = nn.Linear(num_ftrs, max_units).float()

    def forward(self, x):
        features = self.resnet(x)
        entity_name_out = self.fc_entity_name(features)
        entity_value_out = self.fc_entity_value(features).squeeze(1)
        entity_units_out = self.fc_entity_units(features)
        return entity_name_out, entity_value_out, entity_units_out

# Training function
def train_model():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess and load data
    filtered_data = preprocess_data(TRAIN_CSV_PATH)
    train_dataset = ProductImageDataset(filtered_data, TRAIN_IMAGES_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model = ProductImageModel(len(train_dataset.entity_names), train_dataset.max_units)
    model.to(DEVICE)

    # Loss and optimizer
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for i, (images, entity_names, entity_values, entity_units, entity_name_str) in enumerate(progress_bar):
            images = images.to(DEVICE)
            entity_names = entity_names.to(DEVICE)
            entity_values = entity_values.to(DEVICE)
            entity_units = entity_units.to(DEVICE)
            
            # Mixed precision training
            with autocast():
                # Forward pass
                entity_name_out, entity_value_out, entity_units_out = model(images)
                
                # Compute losses
                loss_entity_name = criterion_classification(entity_name_out, entity_names)
                loss_entity_value = criterion_regression(entity_value_out, entity_values)
                loss_entity_units = criterion_classification(entity_units_out, entity_units)
                
                loss = loss_entity_name + loss_entity_value + loss_entity_units
                loss = loss / ACCUMULATION_STEPS  # Normalize the loss

            # Backward and optimize
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': loss.item() * ACCUMULATION_STEPS})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}')
        
        # Update learning rate
        scheduler.step(avg_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1}")

    # Save the final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, MODEL_PATH)

    print("Training completed. Final model saved.")

# Prediction function
@torch.no_grad()
def predict(model, image, train_dataset):
    model.eval()
    with autocast():
        entity_name_out, entity_value_out, entity_units_out = model(image.unsqueeze(0).to(DEVICE))
        
        predicted_entity = train_dataset.entity_names[torch.argmax(entity_name_out).item()]
        predicted_value = entity_value_out.item()
        predicted_unit_idx = torch.argmax(entity_units_out).item()
        predicted_unit = train_dataset.entity_units[predicted_entity][predicted_unit_idx] if predicted_unit_idx < len(train_dataset.entity_units[predicted_entity]) else 'unknown'
        
        return predicted_entity, predicted_value, predicted_unit

# Predictor function for the sample code
def predictor(image_link, category_id, entity_name, model, train_dataset, transform):
    try:
        img = Image.open(image_link).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        predicted_entity, predicted_value, predicted_unit = predict(model, img_tensor, train_dataset)
        formatted_prediction = f"{predicted_value:.2f} {predicted_unit}"
        return formatted_prediction
    except Exception as e:
        print(f"Error processing image {image_link}: {str(e)}")
        return ""

# Main function
def main():
    # Train the model
    train_model()

    # Load the trained model for prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ProductImageDataset(pd.read_csv(TRAIN_CSV_PATH), TRAIN_IMAGES_DIR)
    model = ProductImageModel(len(train_dataset.entity_names), train_dataset.max_units)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Make predictions on the test set
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df['prediction'] = test_df.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model, train_dataset, transform),
        axis=1
    )

    # Save predictions
    test_df[['index', 'prediction']].to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()