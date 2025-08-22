import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline, Dinov2Model, Dinov2Config
from transformers.image_utils import load_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DINOv3FeatureExtractor:
    """
    DINOv3 Feature Extractor wrapper for easy feature extraction
    """
    def __init__(self, model_name="facebook/dinov2-vitb14"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the DINOv3 model
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """
        Extract features from a single image
        Args:
            image: PIL Image or path to image
        Returns:
            features: torch.Tensor of shape (1, feature_dim)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Use the [CLS] token representation
            features = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
        
        return features

class ClassifierNetwork(nn.Module):
    """
    Feature Classifier Network that takes DINOv3 features as input
    """
    def __init__(self, feature_dim=768, num_classes=10, hidden_dims=[512, 256]):
        super(ClassifierNetwork, self).__init__()
        
        layers = []
        input_dim = feature_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Forward pass through classifier
        Args:
            features: torch.Tensor of shape (batch_size, feature_dim)
        Returns:
            logits: torch.Tensor of shape (batch_size, num_classes)
        """
        return self.classifier(features)

class DINOv3Classifier:
    """
    Complete DINOv3 + Classifier pipeline
    """
    def __init__(self, num_classes, dinov3_model="facebook/dinov2-vitb14", 
                 hidden_dims=[512, 256], freeze_backbone=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize feature extractor
        self.feature_extractor = DINOv3FeatureExtractor(dinov3_model)
        
        # Get feature dimension from DINOv3 model
        feature_dim = self.feature_extractor.model.config.hidden_size
        
        # Initialize classifier
        self.classifier = ClassifierNetwork(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            # Freeze DINOv3 parameters
            for param in self.feature_extractor.model.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Complete forward pass: feature extraction + classification
        """
        # Extract features
        features = []
        for image in images:
            feat = self.feature_extractor.extract_features(image)
            features.append(feat)
        
        features = torch.cat(features, dim=0)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def train_classifier(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        """
        Train the classifier network
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                labels = labels.to(self.device)
                
                # Extract features
                features = []
                for image in images:
                    feat = self.feature_extractor.extract_features(image)
                    features.append(feat)
                features = torch.cat(features, dim=0)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            val_acc = self.evaluate(val_loader)
            
            # Update learning rate
            scheduler.step()
            
            # Print progress
            train_acc = 100. * train_correct / train_total
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Acc: {val_acc:.2f}%')
            print('-' * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.classifier.state_dict(), 'best_classifier.pth')
    
    def evaluate(self, data_loader):
        """
        Evaluate the classifier
        """
        self.classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                labels = labels.to(self.device)
                
                # Extract features
                features = []
                for image in images:
                    feat = self.feature_extractor.extract_features(image)
                    features.append(feat)
                features = torch.cat(features, dim=0)
                
                # Classify
                outputs = self.classifier(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def predict(self, image):
        """
        Predict class for a single image
        """
        self.classifier.eval()
        
        with torch.no_grad():
            features = self.feature_extractor.extract_features(image)
            logits = self.classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1)
        
        return predicted_class.item(), probabilities.squeeze().cpu().numpy()

# Example usage and demonstration
def example_usage():
    """
    Example of how to use the DINOv3 + Classifier pipeline
    """
    
    # 1. Initialize the complete pipeline
    num_classes = 10  # e.g., for CIFAR-10
    classifier_pipeline = DINOv3Classifier(
        num_classes=num_classes,
        dinov3_model="facebook/dinov2-vitb14",
        hidden_dims=[512, 256],
        freeze_backbone=True
    )
    
    # 2. Load and test with a single image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = load_image(url)
    
    # Extract features only
    features = classifier_pipeline.feature_extractor.extract_features(image)
    print(f"Extracted features shape: {features.shape}")
    
    # Make prediction (random since not trained)
    predicted_class, probabilities = classifier_pipeline.predict(image)
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}")
    
    print("\n" + "="*60)
    print("IMPLEMENTATION GUIDE:")
    print("="*60)
    
    print("""
    To add a feature classifier network following DINOv3 feature extraction:
    
    1. FEATURE EXTRACTION:
       - Use DINOv3 as a frozen backbone to extract rich visual features
       - Features are typically 768-dimensional vectors (for ViT-B/14)
       - The [CLS] token provides a global image representation
    
    2. CLASSIFIER NETWORK DESIGN:
       - Input: DINOv3 features (768-dim for ViT-B/14)
       - Hidden layers: Fully connected layers with ReLU, Dropout, BatchNorm
       - Output: Number of classes for your specific task
    
    3. TRAINING STRATEGY:
       - Freeze DINOv3 backbone (recommended for most cases)
       - Train only the classifier head
       - Use standard classification loss (CrossEntropy)
       - Apply data augmentation on images before feature extraction
    
    4. INTEGRATION STEPS:
       a) Load pre-trained DINOv3 model
       b) Extract features from your dataset
       c) Design classifier architecture
       d) Train classifier on extracted features
       e) Evaluate and fine-tune
    
    5. ADVANTAGES:
       - Leverages powerful pre-trained representations
       - Faster training (only classifier needs training)
       - Good performance even with limited data
       - Can be easily adapted to different tasks
    
    6. CUSTOMIZATION OPTIONS:
       - Adjust hidden layer dimensions
       - Modify dropout rates
       - Change activation functions
       - Add regularization techniques
       - Fine-tune the entire model (unfreeze backbone)
    """)

if __name__ == "__main__":
    example_usage()
