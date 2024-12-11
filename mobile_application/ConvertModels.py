import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import timm

# --- Step 1: Load Pretrained RexNet Model ---
class MultiTaskRexNet(nn.Module):
    def __init__(self, backbone):
        super(MultiTaskRexNet, self).__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.age_head = nn.Linear(1920, 101)       # Predict age (0–100)
        self.gender_head = nn.Linear(1920, 2)     # Predict gender (Male, Female)
        self.ethnicity_head = nn.Linear(1920, 5)  # Predict ethnicity (5 classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.global_pool(features).view(features.size(0), -1)
        age_pred = self.age_head(features)
        gender_pred = self.gender_head(features)
        ethnicity_pred = self.ethnicity_head(features)
        return age_pred, gender_pred, ethnicity_pred

# Load pretrained RexNet backbone
def load_pretrained_model(file_path):
    model = timm.create_model('rexnet_150', pretrained=False)
    state_dict = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return MultiTaskRexNet(model)

# Step 1: Load the pre-trained PyTorch model
model_path = '/Users/michaelguel/Downloads/multi_task_rexnet_age_gender_ethnicity.pth' 
base_model = load_pretrained_model(model_path) # Replace with your actual model path
# base_model.load_state_dict(model_path, map_location='cpu')
base_model.eval()

# Step 2: Create a Wrapper Model for Output Separation
class EmotionRecognitionModelWithScores(nn.Module):
    def __init__(self, model):
        super(EmotionRecognitionModelWithScores, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass through the base model
        age, gender, ethnicity = self.model(x)


        return age, gender, ethnicity  # Return as separate outputs

# Step 3: Create a Preprocessing Module
class PreprocessingModule(nn.Module):
    def __init__(self, model):
        super(PreprocessingModule, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = x / 255.0  # Scale to [0, 1]
        x = self.resize(x)  # Resize to (1, 3, 224, 224)
        x = (x - self.mean) / self.std  # Normalize
        return self.model(x)  # Pass to wrapped model

# Wrap the base model
wrapped_model = EmotionRecognitionModelWithScores(base_model)

# Wrap the model with preprocessing
preprocessed_model = PreprocessingModule(wrapped_model)
preprocessed_model.eval()

# Step 4: Trace the model
dummy_input = torch.randn(1, 3, 224, 224) * 255.0
with torch.no_grad():
    traced_model = torch.jit.trace(preprocessed_model, dummy_input)

# Step 5: Define input type
input_shape = (1, 3, 224, 224)
input_type = ct.ImageType(
    name="input",
    shape=input_shape,
    scale=1.0,
    bias=[0.0, 0.0, 0.0],
    color_layout='RGB'
)

# Define output types for emotion scores, valence, and arousal
output_emotion_scores = ct.TensorType(
    name="age",
    dtype=np.float32  # Remove shape specification
)

output_valence = ct.TensorType(
    name="gender",
    dtype=np.float32  # Remove shape specification
)

output_arousal = ct.TensorType(
    name="ethnicity",
    dtype=np.float32  # Remove shape specification
)

# Convert to Core ML format with explicit outputs
mlmodel = ct.convert(
    traced_model,
    inputs=[input_type],
    outputs=[output_emotion_scores, output_valence, output_arousal],
    minimum_deployment_target=ct.target.iOS17
)

# Step 7: Save the Core ML model
mlmodel.save('/Users/michaelguel/Desktop/EmotionRecognition2.mlpackage')