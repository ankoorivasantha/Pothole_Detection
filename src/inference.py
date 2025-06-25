import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import HybridConvTransformer
import time

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridConvTransformer().to(device)
try:
    model.load_state_dict(torch.load("models/pothole_hybrid_model.pth", weights_only=True))
except FileNotFoundError:
    print("Error: Model file not found. Please train the model first.")
    exit(1)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Real-time detection
def real_time_detection():
    # Try webcam with DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to 1 or -1 if 0 fails
    if not cap.isOpened():
        print("Error: Could not open webcam. Trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check device connection.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            # Process frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                confidence = torch.sigmoid(output).item()
            label = f"Pothole: {confidence:.2f}" if confidence > 0.5 else "No Pothole"
            color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # Save pothole images
            if confidence > 0.7:
                cv2.imwrite(f"pothole_{int(time.time())}.jpg", frame)
            cv2.imshow("Pothole Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()