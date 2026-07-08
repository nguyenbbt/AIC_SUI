import os
import urllib.request
import ssl

def download_weights():
    os.makedirs("weights", exist_ok=True)
    weights_path = "weights/transnetv2-pytorch-weights.pth"
    
    if os.path.exists(weights_path):
        print(f"Weights already exist at {weights_path}")
        return
        
    try:
        import transnetv2_pytorch
        import shutil
        
        # Get path to the weights inside the pip package
        package_dir = os.path.dirname(transnetv2_pytorch.__file__)
        package_weights = os.path.join(package_dir, "transnetv2-pytorch-weights.pth")
        
        if os.path.exists(package_weights):
            print(f"Found built-in weights at: {package_weights}")
            print(f"Copying to {weights_path}...")
            shutil.copy2(package_weights, weights_path)
            print("Copy complete! You are ready for offline usage.")
        else:
            print(f"Error: Could not find weights inside package at {package_weights}")
            
    except ImportError:
        print("Error: transnetv2-pytorch is not installed. Please run: pip install transnetv2-pytorch")
    except Exception as e:
        print(f"Failed to extract weights: {e}")

if __name__ == "__main__":
    download_weights()
