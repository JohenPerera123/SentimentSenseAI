import sys

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("Running on CPU. Training will be very slow.")
except ImportError:
    print("PyTorch is not installed. Run `pip install torch`")
    sys.exit(1)

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Transformers Version: {transformers.__version__}")
    
    # Test loading tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"Attempting to load tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")
    
    print(f"Attempting to load model '{model_name}' (this might take a moment to download weights)...")
    # Load with 2 labels for our sentiment task
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("Model loaded successfully.")
    
except ImportError:
    print("Transformers is not installed. Run `pip install transformers`")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading transformers: {e}")
    sys.exit(1)

print("\nEnvironment check passed successfully!")
