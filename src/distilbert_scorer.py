import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Your HuggingFace model repo
HF_MODEL_PATH = "asanginov/resume-screener-distilbert"

def load_distilbert(model_path=HF_MODEL_PATH):
    """Load fine-tuned DistilBERT from HuggingFace Hub"""
    print(f"Loading DistilBERT from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("DistilBERT loaded successfully!")
    return model, tokenizer, device

def predict_with_distilbert(resume_text, jd_text, model, tokenizer, device):
    """
    Predict fit probability using fine-tuned DistilBERT
    Returns probability of being a fit (0.0 to 1.0)
    """
    encoding = tokenizer(
        str(resume_text)[:512],
        str(jd_text)[:256],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        fit_probability = probs[0][1].item()

    return fit_probability