from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from django.conf import settings
import torch

def preprocess(title, text, tokenizer):
    
    inputs = tokenizer.encode_plus(
        str(title) + ". " + str(text),
        add_special_tokens=True,
        max_length=settings.MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_attention_mask=True
    )
    
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(mask, dtype=torch.long).unsqueeze(0),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    }
    
    return out

# def predict(inputs, model):
#     """
#     1: Means Real news
#     0: Means Fake News
#     """
#     model.eval()
#     with torch.no_grad():
#         logits = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             token_type_ids=inputs["token_type_ids"]
#         )
#         prob = torch.sigmoid(logits).cpu().detach().numpy().item()
#         y_pred = int(prob > 0.5)
#         logits = logits.cpu().detach().numpy().item()
#         y_pred_text = "Real News" if y_pred == 0 else "True News"
#
#         predictions = {
#             "logits": logits,
#             "predictionValue": y_pred,
#             "predictionText": y_pred_text,
#             "predictionProbability": prob
#         }
#         return predictions

def onnx_predict(inputs, session):
    """
    1: Means Real news
    0: Means Fake News
    """
    logits = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].detach().cpu().numpy(),
            "attention_mask": inputs["attention_mask"].detach().cpu().numpy(),
            "token_type_ids": inputs["token_type_ids"].detach().cpu().numpy()
        }
    )
    logits = torch.tensor(logits).flatten()

    prob = torch.sigmoid(logits).item()
    y_pred = int(prob > 0.5)
    y_pred_text = "Fake News" if y_pred == 0 else "Real News"
    logits = logits.item()

    predictions = {
        "logits": logits,
        "predictedValue": y_pred,
        "predictedText": y_pred_text,
        "predictedProb": prob
    }
    return predictions
