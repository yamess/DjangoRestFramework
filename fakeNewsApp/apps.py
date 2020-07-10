from django.apps import AppConfig
from .fakenews.model import BertClassifier
from transformers import BertTokenizerFast, BertTokenizer
from django.conf import settings
from onnxruntime import InferenceSession, ExecutionMode, SessionOptions

class FakenewsappConfig(AppConfig):
    name = 'fakeNewsApp'
    # Load the pretrained tokenizer
    # tokenizer = BertTokenizer.from_pretrained(settings.TOKENIZER_ROOT)

    # Instanciation of the model
    # model = BertClassifier(
    #     bert_path=settings.BERT_BASE_MODEL_PATH,
    #     dropout=settings.DROPOUT,
    #     n_class=settings.N_CLASS
    # )

    # Load the pretrained model
    # state = torch.load(settings.MODEL_ROOT)
    # model.load_state_dict(state)
    # model.eval()
    # print("MODEL RELOADED")

    # Onnix Model
    # Loading te tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(settings.TOKENIZER_ROOT)

    # Instanciating the onnx runtime class
    options = SessionOptions()
    options.inter_op_num_threads = 4
    options.execution_mode = ExecutionMode.ORT_PARALLEL
    session = InferenceSession(settings.MODEL_ROOT)
    print("Model Loaded with success...")



