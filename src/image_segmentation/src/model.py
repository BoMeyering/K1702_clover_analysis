from transformers import SegformerForSemanticSegmentation

def create_model(num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b5',
        num_labels=num_classes,
    )
    return model
