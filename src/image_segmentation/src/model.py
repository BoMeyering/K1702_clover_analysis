from transformers import SegformerForSemanticSegmentation



def create_model(num_classes: int, variant: str='nvidia/mit-b0'):
    variants = [
        'nvidia/mit-b0', 
        'nvidia/mit-b1', 
        'nvidia/mit-b2', 
        'nvidia/mit-b3', 
        'nvidia/mit-b4', 
        'nvidia/mit-b5'
    ]
    if variant in variants:
        model = SegformerForSemanticSegmentation.from_pretrained(
            variant,
            num_labels=num_classes,
        )
        return model
    else:
        raise ValueError(f"Arg 'variant' must be one of {variants}")
