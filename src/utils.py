import torch
from typing import List,  Union

def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    """
    Casts the training parameters of the model to the specified data type.

    Args:
        model: The PyTorch model whose parameters will be cast.
        dtype: The data type to which the model parameters will be cast.
    """
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)          

def check_parameter_dtypes(pipe):
    print('\nCHECK FOR PARAMETER DATA TYPES\n')
    # List of component names to check
    components_to_check = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",
        "vae",
        "image_encoder",
        "scheduler",
        "feature_extractor"
    ]

    print("Data types per component:\n")

    for name in components_to_check:
        component = getattr(pipe, name, None)
        if component is None:
            continue

        params = list(component.parameters()) if hasattr(component, "parameters") else []
        dtype_counts = {}

        for param in params:
            dtype = param.dtype
            if dtype not in dtype_counts:
                dtype_counts[dtype] = 0
            dtype_counts[dtype] += 1

        if dtype_counts:
            print(f"{name}:")
            for dtype, count in dtype_counts.items():
                print(f"    {dtype}: {count} parameters")

    print("\nCheck if multiple data types are used in the same module:")
    for name, module in pipe.components.items():
        if hasattr(module, "parameters"):
            dtypes = set(param.dtype for param in module.parameters())
            if len(dtypes) > 1:
                print(f"    `{name}` has parameters with multiple data types: {dtypes}")
            else:
                print(f"✅ `{name}` has parameters with a single data type: {dtypes}")

def check_learnable_parameters(pipe):
    print('\nCHECK FOR LEARNABLE PARAMETERS\n')
    # List of component names to check
    components_to_check = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",
        "vae",
        "image_encoder",
        "scheduler",
        "feature_extractor"
    ]

    print("Learnable parameters per component:\n")

    for name in components_to_check:
        component = getattr(pipe, name, None)
        if component is None:
            continue

        params = list(component.parameters()) if hasattr(component, "parameters") else []
        learnable = [p for p in params if p.requires_grad]

        print(f"{name}: {len(learnable)} learnable parameters")

        # Optional: print total count
        total_params = sum(p.numel() for p in learnable)
        if total_params > 0:
            print(f"    Total learnable parameters: {total_params:,}")


    print(" check if some parameters is set to true_grad")
    has_grad = False

    for name, module in pipe.components.items():
        if hasattr(module, "parameters"):
            for param in module.parameters():
                if param.requires_grad:
                    print(f"✅ `{name}` has parameters with `requires_grad=True`")
                    has_grad = True
                    break  # no need to keep checking this module

    if not has_grad:
        print("❌ No modules have trainable (requires_grad=True) parameters.")

def freez_learnable_parameters(pipe):
    print('\nFREEZ LEARNABLE PARAMETERS\n')
    # Freeze parameters for specific components
    components_to_freeze = ["vae", "text_encoder", "text_encoder_2", "unet"]

    for name in components_to_freeze:
        component = getattr(pipe, name, None)
        if component is not None:
            for param in component.parameters():
                param.requires_grad = False

    # Verify that parameters are frozen
    print("Learnable parameters per component:\n")
    for name in components_to_freeze:
        component = getattr(pipe, name, None)
        if component is None:
            continue

        params = list(component.parameters()) if hasattr(component, "parameters") else []
        learnable = [p for p in params if p.requires_grad]

        print(f"{name}: {len(learnable)} learnable parameters")



