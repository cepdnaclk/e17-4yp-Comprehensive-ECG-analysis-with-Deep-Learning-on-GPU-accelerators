from ECGDataSet import ECGDataSet
from VisionTransformer import VisionTransformer
from TransformerHelper import TransformerHelper
from torch.utils.data import DataLoader

# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
input_shape = (8, 5000)  # Modify this according to your input shape
# Number of output units
output_size = 1
# number of epochs
number_of_epochs = 1000
#
learning_rate = 0.0005
#
y_parameters = ["hr"]


for y_parameter in y_parameters:
    # ECG dataset
    train_dataset = ECGDataSet(split="train")
    validate_dataset = ECGDataSet(split="validate")

    # data loaders
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=True, num_workers=20
    )
    validate_dataloader = DataLoader(
        dataset=validate_dataset, batch_size=16, shuffle=False, num_workers=20
    )

    # model
    model = VisionTransformer(
        img_size=5000,
        patch_size=50,
        in_chans=8,
        n_classes=1,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    )

    # train and validate
    resnet = TransformerHelper(model, learning_rate, number_of_epochs, y_parameter)
    resnet.train_and_validate(train_dataloader, validate_dataloader, y_parameter)
