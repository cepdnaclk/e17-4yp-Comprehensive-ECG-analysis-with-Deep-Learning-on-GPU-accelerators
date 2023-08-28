import torch.nn.functional as F
from KanResInit import KanResInit
from KanResModule import KanResModule
from torch import nn

class KanResWide_X2(nn.Module):
    def __init__(self, input_shape, output_size):
        super(KanResWide_X2, self).__init__()

        #print(input_shape[0])
        #print(input_shape[1])

        self.input_shape = input_shape
        self.output_size = output_size
        
        self.init_block = KanResInit(input_shape[0], 64, 64, 8, 3, 1)
        self.pool = nn.AvgPool1d(kernel_size=2)
        
        self.module_blocks = nn.Sequential(
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = self.init_block(x)
        #print("init block trained")
        #print(x.shape)
        x = self.pool(x)
        #print("pool 1 trained")
        #print(x.shape)
        x = self.module_blocks(x)
        #print("module blocks trained")
        x = self.global_avg_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
