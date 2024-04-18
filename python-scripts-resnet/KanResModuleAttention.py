import torch.nn.functional as F
from torch import nn
import torch

# done
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


#done
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


#done
class ChannelGate(nn.Module):


    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types


    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d( x, (x.size(2)), stride=(x.size(2)))
                channel_att_raw = self.mlp( avg_pool )
                # q: explain channel_att_raw = self.mlp( avg_pool )
                # a: channel_att_raw is the output of the mlp layer, which is a linear layer with a ReLU activation function
            elif pool_type=='max':
                max_pool = F.max_pool1d( x, (x.size(2)), stride=(x.size(2)))
                channel_att_raw = self.mlp( max_pool )
            #elif pool_type=='lp':
             #   lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
              #  channel_att_raw = self.mlp( lp_pool )
            #elif pool_type=='lse':
                # LSE pool only
             #   lse_pool = logsumexp_2d(x)
              #  channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        return x * scale
    
#done
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

        
# tune according to the input
class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 50
        self.compress = ChannelPool()
        #self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding='same', relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class KanResModuleAttention(nn.Module):
    def __init__(self, in_channels, filterno_1, filterno_2, filtersize_1, filtersize_2, stride):
        super(KanResModuleAttention, self).__init__()
        # have to use same padding to keep the size of the input and output the same
        # calculate the padding needed for same
        padding = (filtersize_1 - 1) // 2 + (stride - 1)
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride, padding='same')
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2, padding='same')
        self.bn2 = nn.BatchNorm1d(filterno_2)
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.5)
        self.cbam = CBAM(gate_channels=filterno_2, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)

    
        
    def forward(self, x):
        identity = x
        #print(x.shape)      
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.cbam(x)
        x = self.dropout(x)
        x = x + identity
        return x