import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from .mel_spectrogram import MelspectrogramStretch
from torchparse import parse_cfg


class AudioCRNN(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1
        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None,
                                num_mels=128,
                                fft_length=2048,
                                norm='whiten',
                                stretch_param=[0.4, 0.4],
                                random_crop=True)

        # shape -> (channel, freq, token_time)
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        for name, layer in self.net['convs'].named_children():
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride])
                lengths = ((lengths + 2*p - k)//s + 1).long()
        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs
        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)
        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)
        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)
        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)
        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward( x )
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()
            return self.classes[max_ind], out[:,max_ind].item()


class AudioCNN(AudioCRNN):
    def forward(self, batch):
        x, _, _ = batch
        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x = self.spec(x)
        # (batch, channel, freq, time)
        x = self.net['convs'](x)
        # x -> (batch, time*freq*channel)
        x = x.view(x.size(0), -1)
        # (batch, classes)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x


class AudioRNN(AudioCRNN):

    def forward(self, batch):
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs
        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)
        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)
        # x -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)
        x = F.log_softmax(x, dim=1)
        return x

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        # changed kernel size
        # we don't know why, but this is better than (3, 3) kernels.
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Spatial and Channel Squeeze and Exitation
        self.scse = SCse(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, inp, pool_size=(3, 3), pool_type="avg"):
        x = inp
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.scse(self.bn2(self.conv2(x))))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        # Added 'both' pool.
        elif pool_type == "both":
            x1 = F.max_pool2d(x, kernel_size=pool_size)
            x2 = F.avg_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            import pdb
            pdb.set_trace()
        return x


class ConvNet(BaseModel):
    def __init__(self, classes, config={}, state_dict=None):
        super(ConvNet, self).__init__()
        self.spec = MelspectrogramStretch(hop_length=None,
            num_mels=128,
            fft_length=2048,
            norm='whiten',
            stretch_param=[0.4, 0.4],
            random_crop=False)
        self.classes = classes
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)

        self.bn1 = nn.BatchNorm1d(4608) # 5120
        self.drop1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4608, 512) # 5120
        self.prelu = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, len(self.classes))

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, batch):
        x, lengths, _ = batch # unpacking seqs, lengths and srs
        xt = x.float().transpose(1,2)
        x, lengths = self.spec(xt, lengths)
        # Changed pooling policy.
        x = self.conv1(x, pool_size=(1, 1), pool_type="both")
        x = self.conv2(x, pool_size=(4, 1), pool_type="both")
        x = self.conv3(x, pool_size=(1, 3), pool_type="both")
        x = self.conv4(x, pool_size=(4, 1), pool_type="both")
        x = self.conv5(x, pool_size=(1, 3), pool_type="both")

        # Cutting the feature map to arbitrary size.
        x1_max = F.max_pool2d(x, (2, 4)) # 5, 8
        x1_mean = F.avg_pool2d(x, (2, 4)) # 5, 8
        x1 = (x1_max + x1_mean).reshape(x.size(0), -1)

        x2_max = F.max_pool2d(x, (2, 4))
        x2_mean = F.avg_pool2d(x, (2, 4))
        x2 = (x2_max + x2_mean).reshape(x.size(0), -1)

        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.drop1(self.bn1(x))
        x = self.prelu(self.fc1(x))
        x = self.drop2(self.bn2(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward(x)
            max_ind = out_raw.argmax().item()
            return self.classes[max_ind], out_raw[:,max_ind].item()