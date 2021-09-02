import torch
import torch.nn as nn
from .layers import _conv_2d_layer_, _conv_2d_layer_down_
from .rnn_imp import RNN_trial


class _iFAST_model_serial_dependence_(nn.Module): 
    def __init__(self):
        super(_iFAST_model_serial_dependence_, self).__init__()

        # input 1x32x18
        self.extractor_Conv = nn.Sequential(
            _conv_2d_layer_(1, 16, 3, (1, 0)),
            _conv_2d_layer_down_(16, 16, 3, 1),  # 16x8

            _conv_2d_layer_(16, 32, 1, 0),
            _conv_2d_layer_down_(32, 32, 3, 1),  # 8x4
            _conv_2d_layer_down_(32, 64, 3, 1),  # 4x2

            _conv_2d_layer_(64, 16, 1, 0)
        )

        self.fc_desc = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU()
        )

        self.rnn = RNN_trial(input_size=1, hidden_size=4, num_layers=2, batch_first=True).cuda()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0.001)
            # elif isinstance(m, nn.RNN):
            #     nn.init.orthogonal_(m.all_weights[0][0])
            #     nn.init.orthogonal_(m.all_weights[0][1])
            #     nn.init.constant_(m.all_weights[0][2], 0.001)
            #     nn.init.constant_(m.all_weights[0][3], 0.001)
            #     nn.init.orthogonal_(m.all_weights[1][0])
            #     nn.init.orthogonal_(m.all_weights[1][1])
            #     nn.init.constant_(m.all_weights[1][2], 0.001)
            #     nn.init.constant_(m.all_weights[1][3], 0.001)
            elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.all_weights[0][0])
                nn.init.orthogonal_(m.all_weights[0][1])
                nn.init.constant_(m.all_weights[0][2], 0.001)
                nn.init.constant_(m.all_weights[0][3], 0.001)
                nn.init.orthogonal_(m.all_weights[1][0])
                nn.init.orthogonal_(m.all_weights[1][1])
                nn.init.constant_(m.all_weights[1][2], 0.001)
                nn.init.constant_(m.all_weights[1][3], 0.001)

    def forward(self, appearance, motion_content, motion_desc_simi, hidden):

        motion_desc_simi = self.extractor_Conv(motion_desc_simi)
        motion_desc = torch.mean(motion_desc_simi, dim=(2, 3))
        motion_desc = self.fc_desc(motion_desc)

        motion_desc = motion_desc.view(motion_content.shape)
        motion = torch.mean(motion_desc * motion_content, dim=2)

        x = motion.view(appearance.shape) * appearance
        x = self.rnn(x, hidden)
        return x

