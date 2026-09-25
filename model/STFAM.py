import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Global_1D_CNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super(Global_1D_CNN_Autoencoder, self).__init__()
        self.enc_conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16) 
        self.v_linear = nn.Linear(128 * 16, embed_dim)

        self.d_linear = nn.Linear(embed_dim, 128 * 16)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dec_conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.dec_conv2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=2, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc_conv1(x))
        e1 = self.pool(e1)
        e2 = F.relu(self.enc_conv2(e1))
        e2_pooled = self.adaptive_pool(e2)
        e2_flat = e2_pooled.view(e2_pooled.size(0), -1)
        P_embed = self.v_linear(e2_flat)

        d0 = F.relu(self.d_linear(P_embed)).view(-1, 128, 16)
        d1 = self.upsample1(d0)
        d1 = F.relu(self.dec_conv1(d1))
        d2 = self.upsample2(d1)
        out = torch.sigmoid(self.dec_conv2(d2))
        
        out = F.interpolate(out, size=(x.size(2),), mode='linear', align_corners=False)
        return P_embed, out

class Global_2D_CNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super(Global_2D_CNN_Autoencoder, self).__init__()
        # 真正使用 input_dim 以支持双通道
        self.enc_conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.v_linear = nn.Linear(128 * 4 * 4, embed_dim)

        self.d_linear = nn.Linear(embed_dim, 128 * 4 * 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dec_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        # 真正使用 input_dim
        self.dec_conv2 = nn.Conv2d(in_channels=64, out_channels=input_dim, kernel_size=6, padding=2)

    def forward(self, x):
        e1 = self.pool(F.relu(self.enc_conv1(x)))
        e2 = self.pool(F.relu(self.enc_conv2(e1)))
        e2_pooled = self.adaptive_pool(e2)
        e2_flat = e2_pooled.view(e2_pooled.size(0), -1)
        V_embed = self.v_linear(e2_flat) 

        d0 = F.relu(self.d_linear(V_embed)).view(-1, 128, 4, 4)
        d1 = self.upsample1(d0)
        d1 = F.relu(self.dec_conv1(d1))
        d2 = self.upsample2(d1)
        out = torch.sigmoid(self.dec_conv2(d2))
        
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return V_embed, out

class Local_LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embed_dim=64):
        super(Local_LSTM_Autoencoder, self).__init__()
        self.enc_lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.enc_lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.enc_linear = nn.Linear(hidden_dim, embed_dim)
        
        self.dec_lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec_lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec_linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h1, _ = self.enc_lstm1(x)
        h2, (h_n, c_n) = self.enc_lstm2(h1)
        last_h2 = h2[:, -1, :] 
        R_Tr = self.enc_linear(last_h2)

        seq_len = x.size(1)
        d_in = R_Tr.unsqueeze(1).repeat(1, seq_len, 1)
        dh1, _ = self.dec_lstm1(d_in)
        dh2, _ = self.dec_lstm2(dh1)
        out = self.dec_linear(dh2)
        
        return R_Tr, out

class STFAM_Data_Fusion(nn.Module):
    def __init__(self, embed_dim):
        super(STFAM_Data_Fusion, self).__init__()
        self.embed_dim = embed_dim
        self.poi_trans = nn.Linear(embed_dim, embed_dim)
        self.W_R = nn.Linear(embed_dim * 2, embed_dim)
        self.W_V = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, R_Tr, V_U, V_D, P_embed):
        # Feature-wise Attention 修复
        P_trans = self.poi_trans(P_embed) 
        scores = (P_trans * R_Tr) / math.sqrt(self.embed_dim) 
        attn_weights = F.softmax(scores, dim=1) 
        
        attn_out = attn_weights * R_Tr
        R_concat = torch.cat([attn_out, R_Tr], dim=1)
        R = self.W_R(R_concat)
        
        V_UD = (V_U + V_D) / 2.0
        V_concat = torch.cat([V_UD, P_embed], dim=1)
        V = self.W_V(V_concat)
        return R, V

class STFAM_Predictor(nn.Module):
    def __init__(self, embed_dim, num_local_nodes, pred_steps, out_channels):
        super(STFAM_Predictor, self).__init__()
        self.num_local_nodes = num_local_nodes
        self.pred_steps = pred_steps
        self.out_channels = out_channels

        self.shared_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.horizon_heads = nn.ModuleList(
            nn.Linear(embed_dim, num_local_nodes * out_channels)
            for _ in range(pred_steps)
        )

    def forward(self, R, V):
        x = torch.cat([R, V], dim=1)
        shared_repr = F.relu(self.shared_fc(x))
        horizon_outputs = [head(shared_repr) for head in self.horizon_heads]
        return torch.stack(horizon_outputs, dim=1)

class STFAM_Client_Model(nn.Module):
    def __init__(self, t_in, num_local_nodes, embed_dim, in_channels, pred_steps, out_channels):
        super(STFAM_Client_Model, self).__init__()
        self.num_local_nodes = num_local_nodes
        self.pred_steps = pred_steps
        self.out_channels = out_channels
        
        # 显式支持 C 通道计算
        # Compact temporal representation: raw flow plus adjacency-aggregated flow per node.
        # This avoids the dense N*N transition tensor that exhausts GPU memory on TaxiBJ.
        lstm_input_dim = num_local_nodes * in_channels * 2
        self.local_lstm = Local_LSTM_Autoencoder(input_dim=lstm_input_dim, hidden_dim=embed_dim, embed_dim=embed_dim)
        self.local_cnn = Global_2D_CNN_Autoencoder(input_dim=in_channels, embed_dim=embed_dim) 
        self.fusion = STFAM_Data_Fusion(embed_dim)
        
        self.predictor = STFAM_Predictor(
            embed_dim=embed_dim,
            num_local_nodes=num_local_nodes,
            pred_steps=pred_steps,
            out_channels=out_channels,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, Tr, U, V_D, P_embed):
        R_Tr, Tr_recon = self.local_lstm(Tr)
        V_U, U_recon = self.local_cnn(U)
        
        batch_size = R_Tr.size(0)
        V_D_exp = V_D.expand(batch_size, -1) if V_D.size(0) == 1 and batch_size > 1 else V_D
        P_embed_exp = P_embed.expand(batch_size, -1) if P_embed.size(0) == 1 and batch_size > 1 else P_embed

        R, V = self.fusion(R_Tr, V_U, V_D_exp, P_embed_exp)
        out = self.predictor(R, V)
        
        # 直接输出形状完全匹配的目标张量，拒绝随缘 reshape
        out = out.view(batch_size, self.pred_steps, self.num_local_nodes, self.out_channels)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out, Tr_recon, U_recon

    def compute_loss(self, Tr, U, y, V_D, P_embed, alpha=0.1, beta=0.1):
        y_pred, Tr_recon, U_recon = self.forward(Tr, U, V_D, P_embed)
        if y_pred.shape != y.shape:
            y_pred = y_pred.view_as(y)
            
        loss_pred = self.loss_fn(y_pred, y)
        loss_recon_Tr = self.loss_fn(Tr_recon, Tr)
        loss_recon_U = self.loss_fn(U_recon, U)
        return loss_pred + alpha * loss_recon_Tr + beta * loss_recon_U
