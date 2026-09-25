import argparse
import os

parser = argparse.ArgumentParser()

DATASETS = ['PeMS03','PeMS04','PeMSD7','PeMS08']
# DATASETS = ['marseille', 'essen', 'hamburg', 'paris', 'groningen']
TYPES = ["flow", "speed", "occ" ]

parser.add_argument('--data_list',
                    help='name of dataset list;',
                    default=DATASETS,
                    type=str)

parser.add_argument('--dataset_name',
                    help='name of dataset;',
                    default="PeMS04",
                    type=str)

parser.add_argument('--feature_type',
                    choices=TYPES,
                    default='flow',
                    type=str)

parser.add_argument('--normalizer',
                    default='std',
                    type=str)

parser.add_argument('--scaler_fit_scope',
                    choices=['selected', 'full'],
                    default='selected',
                    help='Fit scaler on selected client nodes or full graph training split.')

parser.add_argument('--fcgcn_adj_residual_alpha',
                    default=0.9,
                    type=float,
                    help='FCGCN residual adjacency mix: alpha*I + (1-alpha)*A_hat. PeMSD7 remains identity.')

parser.add_argument('--train_ratio',
                    default=0.7,
                    type=float)

parser.add_argument('--val_ratio',
                    default=0.1,
                    type=float)

parser.add_argument('--test_ratio',
                    default=0.2,
                    type=float)

parser.add_argument('--target_train_ratio',
                    default=0.05,
                    type=float)

parser.add_argument('--target_val_ratio',
                    default=0.1,
                    type=float)

parser.add_argument('--target_test_ratio',
                    default=0.1,
                    type=float)

parser.add_argument('--target_city',
                    help='target city with scarce data;',
                    default='HK',
                    type=str)

parser.add_argument('--num_clients',
                    help='number of clients participate in FL;',
                    default=5,
                    type=int)

parser.add_argument('--cross_device_workers',
                    help=(
                        'Physical FATE worker processes for CNFGNN, REFOL, FedOSTC, '
                        'SFL, FedGODE and FedSTG. A value <= 0 reuses --num_clients. '
                        'For those methods every dataset node remains an independent '
                        'logical client; this option controls only process multiplexing.'
                    ),
                    default=0,
                    type=int)

parser.add_argument('--t_in',
                    help='history traffic step;',
                    default=12,
                    type=int)

parser.add_argument('--t_out',
                    help='predict traffic step;',
                    default=3,
                    type=int)

parser.add_argument('--epochs',
                    help='number of epochs;',
                    type=int,
                    default=5)

parser.add_argument('--target_epochs',
                    help='number of target client epochs;',
                    type=int,
                    default=100)

parser.add_argument('--local_epochs',
                    help='number of local client epochs;',
                    type=int,
                    default=1)

parser.add_argument('--hidden_dim',
                    help='hidden dim;',
                    default=32,
                    type=int)

parser.add_argument('--input_dim',
                    help='input feature dimension (e.g. 1 for PeMS, 2 for TaxiBJ);',
                    default=1,
                    type=int)

parser.add_argument('--output_dim',
                    help='output feature dimension;',
                    default=1,
                    type=int)                    

parser.add_argument('--wd',
                    help='weight decay parameter;',
                    type=float,
                    default=1e-8)

parser.add_argument('--device',
                    help='device',
                    default='cuda:0',
                    type=str)

parser.add_argument('--result_root',
                    help='optional directory for federated benchmark CSV outputs; defaults to logs/logs12 model folders',
                    default='',
                    type=str)

parser.add_argument('--partition_strategy',
                    choices=['auto', 'metis', 'grid', 'louvain', 'geographic', 'sensorid'],
                    default='auto',
                    help='FATE client-node partition. PeMS04 supports 2/4/8/16/32 clients; 32 uses a METIS-seeded topology refinement. TaxiBJ supports 4/32 rectangular grid clients.')

parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=64)

parser.add_argument('--lr',
                    help='learning rate for solver',
                    type=float,
                    default=1e-3)

parser.add_argument('--loss_func',
                    help='loss function: mse, mae, smoothl1/huber',
                    type=str,
                    default='mse')

parser.add_argument('--seed',
                    help='seed for randomness;',
                    type=int,
                    default=42)
                    
parser.add_argument('--model',
                    help='model name [ISTGNN, TwoMGTCN, ASTGAT, STGCN, UFCL_GWN, DyHSL, FedTSE, FedOSTC, STNET, FedTPS, FedGODE, TDLR, SEDLR, TDLR_SEDLR]',
                    type=str,
                    default='FedTPS') 
# ==========================================
#        Added for FedGODE Parameters
# ==========================================
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')

# ==========================================
#        闂備礁鎼崐鐟邦熆濮椻偓璺柛鎰靛枟閺咁剚绻涢崱娆樻綆dGTP Parameters
# ==========================================
parser.add_argument('--poly_k', 
                    type=int, 
                    default=4, 
                    help='FedGTP: Polynomial order K for activation decomposition')

parser.add_argument('--node_emb_dim', 
                    type=int, 
                    default=2, 
                    help='FedGTP: Dimension d of node embeddings')


parser.add_argument('--kl_threshold', type=float, default=0.0003, help='REFOL: KL divergence threshold')
parser.add_argument('--adj_mx', type=str, default='pems.pkl', help='REFOL: Adjacency matrix file name')
parser.add_argument('--refol_aggregation', type=str, default='fedavg_residual',
                    choices=['fedavg_residual', 'fedavg', 'gcn'],
                    help='REFOL aggregation. The archived legacy REFOL trainer uses pure gcn.')
parser.add_argument('--refol_gcn_residual_alpha', type=float, default=0.1,
                    help='REFOL coefficient for the GCN residual in fedavg_residual mode.')
parser.add_argument('--refol_topology', type=str, default='bidirectional',
                    choices=['bidirectional', 'legacy_directed'],
                    help='REFOL client graph topology. Archived REFOL uses bidirectional.')
parser.add_argument('--refol_legacy_rank_init', action='store_true', default=False,
                    help='REFOL historical reproduction: do not reset every FATE rank before model construction.')
parser.add_argument('--refol_no_restore_best', dest='refol_restore_best', action='store_false',
                    help='REFOL diagnostic only: test the final model instead of the best validation model.')
parser.set_defaults(refol_restore_best=True)
parser.add_argument('--refol_legacy_teacher_forcing_eval', action='store_true', default=False,
                    help='REFOL historical-reproduction only: evaluate validation/test with the true future y sequence. '
                         'This is not strict autoregressive multi-step forecasting.')

parser.add_argument('--trainer_mode', type=str, default='fedavg',
                    choices=['fedavg', 'ufcl', 'fed4tp', 'sfl', 'tistgnn_ic', 'fedmssa'],
                    help='Trainer backend mode for federated training experiments')

parser.add_argument('--fedstn_checkpoint_dir', type=str, default='',
                    help='FedSTN: directory for client best-checkpoint files. Defaults to checkpoints/FedSTN/<dataset>/<feature>/seed<seed>.')
parser.add_argument('--fedstn_resume', action='store_true', default=False,
                    help='FedSTN: warm-start each client from its saved best checkpoint when available.')

parser.add_argument('--ufcl_ablation',
                    type=str,
                    default='full',
                    choices=['full', 'no_replay', 'no_kd', 'task_only'],
                    help='UFCL ablation mode: full, no_replay, no_kd, or task_only.')
parser.add_argument('--ufcl_kd_weight', type=float, default=1.0,
                    help='UFCL: weight for teacher knowledge distillation loss.')
parser.add_argument('--ufcl_noise_std', type=float, default=0.05,
                    help='UFCL: noise std used to generate synthetic replay samples.')
parser.add_argument('--ufcl_gwn_channels', type=int, default=29,
                    help='UFCL_GWN: lightweight GraphWaveNet residual channels. Default keeps params near 38k.')
parser.add_argument('--ufcl_gwn_skip_channels', type=int, default=58,
                    help='UFCL_GWN: skip channels for the lightweight GraphWaveNet backbone.')
parser.add_argument('--ufcl_gwn_end_channels', type=int, default=58,
                    help='UFCL_GWN: end channels for the lightweight GraphWaveNet backbone.')
parser.add_argument('--ufcl_gwn_blocks', type=int, default=2,
                    help='UFCL_GWN: number of GraphWaveNet blocks.')
parser.add_argument('--ufcl_gwn_layers', type=int, default=2,
                    help='UFCL_GWN: number of dilated temporal layers in each block.')
parser.add_argument('--ufcl_gwn_dropout', type=float, default=0.1,
                    help='UFCL_GWN: dropout used in diffusion graph convolution.')

# ==========================================
#        Added for FGNNEH Parameters
# ==========================================
parser.add_argument('--fgnneh_P', type=float, default=0.05, 
                    help='FGNNEH: Cumulative contribution rate threshold for backbone (e.g. 5%)')
parser.add_argument('--fgnneh_gamma', type=float, default=0.1, 
                    help='FGNNEH: Gamma for RBF Kernel')
parser.add_argument('--fgnneh_n_components', type=int, default=8, 
                    help='FGNNEH: PCA components dimension')
parser.add_argument('--fgnneh_delta', type=float, default=0.5, 
                    help='FGNNEH: Threshold for global model performance diff')
parser.add_argument('--fgnneh_alpha', type=float, default=0.1, 
                    help='FGNNEH: Learning rate for edge weight increase')
parser.add_argument('--fgnneh_beta', type=float, default=0.05, 
                    help='FGNNEH: Learning rate for edge weight decrease')


# ==========================================
#        Added for UFCL Parameters
# ==========================================
parser.add_argument('--lambda_1', type=float, default=5.0, 
                    help='UFCL: 闂備浇妫勯崐褰掑箖閸岀偑鈧懘骞橀鑲╊唹闂佺粯姊归悘姘婵傚憡鐓涘璺猴工閺嗭綁鏌?(Distillation Loss Weight)')
parser.add_argument('--lambda_2', type=float, default=0.1, 
                    help='UFCL: 濠电偛鐡ㄧ划宥囨暜婵犲啰鈹嶅┑鐘叉搐缁犳帗銇勯弽銊ュ毈闁搞倗濮甸幈銊╂晲閸℃鍙嗗銈嗗笧閸犳劗绮欐繝鍥ㄥ亹闁告劘灏欓弻鍫熺節閵忊€冲姸濡炲顭堥…鍥醇閵夛妇鍘?(GraphCL Loss Weight)')
parser.add_argument('--temperature', type=float, default=0.5, 
                    help='UFCL: InfoNCE 闂佽绨肩徊濠氾綖婢舵劖鍎婃い鏃堟暜閸嬫挸鈽夊畷鍥╃獥缂備讲鈧啿顏柟顖氬暣瀹曠喖顢栭崹顔肩亖闂佺澹堥幓顏呬繆閸ヮ剚鐒鹃柟缁㈠枛閺?(Temperature tau)')

# ==========================================
#        Added for Fed4TP Parameters
# ==========================================

parser.add_argument('--time_window_num', type=int, default=5, 
                    help='Fed4TP: TWT闂備礁鎼崯顐︽偉閻撳宫娑氭崉閵娧呭箵濠德板€愰崑鎾舵喐閻楀牏娲寸€殿噮鍋呯€靛ジ寮堕幋鐑嗕画 (TW)')


parser.add_argument('--fed4tp_gd_use', action='store_true', 
                    help='Fed4TP: 闂備礁鎼€氱兘宕规导鏉戠畾濞撴埃鍋撶€规洏鍎甸、娑橆煥閸曨剚顓归梻浣侯焾缁诲牓宕濆畝鍕剳妞ゆ巻鍋撻摶鐐烘煃瑜滈崜娆撴箒闁诲函绲洪弲婵嬫偡閵忋倖鐓涢柛顐亜婢ь喗绻濋埀顒勫箻椤旂厧鍋嶉柣鐘叉搐瀵爼鎮?GLD)')
parser.add_argument('--rho', type=float, default=0.1, 
                    help='Fed4TP: GLD闁诲孩顔栭崰鏍磹閹间焦鍋夐悹杞扮秿濞戙垹鐒垫い鎺嗗亾闂囧鎮楅敐搴℃灍妞ゎ偄鐭傞弻?(unreliable_gradients_threshold)')
parser.add_argument('--last_gradients_length', type=int, default=10, 
                    help='Fed4TP: history length used by GLD for gradient reliability estimation')


parser.add_argument('--fed4tp_mg_use', action='store_true', 
                    help='Fed4TP: 闂備礁鎼€氱兘宕规导鏉戠畾濞撴埃鍋撶€规洏鍎甸、娑橆煥閸曨剚顓瑰┑鐘灱閸╂牕顫濋妸銉囧酣顢欓悾灞告灃濡炪倕绻愰幊澶愬磻閹捐妲婚柣?MPL)')

# DyHSL backbone parameters used by the Fed4TP protocol branch.
parser.add_argument('--dyhsl_dropout', type=float, default=0.1,
                    help='DyHSL backbone dropout.')
parser.add_argument('--dyhsl_num_backbone_layers', type=int, default=2,
                    help='DyHSL: number of backbone GNN layers per path.')
parser.add_argument('--dyhsl_num_head_layers', type=int, default=2,
                    help='DyHSL: number of ST-hypergraph layers per temporal scale.')
parser.add_argument('--dyhsl_num_hyper_edge', type=int, default=32,
                    help='DyHSL: number of dynamic hyperedges.')
parser.add_argument('--dyhsl_winsize', type=int, default=3,
                    help='DyHSL: temporal interaction window size.')
parser.add_argument('--dyhsl_scales', type=str, default='1,3,6,12',
                    help='DyHSL: comma-separated temporal pooling scales, e.g. 1,3,6,12.')

# ==========================================
#        Added for FedMetro Parameters
# ==========================================
parser.add_argument('--fedmetro_d_D', type=int, default=32, 
                    help='FedMetro: Dimension for Cross-Attention hidden state (D_i^t)')

parser.add_argument('--fedmetro_mask_beta', type=float, default=0.66, 
                    help='FedMetro: Temperature beta for Hard Concrete Distribution')

parser.add_argument('--fedmetro_mask_gamma', type=float, default=-0.1, 
                    help='FedMetro: Lower bound gamma for Hard Concrete Distribution')

parser.add_argument('--fedmetro_mask_zeta', type=float, default=1.1, 
                    help='FedMetro: Upper bound zeta for Hard Concrete Distribution')

parser.add_argument('--lambda_reg', type=float, default=0.001, 
                    help='FedMetro: Regularization weight for L0-norm communication compression [cite: 290, 305]')

parser.add_argument('--sparsification_threshold', type=float, default=0.8, 
                    help='FedMetro: Threshold to control sparsity during inference [cite: 308, 414]')

parser.add_argument('--fedmetro_grad_clip', type=float, default=5.0,
                    help='FedMetro: max gradient norm used to avoid non-finite federated updates. Set <=0 to disable.')

parser.add_argument('--norm_scope', type=str, default='global',
                    choices=['global', 'node', 'column'],
                    help='Scaler fitting scope for graph time-series data: global scalar, per-node, or legacy column-wise.')

# ==========================================
#        FedTSE Parameters
# ==========================================
parser.add_argument('--fedtse_upload_periods', type=str, default='',
                    help='FedTSE: comma-separated client upload periods for the asynchronous simulation, '
                         'for example "1,1,1,2". Empty means every client uploads every round.')
                    

# ==========================================
#        Added for STFAM Parameters
# ==========================================
parser.add_argument('--tsvd_dim', type=int, default=10, 
                    help='STFAM: ')
parser.add_argument('--num_clusters', type=int, default=3, 
                    help='STFAM: number of semantic clusters used by K-Means')
parser.add_argument('--stfam_embed_dim', type=int, default=64, 
                    help='STFAM: embedding dimension used by the global representation module')
parser.add_argument('--stfam_global_pretrain_epochs', type=int, default=200,
                    help='STFAM: epochs for centralized/global autoencoder pretraining')
parser.add_argument('--stfam_heartbeat_batches', type=int, default=50,
                    help='STFAM: emit a client training progress message every N batches (0 disables).')
parser.add_argument('--stfam_ablation',
                    type=str,
                    default='full',
                    choices=['full', 'task_only', 'low_recon', 'custom'],
                    help='STFAM ablation mode for reconstruction losses.')
parser.add_argument('--stfam_recon_alpha', type=float, default=0.1,
                    help='STFAM: weight of transfer-matrix reconstruction loss.')
parser.add_argument('--stfam_recon_beta', type=float, default=0.1,
                    help='STFAM: weight of global/spatial feature reconstruction loss.')

# ==========================================
#        Added for FedSTG Parameters
# ==========================================
parser.add_argument('--fedstg_K', type=int, default=10, 
                    help='FedSTG: Number of temporal pattern categories in TP-Bank (K)')

parser.add_argument('--fedstg_alpha', type=float, default=0.01, 
                    help='FedSTG: Regularization weight for balancing loss (alpha)')

parser.add_argument('--fedstg_beta', type=float, default=0.5, 
                    help='FedSTG: Threshold parameter for evolutionary graph (beta)')

# 高斯核宽度 (Sigma)。若设为 0.0，则代码中自动取 distance 的标准差
parser.add_argument('--fedstg_sigma', type=float, default=0.0, 
                    help='FedSTG: Gaussian kernel width (sigma). 0 means auto-calculate.')

# 距离阈值 (Kappa)。注意：这里是距离的上限阈值！若设为 0.0，则不设截断阈值
parser.add_argument('--fedstg_kappa', type=float, default=0.0, 
                    help='FedSTG: Distance threshold for static graph (kappa). 0 means no threshold.')

# ==========================================
#        Added for SFL Parameters
# ==========================================
parser.add_argument('--sfl_lambda', type=float, default=0.1, 
                    help='SFL: 闂備浇顕栭崹顏堝疾濠靛牏绀婇柟鐑樻尪娴滄粓鏌涢敂璇插籍闁稿繐娲弻?lambda')
parser.add_argument('--sfl_m_steps', type=int, default=1, 
                    help='SFL: 闂備礁鎼悧鍡欑矓鐎涙ɑ鍙忛柣鏃囨閸?GCN 闂備浇澹堟ご鎼佹嚌妤ｅ啫绠栭幖娣妽閸庡秹鏌涢弴銊ュ⒒闁告柨瀚伴弻?m')
parser.add_argument('--sfl_gamma', type=float, default=0.01, 
                    help='SFL: Structure learning 闂備焦鐪归崝宀€鈧凹鍓欓湁婵☆垯璀﹀浼存煥濠靛棙鍣归柣蹇曞枛閺屾盯寮介妸褍鈷堥梻渚囧枟閹稿啿顕?(闂備礁鎲￠悷顖炲垂閸洖鐒?')

# ==========================================
#        Added for FUELS Parameters
# ==========================================
parser.add_argument('--sfl_topk', type=int, default=0,
                    help='SFL: sparsify learned client relation graph with per-row top-k neighbors; 0 means auto.')

parser.add_argument('--fuels_tau', type=float, default=0.02, 
                    help='FUELS: Temperature factor for contrastive loss')
parser.add_argument('--fuels_rho', type=float, default=5.0, 
                    help='FUELS: Additive weight of Inter-client contrastive loss')
parser.add_argument('--fuels_beta_percentile', type=float, default=50.0, 
                    help='FUELS: Percentile for JSD threshold (e.g., 50 means median)')
parser.add_argument('--fuels_intra_weight', type=float, default=1.0,
                    help='FUELS: Weight of local intra-client contrastive loss')
parser.add_argument('--fuels_val_metric', type=str, default='normalized_mae',
                    choices=['normalized_mae', 'real_mae'],
                    help='FUELS: Validation metric for selecting the best checkpoint')
parser.add_argument('--fuels_dr', type=int, default=128, 
                    help='FUELS: Dimension of latent representation dr')
parser.add_argument('--fuels_c', type=int, default=6, 
                    help='FUELS: Size of closeness window')
parser.add_argument('--fuels_q', type=int, default=6, 
                    help='FUELS: Size of periodic window')
parser.add_argument('--fuels_aug_noise_std', type=float, default=0.01)
parser.add_argument('--fuels_aug_mask_ratio', type=float, default=0.10)
parser.add_argument('--fuels_aug_shift_prob', type=float, default=0.50)
parser.add_argument('--fuels_aug_shift_pad_mode', type=str, default='edge')

# ==========================================
#        Added for Privacy (DP Noise)
# ==========================================
parser.add_argument('--protection', choices=('plain', 'dp', 'he'), default='plain',
                    help='Privacy condition: plain, DP, or HE protection.')
parser.add_argument('--he_backend', choices=('auto', 'he_sa', 'he_ttp'), default='auto',
                    help='HE backend selected after communication audit.')
parser.add_argument('--he_key_bits', type=int, default=2048,
                    help='Paillier modulus size used by HE experiments.')
parser.add_argument('--he_scheme', choices=('ckks', 'paillier'), default='ckks',
                    help='HE ciphertext scheme. FedGODE defaults to practical packed CKKS.')
parser.add_argument('--he_ckks_poly_modulus_degree', type=int, default=8192,
                    help='CKKS polynomial modulus degree; slots equal half this value.')
parser.add_argument('--he_ckks_scale_bits', type=int, default=40,
                    help='CKKS global scale in bits.')
parser.add_argument('--he_encrypt_return', type=int, choices=(0, 1), default=0,
                    help='Encrypt Arbiter-to-client payloads only after client-side key support is configured.')
parser.add_argument('--privacy_audit', action='store_true', default=False,
                    help='Log tensor-bearing federated messages without modifying training.')
parser.add_argument('--dp_upload_log_interval', type=int, default=0,
                    help='DP upload logging interval per payload type. 0 logs only the first upload of each type per rank; positive N additionally logs every Nth upload.')
parser.add_argument('--efficiency_audit_plain', action='store_true', default=False,
                    help='Run the explicit privacy trainer in Plain mode for a like-for-like HE efficiency audit; does not replace the original baseline result.')
parser.add_argument('--ufcl_dp_protocol_audit', action='store_true', default=False,
                    help='UFCL only: force the explicit DP delta-aggregation loop with sigma=0 for a protocol-equivalence audit. This is distinct from the native Plain equivalence run.')
parser.add_argument('--ufcl_native_delta_dp_audit', action='store_true', default=False,
                    help='UFCL only: run the native delta-DP aggregation adapter with sigma=0 and an inactive clipping bound.')
parser.add_argument('--dp_sigma', type=float, default=0.5,
                    help='Gaussian update-level DP noise multiplier.')
parser.add_argument('--dp_clip_norm', type=float, default=0.0,
                    help='Global L2 clip norm; 0 automatically calibrates the first-round delta 90th percentile.')
parser.add_argument('--dp_post_clip_ratio', type=float, default=0.0,
                    help='Optional post-noise global L2 radius R/C. 0 disables it; 1 clips each complete noised upload to R=C.')
parser.add_argument('--experimental_noise_clip_ratio', type=float, default=0.0,
                    help='EXPERIMENTAL ONLY: cap sampled noise L2 at R/C before addition; not standard Gaussian DP.')
parser.add_argument('--experimental_skip_update_clip', action='store_true', default=False,
                    help='EXPERIMENTAL ONLY: do not clip the true update. This disables the standard DP guarantee.')
parser.add_argument('--dp_noise_seed', type=int, default=-1,
                    help='Optional deterministic seed for DP upload noise; -1 uses the process RNG.')
parser.add_argument('--dp_noise', type=float, default=0.0,
                    help='Legacy baseline-specific DP noise scale (deprecated for privacy experiments).')
parser.add_argument('--privacy_result_dir', type=str, default='',
                    help='Optional directory for privacy-experiment CSV results. Empty keeps the default log directory.')
parser.add_argument('--privacy_trace_dir', type=str, default=os.environ.get('PRIVACY_TRACE_DIR', ''),
                    help='Optional directory for one protected communication trace used by reconstruction experiments.')
parser.add_argument('--privacy_trace_client', type=int, default=int(os.environ.get('PRIVACY_TRACE_CLIENT', '0')),
                    help='Client rank whose protected upload metadata is exported for privacy reconstruction.')
parser.add_argument('--privacy_trace_max_records', type=int, default=int(os.environ.get('PRIVACY_TRACE_MAX_RECORDS', '1')),
                    help='Maximum matching protected messages exported by each process.')
parser.add_argument('--privacy_trace_tag', type=str, default=os.environ.get('PRIVACY_TRACE_TAG', ''),
                    help='Optional substring filter for protected message tags exported to privacy_trace_dir.')
parser.add_argument('--privacy_trace_save_insider', action='store_true', default=os.environ.get('PRIVACY_TRACE_SAVE_INSIDER', '0') == '1',
                    help='Save decrypted-equivalent client payload locally for a separately labelled HE-TTP insider upper-bound study.')
parser.add_argument('--privacy_trace_save_collusion', action='store_true', default=os.environ.get('PRIVACY_TRACE_SAVE_COLLUSION', '0') == '1',
                    help='Save an HE-SA target update replay trace for the separately labelled server-plus-K-1-client collusion study.')
parser.add_argument('--privacy_trace_save_kminus2', action='store_true', default=os.environ.get('PRIVACY_TRACE_SAVE_KMINUS2', '0') == '1',
                    help='Save the two hidden-client local terms required to construct a separately labelled HE-SA K-2 collusion residual trace.')
parser.add_argument('--privacy_trace_save_kminus3', action='store_true', default=os.environ.get('PRIVACY_TRACE_SAVE_KMINUS3', '0') == '1',
                    help='Save the three hidden-client local terms required to construct a separately labelled HE-SA one-client-collusion residual trace.')
parser.add_argument('--privacy_trace_save_server_aggregate', action='store_true', default=os.environ.get('PRIVACY_TRACE_SAVE_SERVER_AGGREGATE', '0') == '1',
                    help='Save four protocol-instrumentation terms to construct a separately labelled HE-SA server-only aggregate trace.')
parser.add_argument('--privacy_trace_only', action='store_true', default=os.environ.get('PRIVACY_TRACE_ONLY', '0') == '1',
                    help='Capture HE reconstruction traces without appending efficiency rows to benchmark CSV files.')
parser.add_argument('--privacy_trace_sample_index', type=int, default=int(os.environ.get('PRIVACY_TRACE_SAMPLE_INDEX', '0')),
                    help='One deterministic local sample retained by a trace-only run; used with batch_size=1.')
parser.add_argument('--quant_prediction_bits', type=int, default=int(os.environ.get('PRIVACY_QUANT_PREDICTION_BITS', '4')),
                    help='Public fixed-point bit width for the revised prediction side channel.')
parser.add_argument('--quant_prediction_clip', type=float, default=float(os.environ.get('PRIVACY_QUANT_PREDICTION_CLIP', '3.0')),
                    help='Public symmetric normalized clipping range for the revised prediction side channel.')
parser.add_argument('--quant_prediction_summary', choices=('full', 'mean_std'),
                    default=os.environ.get('PRIVACY_QUANT_PREDICTION_SUMMARY', 'full'),
                    help='Public prediction side-channel surface: full tensor, or only per-sample mean and std.')
parser.add_argument('--cnfgnn_trace_max_steps', type=int,
                    default=int(os.environ.get('CNFGNN_TRACE_MAX_STEPS', '0')),
                    help='CNFGNN trace-only diagnostic: cap every synchronized phase at this many batches. 0 keeps full training.')
parser.add_argument('--dp_reconstruction_trace_dir', type=str, default='',
                    help='Optional local directory for one DP reconstruction replay trace (noisy payload plus model-state-assisted audit snapshot).')
parser.add_argument('--dp_reconstruction_trace_client', type=int, default=0,
                    help='Client rank whose DP reconstruction trace is retained.')
parser.add_argument('--dp_reconstruction_trace_tag', type=str, default='',
                    help='Optional substring filter for DP reconstruction trace message tags.')

parser.add_argument('--fedtps_k', type=int, default=2, 
                    help='FedTPS: Number of selected patterns k during similarity aggregation')

# ==========================================
#        Added for pFedCTP Parameters
# ==========================================
parser.add_argument('--finetune_all', action='store_true', default=False,
                    help='pFedCTP: allow all clients to run Stage-2 fine-tuning (default: target client only)')
parser.add_argument('--pfedctp_stage1_batch_ratio', type=float, default=0.2,
                    help='pFedCTP: ratio of local batches used per communication round in Stage-1')




# T-ISTGNN(i-c) 专用参数
parser.add_argument('--target_init_epochs',
                    type=int,
                    default=10,
                    help='T-ISTGNN(i-c): epochs for target-associated initialization')

parser.add_argument('--transfer_epochs',
                    type=int,
                    default=30,
                    help='T-ISTGNN(i-c): epochs for target adaptation after source aggregation')

parser.add_argument('--lambda_mmd',
                    type=float,
                    default=0.1,
                    help='T-ISTGNN(i-c): weight of MMD loss in target adaptation')

parser.add_argument('--freeze_predictor',
                    action='store_true',
                    default=True,
                    help='T-ISTGNN(i-c): freeze predictor during target adaptation')

# ==========================================
#        Added for FedmSSA Parameters
# ==========================================
parser.add_argument('--fedmssa_page_length',
                    type=int,
                    default=48,
                    help='FedmSSA: page matrix window length L.')
parser.add_argument('--fedmssa_rank',
                    type=int,
                    default=0,
                    help='FedmSSA: explicit low-rank truncation; <=0 means auto-select.')
parser.add_argument('--fedmssa_sv_scale',
                    type=float,
                    default=2.0,
                    help='FedmSSA: singular value threshold scale for auto rank selection.')
parser.add_argument('--fedmssa_gru_layers',
                    type=int,
                    default=2,
                    help='FedmSSA: number of GRU layers in the prediction head.')
parser.add_argument('--fedmssa_dropout',
                    type=float,
                    default=0.0,
                    help='FedmSSA: dropout used in the GRU predictor.')
parser.add_argument('--fedmssa_patience',
                    type=int,
                    default=50,
                    help='FedmSSA: default patience when running through fate_main.py.')
parser.add_argument('--fedmssa_min_delta',
                    type=float,
                    default=0.0,
                    help='FedmSSA: default early-stop min_delta when running through fate_main.py.')
parser.add_argument('--fedmssa_missing_ratio',
                    type=float,
                    default=0.0,
                    help='FedmSSA: random missing ratio used to build an explicit observation mask for phase-1 imputation.')
parser.add_argument('--fedmssa_impute_rounds',
                    type=int,
                    default=20,
                    help='FedmSSA: number of global consensus rounds in phase-1 federated imputation.')
parser.add_argument('--fedmssa_impute_local_steps',
                    type=int,
                    default=10,
                    help='FedmSSA: number of local optimization steps for each U_i update per imputation round.')
parser.add_argument('--fedmssa_impute_lr',
                    type=float,
                    default=5e-2,
                    help='FedmSSA: local step size for phase-1 imputation basis updates.')
parser.add_argument('--fedmssa_consensus_weight',
                    type=float,
                    default=1.0,
                    help='FedmSSA: weight for the consensus penalty ||U_i - Z||^2.')
parser.add_argument('--fedmssa_ortho_weight',
                    type=float,
                    default=1.0,
                    help='FedmSSA: weight for the orthogonality penalty ||U_i^T U_i - I||^2.')
parser.add_argument('--fedmssa_diag_weight',
                    type=float,
                    default=0.2,
                    help='FedmSSA: weight for the decorrelation/off-diagonal penalty.')
parser.add_argument('--fedmssa_server_momentum',
                    type=float,
                    default=0.0,
                    help='FedmSSA: optional server-side momentum when updating the global consensus basis Z.')

# ==========================================
#        Added for TDLR / SED-LR Parameters
# ==========================================
parser.add_argument('--tdlr_stream_rounds',
                    type=int,
                    default=200,
                    help='TDLR/SED-LR: number of streaming federated rounds. <=0 means use args.epochs.')
parser.add_argument('--tdlr_lstm_layers',
                    type=int,
                    default=2,
                    help='TDLR/SED-LR: number of LSTM layers in the lightweight streaming predictor.')
parser.add_argument('--tdlr_dropout',
                    type=float,
                    default=0.2,
                    help='TDLR/SED-LR: dropout in the 2-layer LSTM predictor.')
parser.add_argument('--tdlr_schedule',
                    type=str,
                    default='0:1.0,50:1.5,100:2.0,150:1.5',
                    help='TDLR: manual round-to-lr-multiplier schedule formatted as round:multiplier,...')
parser.add_argument('--tdlr_patience',
                    type=int,
                    default=50,
                    help='TDLR/SED-LR: early stopping patience on averaged client validation MAE.')
parser.add_argument('--tdlr_min_delta',
                    type=float,
                    default=0.0,
                    help='TDLR/SED-LR: early stopping min delta.')
parser.add_argument('--tdlr_recent_buffer_size',
                    type=int,
                    default=8,
                    help='TDLR/SED-LR: number of recent streaming batches kept for trigger statistics and local lr optimization.')
parser.add_argument('--sedlr_occ_threshold',
                    type=float,
                    default=0.8,
                    help='SED-LR: occupancy threshold for triggering asynchronous local lr adaptation.')
parser.add_argument('--sedlr_anomaly_sigma',
                    type=float,
                    default=2.0,
                    help='SED-LR: anomaly trigger threshold in standard deviations over recent batch means.')
parser.add_argument('--sedlr_warmup_rounds',
                    type=int,
                    default=10,
                    help='SED-LR: warm-up rounds before asynchronous local triggers are allowed.')
parser.add_argument('--sedlr_cooldown_rounds',
                    type=int,
                    default=5,
                    help='SED-LR: minimum rounds between two local trigger events on the same client.')
parser.add_argument('--sedlr_aggressive_mult',
                    type=float,
                    default=2.0,
                    help='SED-LR: multiplier applied to the base learning rate when a local trigger fires.')
parser.add_argument('--sedlr_calm_mult',
                    type=float,
                    default=1.0,
                    help='SED-LR: multiplier applied to the base learning rate when no local trigger fires.')
parser.add_argument('--tdlr_bayes_enable',
                    action='store_true',
                    default=False,
                    help='TDLR-B: periodically optimize the local learning rate with a lightweight Bayesian search.')
parser.add_argument('--tdlr_bayes_every',
                    type=int,
                    default=50,
                    help='TDLR-B: run local lr optimization every N rounds.')
parser.add_argument('--tdlr_bayes_iter',
                    type=int,
                    default=5,
                    help='TDLR-B/SED-LR: number of optimization steps for local lr search.')
parser.add_argument('--tdlr_bayes_lb',
                    type=float,
                    default=1e-4,
                    help='TDLR-B/SED-LR: lower bound of the learning-rate search space.')
parser.add_argument('--tdlr_bayes_ub',
                    type=float,
                    default=5e-3,
                    help='TDLR-B/SED-LR: upper bound of the learning-rate search space.')
parser.add_argument('--tdlr_bayes_kfold',
                    type=int,
                    default=3,
                    help='TDLR-B/SED-LR: K-fold count used by local lr optimization.')
parser.add_argument('--sedlr_bayes_enable',
                    action='store_true',
                    default=False,
                    help='SED-LR: when a local trigger fires, optimize the lr with the same lightweight Bayesian search instead of using a fixed multiplier.')
parser.add_argument('--sedlr_bayes_warmup_rounds',
                    type=int,
                    default=10,
                    help='SED-LR Bayes: warm-up rounds before trigger-driven Bayesian lr re-optimization is allowed.')
parser.add_argument('--sedlr_bayes_reopt_every',
                    type=int,
                    default=20,
                    help='SED-LR Bayes: minimum rounds between two trigger-driven Bayesian lr re-optimizations on the same client.')


args, _ = parser.parse_known_args()
