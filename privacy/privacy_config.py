from __future__ import annotations

import argparse
from dataclasses import dataclass


GRAPH_DATASETS = ("PeMS03", "PeMS04", "PeMSD7", "PeMS08")
GRID_DATASETS = ("TaxiBJ", "TaxiNYC", "BikeNYC")
FEATURE_TYPES = ("flow", "speed", "occ")


@dataclass
class PrivacyDefaults:
    attack_iters: int = 500
    attack_lr: float = 5e-2
    sample_index: int = 0
    client_rank: int = 0
    result_dir: str = "privacy_results"
    mape_eps: float = 10.0


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = PrivacyDefaults()
    parser = argparse.ArgumentParser(
        description="Data reconstruction attack entrypoint for traffic baselines."
    )

    parser.add_argument("--model", default="FGNNEH", type=str)
    parser.add_argument("--attack", default="auto", type=str,
                        help="auto, gradient, activation, or model_update")
    parser.add_argument("--dataset_name", default="PeMS04", type=str)
    parser.add_argument("--feature_type", choices=FEATURE_TYPES, default="flow", type=str)
    parser.add_argument("--normalizer", default="std", type=str)

    parser.add_argument("--train_ratio", default=0.7, type=float)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--num_clients", default=4, type=int)
    parser.add_argument(
        "--partition_strategy",
        choices=("auto", "metis", "grid", "louvain", "geographic", "sensorid"),
        default="auto",
        help="Must match the partition artifact used by the corresponding FATE run.",
    )
    parser.add_argument("--client_rank", default=defaults.client_rank, type=int,
                        help="Client id to attack. Follows the local rank used by fate_main.py.")
    parser.add_argument("--sample_index", default=defaults.sample_index, type=int,
                        help="Start index in the selected split for the reconstruction batch.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Number of consecutive samples to stack into one reconstruction batch.")
    parser.add_argument("--attack_node_index", default=-1, type=int,
                        help="Diagnostic only: attack one local node instead of all local nodes.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train", type=str)

    parser.add_argument("--t_in", default=12, type=int)
    parser.add_argument("--t_out", default=3, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--input_dim", default=1, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    # UFCL's prediction entrypoint is ``UFCL_GWN.LightGraphWaveNet``.  These
    # options are kept here (rather than relying on implicit defaults) so a
    # reconstruction model can be made structurally identical to the model
    # that produced an HE-SA trace.
    parser.add_argument("--ufcl_gwn_channels", default=29, type=int)
    parser.add_argument("--ufcl_gwn_skip_channels", default=58, type=int)
    parser.add_argument("--ufcl_gwn_end_channels", default=58, type=int)
    parser.add_argument("--ufcl_gwn_blocks", default=2, type=int)
    parser.add_argument("--ufcl_gwn_layers", default=2, type=int)
    parser.add_argument("--ufcl_gwn_dropout", default=0.1, type=float)
    # Reconstruction must instantiate the same architecture as the matching
    # prediction run.  FedGTP/FedMetro adapters consume these two settings.
    parser.add_argument("--node_emb_dim", default=4, type=int)
    parser.add_argument("--poly_k", default=2, type=int)
    parser.add_argument("--loss_func", choices=("mse", "mae", "l1"), default="mse", type=str)
    parser.add_argument("--wd", default=1e-8, type=float)
    parser.add_argument("--fca_dim", default=100, type=int,
                        help="FCFedGCN FCA feature dimension used by the privacy adapter.")
    parser.add_argument("--ext_dim", default=21, type=int,
                        help="External feature dimension for TwoMGTCN zero-filled covariates.")
    parser.add_argument("--fgnneh_P", default=0.05, type=float)
    parser.add_argument("--fgnneh_gamma", default=0.1, type=float)
    parser.add_argument("--fgnneh_n_components", default=8, type=int)
    parser.add_argument("--fgnneh_activation_scope", choices=("hypernode", "backbone", "quantized_prediction"), default="hypernode",
                        help="FGNNEH activation leak: realistic server-visible hypernode or stronger backbone diagnostic.")
    parser.add_argument("--fgnneh_hypernode_leak_ratio", default=1.0, type=float,
                        help="Fixed fraction (0,1] of FGNNEH hypernode coordinates disclosed to the observer."
                             " The first coordinates are retained deterministically for reproducible partial-leakage ablations.")
    parser.add_argument("--fgnneh_quant_prediction_bits", default=4, type=int,
                        help="Public fixed-point bit width for the FGNNEH quantized-prediction side-channel study.")
    parser.add_argument("--fgnneh_quant_prediction_clip", default=3.0, type=float,
                        help="Public symmetric normalized clipping range for FGNNEH quantized predictions.")
    parser.add_argument("--quantized_prediction_sidechannel", action="store_true",
                        help="Revised-protocol ablation: client returns a quantized prediction in addition to the protected native upload.")
    parser.add_argument("--quant_prediction_bits", default=4, type=int,
                        help="Fixed public bit width for the revised-protocol quantized prediction.")
    parser.add_argument("--quant_prediction_clip", default=3.0, type=float,
                        help="Fixed public normalized clipping range for the revised-protocol prediction.")
    parser.add_argument("--quant_prediction_summary", choices=("full", "mean_std"), default="full",
                        help="Public prediction side-channel surface: full tensor, or only per-sample mean and std.")
    parser.add_argument("--sfl_base_model", default="SFL_RNN", type=str,
                        help="Base local model used inside the SFL privacy adapter.")
    parser.add_argument("--sfl_lambda", default=0.1, type=float,
                        help="SFL personalization regularization weight; logged for attack metadata.")
    parser.add_argument("--sfl_m_steps", default=1, type=int,
                        help="SFL graph aggregation steps; logged for attack metadata.")
    parser.add_argument("--sfl_attack_surface", choices=("v", "v_graph"), default="v_graph",
                        help="SFL leak surface: uploaded local weights only, or weights plus server graph-learning vector.")
    parser.add_argument("--sfl_update_optimizer", choices=("sgd", "adam"), default="sgd",
                        help="Optimizer simulated inside SFL model_update leak.")
    parser.add_argument("--sfl_graph_weight", default=1e-3, type=float,
                        help="Weight for matching the normalized client-parameter vector used by SFL server graph learning.")
    parser.add_argument("--sfl_reg_anchor", choices=("none", "initial", "server"), default="none",
                        help="Proxy for SFL's known w/u regularization anchors during multi-step local updates.")
    parser.add_argument("--sfl_server_unroll_epochs", default=2, type=int,
                        help="When sfl_reg_anchor=server, unroll this many SFL epochs: epoch0 uploads v, later epochs use server-returned w/u.")
    parser.add_argument("--sfl_num_surrogate_clients", default=0, type=int,
                        help="Number of extra surrogate clients used to approximate SFL server aggregation. 0 falls back to num_clients-1 or 3.")
    parser.add_argument("--sfl_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by SFL.")
    parser.add_argument("--sfl_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by SFL.")
    parser.add_argument("--sfl_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by SFL.")
    parser.add_argument("--stfam_global_pretrain_epochs", default=200, type=int,
                        help="STFAM server-side global CNN pretraining epochs used to derive V_D/P_embed.")
    parser.add_argument("--stfam_global_lr", default=1e-3, type=float,
                        help="STFAM server-side global CNN pretraining learning rate.")
    parser.add_argument("--stfam_embed_dim", default=64, type=int,
                        help="STFAM embedding width for the reconstruction surrogate.")
    parser.add_argument("--tsvd_dim", default=10, type=int,
                        help="STFAM TSVD communication-vector dimension.")
    parser.add_argument("--num_clusters", default=3, type=int,
                        help="STFAM server-side clustering count.")
    parser.add_argument("--twomgtcn_lfac_alpha", default=0.01, type=float,
                        help="TwoMGTCN feature-factorization loss weight used by its local trainer.")
    parser.add_argument("--twomgtcn_use_ext", choices=(0, 1), default=1, type=int,
                        help="1 lets TwoMGTCN use grid external covariates; 0 falls back to no external features.")
    parser.add_argument("--twomgtcn_attack_surface",
                        choices=("feature_update", "feature_weight_update", "update", "update_fpass"),
                        default="feature_update",
                        help="TwoMGTCN leak surface. feature_update focuses on the input fusion layer; update_fpass also matches the FPASS proxy.")
    parser.add_argument("--twomgtcn_update_optimizer", choices=("sgd", "adam"), default="sgd",
                        help="Optimizer simulated inside TwoMGTCN model_update leak.")
    parser.add_argument("--twomgtcn_fpass_weight", default=0.1, type=float,
                        help="Weight for matching the flattened local weights used by TwoMGTCN FPASS server similarity.")
    parser.add_argument("--twomgtcn_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by TwoMGTCN.")
    parser.add_argument("--twomgtcn_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by TwoMGTCN.")
    parser.add_argument("--twomgtcn_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by TwoMGTCN.")
    parser.add_argument("--fedstn_activation_surface", choices=("hs", "hs_context_agg", "hs_local_upper", "fedgat_batch_context"),
                        default="hs_context_agg",
                        help="FedSTN activation leak: uploaded h_s_i, h_s_i plus server context/agg_hs, local-output upper bound, or batch-pooled FedGAT context only.")
    parser.add_argument("--fedstn_hs_weight", default=1.0, type=float,
                        help="Weight for matching FedSTN's uploaded h_s_i activation.")
    parser.add_argument("--fedstn_context_weight", default=0.1, type=float,
                        help="Weight for matching FedSTN's server attention context proxy.")
    parser.add_argument("--fedstn_agg_weight", default=0.1, type=float,
                        help="Weight for matching FedSTN's server-returned agg_hs proxy.")
    parser.add_argument("--fedstn_rlcn_weight", default=0.1, type=float,
                        help="Diagnostic-only weight for FedSTN local RLCN output upper-bound leak.")
    parser.add_argument("--fedstn_scn_weight", default=0.1, type=float,
                        help="Diagnostic-only weight for FedSTN local SCN output upper-bound leak.")
    parser.add_argument("--fedmetro_activation_scope", choices=("server", "full"), default="server",
                        help="FedMetro activation leak scope: server-visible AGG only, or full AGG+F_E upper bound.")
    parser.add_argument("--fedmetro_attack_surface", choices=("agg", "agg_gagg", "agg_gagg_update"),
                        default="agg_gagg_update",
                        help="FedMetro server-visible leak surface used by model_update attacks.")
    parser.add_argument("--fedmetro_agg_weight", default=1.0, type=float,
                        help="Weight for matching FedMetro's uploaded AGG_i activation.")
    parser.add_argument("--fedmetro_gagg_weight", default=0.05, type=float,
                        help="Weight for matching FedMetro's uploaded split gradient g_agg.")
    parser.add_argument("--fedmetro_update_weight", default=0.005, type=float,
                        help="Weight for matching FedMetro's epoch uploaded weight-update direction.")
    parser.add_argument("--fedmetro_lambda_reg", default=0.001, type=float,
                        help="FedMetro mask regularization weight used when lambda_reg is unavailable.")
    parser.add_argument("--fedmssa_page_length", default=12, type=int,
                        help="FedmSSA page matrix window length L.")
    parser.add_argument("--fedmssa_rank", default=0, type=int,
                        help="FedmSSA explicit low-rank truncation; <=0 means auto-select.")
    parser.add_argument("--fedmssa_sv_scale", default=2.0, type=float,
                        help="FedmSSA singular value threshold scale for auto rank selection.")
    parser.add_argument("--fedmssa_gru_layers", default=2, type=int,
                        help="FedmSSA number of GRU layers in the prediction head.")
    parser.add_argument("--fedmssa_dropout", default=0.0, type=float,
                        help="FedmSSA dropout used in the GRU predictor.")
    parser.add_argument("--fedmssa_patience", default=20, type=int,
                        help="FedmSSA default patience when running through fate_main.py.")
    parser.add_argument("--fedmssa_min_delta", default=1e-4, type=float,
                        help="FedmSSA default early-stop min_delta when running through fate_main.py.")
    parser.add_argument("--fedmssa_missing_ratio", default=0.0, type=float,
                        help="FedmSSA random missing ratio used to build an explicit observation mask for phase-1 imputation.")
    parser.add_argument("--fedmssa_impute_rounds", default=20, type=int,
                        help="FedmSSA number of global consensus rounds in phase-1 federated imputation.")
    parser.add_argument("--fedmssa_impute_local_steps", default=10, type=int,
                        help="FedmSSA number of local optimization steps for each U_i update per imputation round.")
    parser.add_argument("--fedmssa_impute_lr", default=5e-2, type=float,
                        help="FedmSSA local step size for phase-1 imputation basis updates.")
    parser.add_argument("--fedmssa_consensus_weight", default=1.0, type=float,
                        help="FedmSSA weight for the consensus penalty ||U_i - Z||^2.")
    parser.add_argument("--fedmssa_ortho_weight", default=1.0, type=float,
                        help="FedmSSA weight for the orthogonality penalty ||U_i^T U_i - I||^2.")
    parser.add_argument("--fedmssa_diag_weight", default=0.2, type=float,
                        help="FedmSSA weight for the decorrelation/off-diagonal penalty.")
    parser.add_argument("--fedmssa_server_momentum", default=0.0, type=float,
                        help="FedmSSA optional server-side momentum when updating the global consensus basis Z.")
    parser.add_argument("--fedmssa_phase1_device", default="cpu", type=str,
                        help="Device for FedmSSA phase-1 low-rank imputation simulation. cpu is slower but much more memory-stable for privacy attacks.")
    parser.add_argument("--fedmssa_update_optimizer", choices=("sgd", "adam"), default="sgd",
                        help="Optimizer simulated for FedmSSA phase-2 predictor uploads inside model_update attacks.")
    parser.add_argument("--fedmssa_gradient_scope", choices=("head", "predictor", "full"), default="head",
                        help="FedmSSA gradient attack scope: only output head gradients, the whole predictor, or the full model.")
    parser.add_argument("--fedmssa_series_scope", choices=("full", "local"), default="full",
                        help="FedmSSA privacy attack scope for phase-1 denoising: full split for higher fidelity, local crop for much faster attacks.")
    parser.add_argument("--fedmssa_context_pages", default=2, type=int,
                        help="FedmSSA local-scope context pages kept before/after the attacked batch when fedmssa_series_scope=local.")
    parser.add_argument("--fedmssa_activation_surface", choices=("phase1_local", "phase1_local_consensus", "phase1_trace"), default="phase1_trace",
                        help="FedmSSA realistic server-visible phase-1 leak: uploaded local bases only, local plus returned consensus, or the full phase-1 trace summary.")
    parser.add_argument("--fedmssa_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by FedmSSA phase-2 predictor updates.")
    parser.add_argument("--fedmssa_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by FedmSSA phase-2 predictor updates.")
    parser.add_argument("--fedmssa_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by FedmSSA phase-2 predictor updates.")
    parser.add_argument("--fuels_dr", default=64, type=int,
                        help="FUELS representation/prototype dimension dr.")
    parser.add_argument("--fuels_c", default=3, type=int,
                        help="FUELS closeness length c.")
    parser.add_argument("--fuels_q", default=3, type=int,
                        help="FUELS pseudo-periodicity length q.")
    parser.add_argument("--fuels_protocol_batch_size", default=0, type=int,
                        help="FUELS batch size used to construct the uploaded client prototype. "
                             "0 reuses --batch_size; set this to the training batch size when "
                             "running a batch-size-one reconstruction.")
    parser.add_argument("--fuels_activation_scope", choices=("prototype", "prototype_batch", "prototype_batch_aug", "prototype_batch_aug_prnr", "batch_repr_upper"),
                        default="prototype_batch",
                        help="FUELS leak surface: uploaded prototype only, plus current-batch representation, augmented-view representation, or a diagnostic per-sample representation upper bound.")
    parser.add_argument("--fuels_tau", default=0.02, type=float,
                        help="FUELS contrastive temperature used by intra/inter prototype training.")
    parser.add_argument("--fuels_rho", default=5.0, type=float,
                        help="FUELS inter-client contrastive loss weight.")
    parser.add_argument("--fuels_beta_percentile", default=50.0, type=float,
                        help="FUELS server threshold percentile for splitting positive/negative prototypes.")
    parser.add_argument("--fuels_intra_weight", default=1.0, type=float,
                        help="FUELS intra-view contrastive loss weight.")
    parser.add_argument("--fuels_num_surrogate_clients", default=0, type=int,
                        help="Number of surrogate peer prototypes used to approximate other FUELS clients on the server. 0 falls back to num_clients-1.")
    parser.add_argument("--fuels_feedback_temperature", default=0.05, type=float,
                        help="Soft threshold temperature for differentiable FUELS PR/NR approximation.")
    parser.add_argument("--fuels_aug_noise_std", default=0.01, type=float,
                        help="FUELS augmentation noise std used for prototype_batch_aug leakage.")
    parser.add_argument("--fuels_aug_mask_ratio", default=0.10, type=float,
                        help="FUELS temporal masking ratio used for prototype_batch_aug leakage.")
    parser.add_argument("--fuels_aug_shift_prob", default=0.50, type=float,
                        help="FUELS temporal shift probability used for prototype_batch_aug leakage.")
    parser.add_argument("--fuels_aug_shift_pad_mode", default="edge", type=str,
                        help="FUELS temporal shift padding mode.")
    parser.add_argument("--fuels_periodicity_weight", default=0.0, type=float,
                        help="FUELS attack-only periodicity prior weight on batch representations; encourages encoded samples within a batch to be average-able into a stable prototype.")
    parser.add_argument("--fuels_periodicity_mode", choices=("mean_l2", "mean_cos", "mixed"), default="mixed",
                        help="FUELS periodicity prior form on model.encode(dummy_x): L2-to-mean, cosine-to-mean, or both.")
    parser.add_argument("--protection", choices=("plain", "dp", "he"), default="plain",
                        help="Observed protection condition for this reconstruction run.")
    parser.add_argument("--he_backend", choices=("he_sa", "he_ttp"), default="he_sa",
                        help="HE backend that defines the attacker-visible leakage surface.")
    parser.add_argument("--he_attack_scenario", choices=("aggregate_only", "ciphertext_only_prior"),
                        default="aggregate_only", help="HE reconstruction scenario.")
    parser.add_argument("--he_trace_path", default="", type=str,
                        help="External HE trace exported by fate_main.py. HE-SA requires its aggregate trace; HE-TTP uses its ciphertext-visible metadata trace.")
    parser.add_argument("--he_insider_trace", default="", type=str,
                        help="Explicitly privileged HE replay trace containing a decrypted target upload and public round-start model. It is an insider upper bound, never an external HE result.")
    parser.add_argument("--he_ttp_insider_trace", default="", type=str,
                        help="Replay-ready decrypted HE-TTP target payload for the explicitly labelled trusted-Arbiter insider upper bound.")
    parser.add_argument("--he_collusion_trace", default="", type=str,
                        help="Explicit HE-SA replay trace for a server-plus-K-1-client collusion attack. It is not an external-server-only result and not an Arbiter-insider upper bound.")
    parser.add_argument("--he_kminus2_trace", default="", type=str,
                        help="HE-SA residual trace for a server-plus-K-2-client collusion attack. It mixes two unknown client updates and is not an external-server-only result.")
    parser.add_argument("--he_kminus3_trace", default="", type=str,
                        help="HE-SA residual trace for a server-plus-one-client collusion attack. It mixes three unknown client updates and is not an external-server-only result.")
    parser.add_argument("--he_server_aggregate_trace", default="", type=str,
                        help="HE-SA trace containing only the server-visible decrypted aggregate of all clients; no individual update or colluding client is exposed.")
    parser.add_argument("--attack_return_terminal_iterate", action="store_true", default=False,
                        help="Report the terminal early-stop iterate rather than the historical minimum-loss snapshot; never use PCC for selection.")
    parser.add_argument("--attack_initialization_seed", default=-1, type=int,
                        help="Optional explicit seed applied immediately before a single attack run; controls dummy initialization only.")
    parser.add_argument("--dp_sigma", default=0.5, type=float,
                        help="Gaussian DP noise multiplier; the study default is 0.5.")
    parser.add_argument("--dp_clip_norm", default=0.0, type=float,
                        help="Fallback global L2 clipping norm for one complete upload. A positive value is required for DP unless --dp_calibration_log or --dp_clip_norms supplies the observed per-payload C.")
    parser.add_argument("--dp_clip_norms", default="", type=str,
                        help="Optional JSON mapping of observed DP payload type to clip C, e.g. '{\"encoding\": 1.2}'.")
    parser.add_argument("--dp_calibration_log", default="", type=str,
                        help="DP prediction log containing [DPCalibration] records. Its per-payload C values are replayed by the DP reconstruction attack.")
    parser.add_argument("--dp_observed_trace", default="", type=str,
                        help="Optional explicit DP replay trace containing a noisy observed payload and model-state-assisted audit snapshot.")
    parser.add_argument("--dp_attack_auto_clip", action="store_true", default=False,
                        help="Fallback only: when no prediction calibration log is available, set each batch_size=1 attack payload C to its observed raw L2 norm. Results are labelled attack_local_l2, not training-q90 replay.")
    parser.add_argument("--dp_post_clip_ratio", default=0.0, type=float,
                        help="Optional DP-safe post-noise global L2 clipping radius R/C. 0 disables it; 1 sets R=C.")
    parser.add_argument("--experimental_noise_clip_ratio", default=0.0, type=float,
                        help="EXPERIMENTAL ONLY: cap sampled noise L2 at R/C before addition. This creates truncated, non-Gaussian noise and is not standard DP.")
    parser.add_argument("--dp_noise_seed", default=-1, type=int,
                        help="Seed for the one fixed observed DP upload; -1 derives it from --seed.")
    parser.add_argument("--dp_noise", default=0.0, type=float,
                        help="Legacy baseline-specific noise parameter; do not use for the unified DP experiment.")
    parser.add_argument("--fedtse_update_optimizer", choices=("sgd", "adam"), default="sgd",
                        help="Optimizer simulated inside FedTSE model_update leakage.")
    parser.add_argument("--fedtse_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by FedTSE.")
    parser.add_argument("--fedtse_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by FedTSE.")
    parser.add_argument("--fedtse_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by FedTSE.")
    # Keep the TDLR/SED-LR reconstruction model structurally identical to the
    # model that produced a captured HE aggregate trace.
    parser.add_argument("--tdlr_lstm_layers", default=2, type=int,
                        help="Number of TDLR/SED-LR LSTM layers in the reconstructed model.")
    parser.add_argument("--tdlr_dropout", default=0.2, type=float,
                        help="TDLR/SED-LR LSTM dropout in the reconstructed model.")
    parser.add_argument("--fedgru_update_optimizer", choices=("sgd", "adam"), default="adam",
                        help="Optimizer simulated inside FedGRU model_update leakage. FedGRU training uses Adam in fate_main.py.")
    parser.add_argument("--fedgru_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by FedGRU.")
    parser.add_argument("--fedgru_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by FedGRU.")
    parser.add_argument("--fedgru_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by FedGRU.")
    parser.add_argument("--tdlr_update_optimizer", choices=("sgd", "adam"), default="adam",
                        help="Optimizer simulated inside TDLR/SED-LR model_update leakage.")
    parser.add_argument("--tdlr_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by TDLR/SED-LR.")
    parser.add_argument("--tdlr_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by TDLR/SED-LR.")
    parser.add_argument("--tdlr_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by TDLR/SED-LR.")

    parser.add_argument("--attack_iters", default=defaults.attack_iters, type=int)
    parser.add_argument("--attack_lr", default=defaults.attack_lr, type=float)
    parser.add_argument(
        "--attack_restarts",
        default=1,
        type=int,
        help="Independent dummy-initialization restarts on one fixed observed leak; select by minimum attacker-visible loss.",
    )
    parser.add_argument(
        "--attack_restart_seed",
        default=10000,
        type=int,
        help="Base seed for dummy-initialization restarts when --attack_restarts is greater than one.",
    )
    parser.add_argument("--attack_optimizer", choices=("adam", "adamw"), default="adam",
                        help="Optimizer used for dummy reconstruction variables.")
    parser.add_argument("--grad_match", choices=("normalized", "mse", "mixed", "global_mse", "global_normalized", "global_mixed"), default="mixed",
                        help="How to compare leaks. mse preserves update magnitude; mixed adds direction matching.")
    parser.add_argument("--local_update_steps", default=1, type=int,
                        help="Number of differentiable local update steps to simulate for model_update attacks.")
    parser.add_argument("--local_update_lr", default=1e-3, type=float,
                        help="Client learning rate used inside model_update attack simulation.")
    parser.add_argument("--local_update_wd", default=None, type=float,
                        help="Weight decay used inside model_update attack simulation. Defaults to --wd.")
    parser.add_argument("--fedtps_update_scope", choices=("patterns", "full"), default="patterns",
                        help="FedTPS model_update leak scope: realistic uploaded Patterns or full-model upper bound.")
    parser.add_argument("--fedtps_update_optimizer", choices=("sgd", "adam"), default="adam",
                        help="Optimizer simulated inside FedTPS model_update leak.")
    parser.add_argument("--fedtps_include_agg_patterns", choices=(0, 1), default=1, type=int,
                        help="1 also matches FedTPS server-returned top-k aggregated Patterns proxy.")
    parser.add_argument("--fedtps_agg_patterns_weight", default=0.05, type=float,
                        help="Weight for matching the FedTPS aggregated Patterns proxy.")
    parser.add_argument("--fedtps_k", default=2, type=int,
                        help="FedTPS top-k pattern count used by server-side pattern aggregation.")
    parser.add_argument("--fedtps_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by FedTPS.")
    parser.add_argument("--fedtps_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by FedTPS.")
    parser.add_argument("--fedtps_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by FedTPS.")
    parser.add_argument("--fed4tp_attack_surface", choices=("weights", "weights_mask"), default="weights_mask",
                        help="Fed4TP leak surface: uploaded weights only, or weights plus MPL Top-K mask.")
    parser.add_argument("--fed4tp_mask_weight", default=1.0, type=float,
                        help="Weight for matching Fed4TP's uploaded MPL Top-K mask leak.")
    parser.add_argument("--fed4tp_mask_temperature", default=0.1, type=float,
                        help="Temperature for the differentiable surrogate of Fed4TP's hard Top-K mask.")
    parser.add_argument("--fed4tp_update_optimizer", choices=("sgd", "adam"), default="adam",
                        help="Optimizer simulated inside Fed4TP model_update leak. Fed4TP training uses Adam in fate_main.py.")
    parser.add_argument("--fed4tp_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by Fed4TP.")
    parser.add_argument("--fed4tp_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by Fed4TP.")
    parser.add_argument("--fed4tp_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by Fed4TP.")
    parser.add_argument("--fedagat_update_optimizer", choices=("sgd", "adam"), default="adam",
                        help="Optimizer simulated inside FedAGAT model_update leak.")
    parser.add_argument("--fedagat_adam_beta1", default=0.9, type=float,
                        help="Beta1 for the differentiable Adam simulation used by FedAGAT.")
    parser.add_argument("--fedagat_adam_beta2", default=0.999, type=float,
                        help="Beta2 for the differentiable Adam simulation used by FedAGAT.")
    parser.add_argument("--fedagat_adam_eps", default=1e-8, type=float,
                        help="Epsilon for the differentiable Adam simulation used by FedAGAT.")
    parser.add_argument("--fedagat_loss_start_step", default=0, type=int,
                        help="FedAGAT loss batch_count offset used for dynamic graph regularization.")
    parser.add_argument("--fedagat_batches_per_epoch", default=0, type=int,
                        help="FedAGAT batches_per_epoch for dynamic graph regularization. 0 infers from data.")
    parser.add_argument("--fedagat_train_batch_size", default=64, type=int,
                        help="Training batch size used to infer FedAGAT batches_per_epoch when not set.")
    parser.add_argument("--fedagat_max_epochs", default=5, type=int,
                        help="FedAGAT max_epochs used in its dynamic graph regularization schedule.")
    parser.add_argument("--stagcn_ec_update_optimizer", choices=("sgd", "adam"), default="sgd",
                        help="Optimizer simulated for STAGCN-EC malicious_rsu_update/model_update upper-bound.")
    parser.add_argument("--stagcn_ec_adam_beta1", default=0.9, type=float,
                        help="Beta1 for STAGCN-EC differentiable Adam update simulation.")
    parser.add_argument("--stagcn_ec_adam_beta2", default=0.999, type=float,
                        help="Beta2 for STAGCN-EC differentiable Adam update simulation.")
    parser.add_argument("--stagcn_ec_adam_eps", default=1e-8, type=float,
                        help="Epsilon for STAGCN-EC differentiable Adam update simulation.")
    parser.add_argument("--traffic_prior", choices=(0, 1), default=0, type=int,
                        help="1 adds traffic-domain priors during dummy_x optimization.")
    parser.add_argument("--traffic_range_weight", default=1e-2, type=float,
                        help="Penalty weight for keeping dummy_x inside the training value range.")
    parser.add_argument("--traffic_nonnegative_weight", default=1e-2, type=float,
                        help="Penalty weight for nonnegative physical traffic values.")
    parser.add_argument("--traffic_temporal_weight", default=1e-3, type=float,
                        help="Penalty weight for temporal smoothness.")
    parser.add_argument("--traffic_spatial_weight", default=1e-4, type=float,
                        help="Penalty weight for graph-neighbor smoothness.")
    parser.add_argument("--traffic_std_weight", default=0.0, type=float,
                        help="Penalty weight for preventing dummy_x from collapsing below training-set variation.")
    parser.add_argument("--traffic_std_ratio", default=1.0, type=float,
                        help="Fraction of training-set input std used as the dummy_x variation floor.")
    parser.add_argument("--traffic_prior_samples", default=256, type=int,
                        help="Number of training samples used to estimate traffic prior statistics for dataset objects.")
    parser.add_argument("--dummy_y_mode", choices=("optimize", "real", "zeros"), default="optimize",
                        help="optimize is realistic for unknown regression targets; real is an oracle diagnostic.")
    parser.add_argument("--log_interval", default=50, type=int,
                        help="Print attack progress every N rounds. 0 disables progress logs.")
    parser.add_argument("--early_stop_patience", default=0, type=int,
                        help="Stop if attack loss does not improve for this many rounds. 0 disables it.")
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float,
                        help="Minimum attack-loss decrease counted as an improvement.")
    parser.add_argument(
        "--attack_checkpoint_interval",
        default=0,
        type=int,
        help="Save dummy reconstruction checkpoint every N attack rounds. 0 disables it.",
    )
    parser.add_argument(
        "--resume_attack_checkpoint",
        default="",
        type=str,
        help="Optional attack checkpoint .pt file to resume dummy_x/dummy_y optimization from.",
    )
    parser.add_argument(
        "--resume_optimizer_state",
        choices=(0, 1),
        default=0,
        type=int,
        help="1 resumes Adam state from checkpoint; 0 resumes dummy tensors only and starts a fresh optimizer.",
    )
    parser.add_argument("--init", choices=("randn", "zeros"), default="randn", type=str)
    parser.add_argument("--optimize_dummy_y", action="store_true",
                        help="Force optimizing dummy_y for gradient-style attacks.")
    parser.add_argument("--no_optimize_dummy_y", action="store_true",
                        help="Disable dummy_y optimization even if the adapter requests it.")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--checkpoint", default="", type=str,
                        help="Optional model checkpoint/state_dict to load before attacking.")
    parser.add_argument("--result_dir", default=defaults.result_dir, type=str)
    parser.add_argument(
        "--result_dir_flat",
        action="store_true",
        default=False,
        help="Write outputs directly into --result_dir instead of adding a model-name subdirectory.",
    )
    parser.add_argument(
        "--result_tag",
        default="",
        type=str,
        help="Optional suffix used to distinguish output files from otherwise identical attacks.",
    )
    parser.add_argument(
        "--mape_eps",
        default=defaults.mape_eps,
        type=float,
        help="Ignore target values with abs(value) <= this threshold when computing masked MAPE.",
    )
    parser.add_argument("--save_reconstruction", action="store_true")

    return parser
