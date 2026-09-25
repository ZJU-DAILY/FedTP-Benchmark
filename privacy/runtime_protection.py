"""Central DP, HE-TTP and communication-audit wrappers for FATE contexts."""
from __future__ import annotations

import time
import zlib
from typing import Any, Mapping

import torch

from privacy.he_backend import ciphertext_bytes, decrypt_tree, encrypt_tree, export_public_key, generate_keypair, import_public_key
from privacy.protection import gaussian_dp_upload, l2_norm
from privacy.attack_trace import capture_he_ttp_upload


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _plain_tensor_bytes(value: Any) -> int:
    """Byte count for a plaintext tensor tree transmitted by FATE."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_plain_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_plain_tensor_bytes(item) for item in value)
    return 0


def _payload_type(tag: str) -> str:
    tag = tag.lower()
    # Delta tags must be classified before the broad prototype aliases below:
    # ``tdlr_dp_delta_*`` contains ``r_`` as part of its model name.
    if "delta" in tag:
        return "model_update"
    if any(item in tag for item in ("grad", "g_")):
        return "gradient"
    if any(item in tag for item in ("activation", "agg", "h_time", "hs", "htau")):
        return "activation"
    if any(item in tag for item in ("prototype", "pr_", "nr_", "r_")):
        return "prototype"
    if any(item in tag for item in ("embed", "z_", "client_z")):
        return "embedding"
    if any(item in tag for item in ("weight", "w_", "payload", "local_gru", "v_")):
        return "model_update"
    return "hidden_state"


def _communication_phase(tag: str | None) -> str:
    """Classify an HE message for correct one-round efficiency extrapolation."""
    normalized = str(tag or "").lower()
    if (
        normalized.startswith("init")
        or "tsvd" in normalized
        or "neighbor_weights" in normalized
    ):
        return "initialization"
    # Several legacy split-learning loops abbreviate the phase as ``te_s0``
    # (instead of spelling out ``test``).  Treat it as test traffic so that
    # test activations/gradients are never added to one-round train traffic.
    if (
        "test" in normalized
        or normalized.startswith(("te_", "te-"))
        or "_te_" in normalized
        or "-te-" in normalized
    ):
        return "test"
    if (
        "val" in normalized
        or "valid" in normalized
        or normalized.startswith(("va_", "va-"))
        or "_va_" in normalized
        or "-va-" in normalized
    ):
        return "validation"
    return "train"


def _record_he_ttp_phase_bytes(args: Any, direction: str, tag: str | None, byte_count: int) -> None:
    """Keep phase totals in addition to the legacy all-message total."""
    phase = _communication_phase(tag)
    attr = f"_he_ttp_{direction}_bytes_by_phase"
    totals = dict(getattr(args, attr, {}) or {})
    totals[phase] = int(totals.get(phase, 0)) + int(byte_count)
    setattr(args, attr, totals)


def _record_he_ttp_phase_seconds(args: Any, tag: str | None, elapsed: float) -> None:
    phase = _communication_phase(tag)
    totals = dict(getattr(args, "_he_ttp_encrypt_seconds_by_phase", {}) or {})
    totals[phase] = float(totals.get(phase, 0.0)) + float(elapsed)
    args._he_ttp_encrypt_seconds_by_phase = totals


def _audit(ctx: Any, direction: str, tag: str, value: Any) -> None:
    if _contains_tensor(value):
        print(
            f"[PrivacyAudit] rank={getattr(ctx, 'rank', '?')} direction={direction} tag={tag} "
            f"type={_payload_type(tag)} l2={l2_norm(value).item():.8f}", flush=True,
        )


def _should_log_dp_upload(args: Any, tag: str) -> bool:
    """Avoid multi-GB logs from per-batch DP split-learning uploads.

    Calibration is logged separately by every trainer.  For ordinary upload
    diagnostics, the first message of each semantic type is sufficient unless
    the caller explicitly asks for periodic samples.
    """
    key = _payload_type(tag)
    counts = dict(getattr(args, "_privacy_dp_upload_log_counts", {}) or {})
    count = int(counts.get(key, 0))
    counts[key] = count + 1
    args._privacy_dp_upload_log_counts = counts
    interval = max(0, int(getattr(args, "dp_upload_log_interval", 0)))
    return count == 0 or (interval > 0 and (count + 1) % interval == 0)


def _wrap_put(party: Any, transform, ctx: Any, direction: str) -> None:
    original = party.put
    def wrapped(tag: str, value: Any, *args: Any, **kwargs: Any):
        _audit(ctx, direction, tag, value)
        return original(tag, transform(tag, value), *args, **kwargs)
    party.put = wrapped
    print(f"[PrivacyRuntime] rank={getattr(ctx, 'rank', '?')} installed_put_wrapper direction={direction}", flush=True)


def _wrap_get(party: Any, transform) -> None:
    original = party.get
    def wrapped(tag: str, *args: Any, **kwargs: Any):
        return transform(original(tag, *args, **kwargs))
    party.get = wrapped


def _install_dp(ctx: Any, args: Any) -> None:
    if bool(getattr(ctx, "is_on_arbiter", False)):
        return
    if float(args.dp_clip_norm) <= 0:
        print("[PrivacyRuntime] DP clip norm will be calibrated automatically from first-round delta norms.", flush=True)
        return
    party = ctx.arbiter
    if getattr(party, "_privacy_dp_put_wrapped", False):
        return
    base_seed, rank, message_index = int(args.dp_noise_seed), int(ctx.rank), 0
    original = party.put
    def wrapped(tag: str, value: Any, *put_args: Any, **put_kwargs: Any):
        nonlocal message_index
        # Some custom FL loops first exchange a data-independent, public
        # model initialisation.  It is a protocol synchronisation message,
        # not a client update, and therefore must neither be clipped nor
        # noised.  In particular, perturbing it would make clients start a
        # delta round from different reference models.
        if str(tag).startswith((
            "fedgode_public_init_", "refol_public_init_", "fedagat_public_init_",
            "ufcl_dp_public_init", "ufcl_dp_initial_reference",
        )):
            return original(tag, value, *put_args, **put_kwargs)
        log_this_upload = _should_log_dp_upload(args, tag)
        if log_this_upload:
            _audit(ctx, "client_to_arbiter", tag, value)
        if not _contains_tensor(value):
            return original(tag, value, *put_args, **put_kwargs)
        generator = None
        if base_seed >= 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + rank * 1_000_003 + message_index)
        message_index += 1
        protected, info = gaussian_dp_upload(
            value, clip_norm=float(args.dp_clip_norm), sigma=float(args.dp_sigma),
            post_clip_ratio=float(getattr(args, "dp_post_clip_ratio", 0.0)), generator=generator,
            experimental_noise_clip_ratio=float(getattr(args, "experimental_noise_clip_ratio", 0.0)),
            experimental_skip_update_clip=bool(getattr(args, "experimental_skip_update_clip", False)),
        )
        if log_this_upload:
            print(f"[DPUpload] rank={rank} tag={tag} type={_payload_type(tag)} l2={info['upload_l2_norm']:.8f} clip={info['clip_coefficient']:.8f} noise_std={info['noise_std']:.8f} noise_l2={info['noise_l2_norm']:.8f} noise_clip={info['experimental_noise_clip_coefficient']:.8f} post_l2={info['post_noise_l2_norm']:.8f} post_clip={info['post_clip_coefficient']:.8f} final_l2={info['final_upload_l2_norm']:.8f}", flush=True)
        return original(tag, protected, *put_args, **put_kwargs)
    party.put = wrapped
    party._privacy_original_put = original
    party._privacy_dp_put_wrapped = True


def protected_arbiter_put(ctx: Any, args: Any, tag: str, value: Any, *, clip_norm: float | None = None,
                          return_protected_payload: bool = False) -> Any:
    """Explicit client upload path for FATE versions that recreate Party proxies.

    Unlike instance monkey-patching, call sites using this function are stable
    even when ``ctx.arbiter`` returns a fresh proxy for every access.
    """
    protection = str(getattr(args, "protection", "plain")).lower()
    log_this_upload = protection == "dp" and _should_log_dp_upload(args, tag)
    if bool(getattr(args, "privacy_audit", False)) and (protection != "dp" or log_this_upload):
        _audit(ctx, "client_to_arbiter", tag, value)
    dp_info = None
    if protection == "dp" and _contains_tensor(value):
        selected_clip_norm = float(args.dp_clip_norm if clip_norm is None else clip_norm)
        if selected_clip_norm <= 0:
            raise ValueError("--dp_clip_norm must be > 0 when --protection dp")
        seed = int(getattr(args, "dp_noise_seed", -1))
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu")
            # Different round tags need different noise even when the
            # experiment seed is fixed. CRC32 is stable across processes.
            tag_offset = zlib.crc32(tag.encode("utf-8"))
            generator.manual_seed(seed + int(getattr(ctx, "rank", 0)) * 1_000_003 + tag_offset)
        value, dp_info = gaussian_dp_upload(
            value, clip_norm=selected_clip_norm, sigma=float(args.dp_sigma),
            post_clip_ratio=float(getattr(args, "dp_post_clip_ratio", 0.0)), generator=generator,
            experimental_noise_clip_ratio=float(getattr(args, "experimental_noise_clip_ratio", 0.0)),
            experimental_skip_update_clip=bool(getattr(args, "experimental_skip_update_clip", False)),
        )
        if log_this_upload:
            print(f"[DPUpload] rank={getattr(ctx, 'rank', '?')} tag={tag} type={_payload_type(tag)} l2={dp_info['upload_l2_norm']:.8f} clip={dp_info['clip_coefficient']:.8f} noise_std={dp_info['noise_std']:.8f} noise_l2={dp_info['noise_l2_norm']:.8f} noise_clip={dp_info['experimental_noise_clip_coefficient']:.8f} post_l2={dp_info['post_noise_l2_norm']:.8f} post_clip={dp_info['post_clip_coefficient']:.8f} final_l2={dp_info['final_upload_l2_norm']:.8f}", flush=True)
    elif protection == "he" and _contains_tensor(value):
        result = protected_he_ttp_arbiter_put(ctx, args, tag, value)
        return (result, value, None) if return_protected_payload else result
    # Do not apply DP twice when this Party proxy is also wrapped by
    # ``_install_dp``.  Some FATE versions create fresh proxies, so retain
    # the normal ``put`` fallback when the original method is unavailable.
    party = ctx.arbiter
    raw_put = getattr(party, "_privacy_original_put", None)
    if raw_put is not None:
        result = raw_put(tag, value)
    else:
        result = party.put(tag, value)
    return (result, value, dp_info) if return_protected_payload else result


def _he_ttp_encrypt_floating_tree(value: Any, context: Any, slot_count: int) -> Any:
    """CKKS-encrypt only floating tensors; keep protocol metadata public."""
    from privacy.ckks_backend import encrypt_tree as ckks_encrypt_tree
    if torch.is_tensor(value):
        return (
            ckks_encrypt_tree(value, context, slot_count=slot_count)
            if torch.is_floating_point(value) else value
        )
    if isinstance(value, Mapping):
        return {key: _he_ttp_encrypt_floating_tree(item, context, slot_count) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_he_ttp_encrypt_floating_tree(item, context, slot_count) for item in value)
    if isinstance(value, list):
        return [_he_ttp_encrypt_floating_tree(item, context, slot_count) for item in value]
    return value


def protected_he_ttp_arbiter_put(ctx: Any, args: Any, tag: str, value: Any) -> Any:
    """Explicit HE-TTP upload for custom FATE trainers.

    Unlike monkey-patching ``ctx.arbiter.put``, this remains effective when
    FATE creates a fresh Party proxy for each access.
    """
    if str(getattr(args, "protection", "plain")).lower() != "he":
        return ctx.arbiter.put(tag, value)
    if str(getattr(args, "he_backend", "auto")).lower() != "he_ttp":
        return ctx.arbiter.put(tag, value)
    if str(getattr(args, "he_scheme", "ckks")).lower() != "ckks":
        raise RuntimeError("Explicit HE-TTP uploads require --he_scheme ckks.")
    # This writes ciphertext-visible metadata only by default.  The optional
    # plaintext-equivalent trace is isolated as an explicitly named insider
    # study, never as the external HE attack input.
    capture_he_ttp_upload(ctx, args, tag, value)
    context = getattr(args, "_he_ttp_ckks_public_context", None)
    if context is None:
        raise RuntimeError("HE-TTP public context is unavailable on this client.")
    from privacy.ckks_backend import ciphertext_bytes as ckks_ciphertext_bytes
    started = time.perf_counter()
    encrypted = _he_ttp_encrypt_floating_tree(
        value, context, int(args.he_ckks_poly_modulus_degree) // 2,
    )
    elapsed = time.perf_counter() - started
    byte_count = ckks_ciphertext_bytes(encrypted)
    args._he_ttp_upload_bytes = int(getattr(args, "_he_ttp_upload_bytes", 0)) + byte_count
    _record_he_ttp_phase_bytes(args, "upload", tag, byte_count)
    args._he_ttp_encrypt_seconds = float(getattr(args, "_he_ttp_encrypt_seconds", 0.0)) + elapsed
    _record_he_ttp_phase_seconds(args, tag, elapsed)
    print(
        f"[HETTP] rank={ctx.rank} direction=client_to_arbiter tag={tag} "
        f"type={_payload_type(tag)} bytes={byte_count} encrypt_s={elapsed:.6f}",
        flush=True,
    )
    party = ctx.arbiter
    raw_put = getattr(party, "_privacy_original_put", None)
    return (raw_put or party.put)(tag, encrypted)


def unprotect_he_ttp_payload(args: Any, value: Any) -> Any:
    """Explicit Arbiter-side decode; safe when a wrapper already decoded it."""
    if (
        str(getattr(args, "protection", "plain")).lower() == "he"
        and str(getattr(args, "he_backend", "auto")).lower() == "he_ttp"
        and str(getattr(args, "he_scheme", "ckks")).lower() == "ckks"
    ):
        context = getattr(args, "_he_ttp_ckks_secret_context", None)
        if context is None:
            raise RuntimeError("HE-TTP secret context is unavailable on Arbiter.")
        from privacy.ckks_backend import decrypt_tree as ckks_decrypt_tree
        return ckks_decrypt_tree(value, context)
    return value


def record_he_ttp_downlink(args: Any, value: Any, *, tag: str | None = None) -> Any:
    """Record an explicit plaintext Arbiter->client HE-TTP response."""
    if (
        str(getattr(args, "protection", "plain")).lower() == "he"
        and str(getattr(args, "he_backend", "auto")).lower() == "he_ttp"
    ):
        byte_count = _plain_tensor_bytes(value)
        args._he_ttp_download_bytes = int(getattr(args, "_he_ttp_download_bytes", 0)) + byte_count
        _record_he_ttp_phase_bytes(args, "downlink", tag, byte_count)
        print(
            f"[HETTP] rank={getattr(args, '_privacy_rank', '?')} "
            f"direction=arbiter_to_client tag={tag or '<untagged>'} "
            f"type={_payload_type(str(tag or 'downlink'))} bytes={byte_count}",
            flush=True,
        )
    return value


def _install_he_ttp(ctx: Any, args: Any) -> None:
    """Encrypt client uploads for a trusted-arbiter protocol.

    CKKS is the practical default: it packs a full tensor into ciphertext
    chunks, unlike the legacy Paillier implementation which encrypts every
    scalar separately and is unusable for activation-heavy split protocols.
    Public boolean/integer protocol metadata is deliberately left plaintext.
    """
    scheme = str(getattr(args, "he_scheme", "ckks")).lower()
    if scheme == "ckks":
        from privacy.ckks_backend import (
            ciphertext_bytes as ckks_ciphertext_bytes,
            decrypt_tree as ckks_decrypt_tree,
            encrypt_tree as ckks_encrypt_tree,
            export_public_context as ckks_export_public_context,
            generate_context as ckks_generate_context,
            import_context as ckks_import_context,
        )

        key_tag = "__privacy_he_ttp_ckks_context"
        if bool(getattr(ctx, "is_on_arbiter", False)):
            secret_context = ckks_generate_context(
                int(args.he_ckks_poly_modulus_degree), int(args.he_ckks_scale_bits),
            )
            args._he_ttp_ckks_secret_context = secret_context
            public_context = ckks_export_public_context(secret_context)
            ctx.guest.put(key_tag, public_context)
            ctx.hosts.put(key_tag, public_context)
            _wrap_get(ctx.guest, lambda value: ckks_decrypt_tree(value, secret_context))
            _wrap_get(ctx.hosts, lambda value: ckks_decrypt_tree(value, secret_context))
            if int(args.he_encrypt_return):
                raise RuntimeError("HE-TTP client return encryption is not configured.")
            print(
                f"[HETTP] packed CKKS arbiter context ready "
                f"poly_degree={args.he_ckks_poly_modulus_degree}", flush=True,
            )
            return

        args._he_ttp_upload_bytes = 0
        args._he_ttp_download_bytes = 0
        args._he_ttp_upload_bytes_by_phase = {}
        args._he_ttp_downlink_bytes_by_phase = {}
        args._he_ttp_encrypt_seconds = 0.0
        args._he_ttp_encrypt_seconds_by_phase = {}
        # ``record_he_ttp_downlink`` is intentionally context-free because
        # trainers call it after FATE has recreated a Party proxy.  Retain
        # the client identity on args so every audited downlink is still
        # attributable to exactly one recipient.
        args._privacy_rank = int(getattr(ctx, "rank", -1))
        public_context_payload = ctx.arbiter.get(key_tag)
        # The public CKKS context is a real one-time Arbiter -> client setup
        # transmission.  Count it once for each recipient, separately from
        # round traffic; previously it was silently omitted from HE-TTP
        # communication metrics.
        context_bytes = len(public_context_payload) if isinstance(public_context_payload, (bytes, bytearray)) else 0
        args._he_ttp_download_bytes += context_bytes
        _record_he_ttp_phase_bytes(args, "downlink", "init_ckks_public_context", context_bytes)
        print(
            f"[HETTP] rank={ctx.rank} direction=arbiter_to_client "
            f"tag=init_ckks_public_context type=context bytes={context_bytes}",
            flush=True,
        )
        public_context = ckks_import_context(public_context_payload)
        args._he_ttp_ckks_public_context = public_context

        def encrypt(tag: str, value: Any) -> Any:
            if not _contains_tensor(value):
                return value
            capture_he_ttp_upload(ctx, args, tag, value)
            started = time.perf_counter()
            result = _he_ttp_encrypt_floating_tree(
                value, public_context, int(args.he_ckks_poly_modulus_degree) // 2,
            )
            elapsed = time.perf_counter() - started
            args._he_ttp_upload_bytes += ckks_ciphertext_bytes(result)
            _record_he_ttp_phase_bytes(args, "upload", tag, ckks_ciphertext_bytes(result))
            args._he_ttp_encrypt_seconds += elapsed
            _record_he_ttp_phase_seconds(args, tag, elapsed)
            print(
                f"[HETTP] rank={ctx.rank} direction=client_to_arbiter tag={tag} "
                f"type={_payload_type(tag)} bytes={ckks_ciphertext_bytes(result)} "
                f"encrypt_s={elapsed:.6f}", flush=True,
            )
            return result

        _wrap_put(ctx.arbiter, encrypt, ctx, "client_to_arbiter")

        return

    if scheme != "paillier":
        raise ValueError(f"Unsupported HE-TTP scheme: {scheme}")

    # Legacy scalar Paillier path, retained only for reproducibility.
    key_tag = "__privacy_he_public_key"
    if bool(getattr(ctx, "is_on_arbiter", False)):
        public_key, private_key = generate_keypair(int(args.he_key_bits))
        ctx.guest.put(key_tag, export_public_key(public_key))
        ctx.hosts.put(key_tag, export_public_key(public_key))
        def decrypt(value: Any) -> Any:
            return decrypt_tree(value, private_key)
        _wrap_get(ctx.guest, decrypt)
        _wrap_get(ctx.hosts, decrypt)
        if int(args.he_encrypt_return):
            raise RuntimeError(
                "HE-TTP encrypted return needs a separate client keypair/threshold-decryption protocol; "
                "it is intentionally disabled until that protocol is installed."
            )
        return
    public_key = import_public_key(ctx.arbiter.get(key_tag))
    def encrypt(tag: str, value: Any) -> Any:
        if not _contains_tensor(value):
            return value
        started = time.perf_counter(); result = encrypt_tree(value, public_key)
        print(f"[HETTP] rank={ctx.rank} direction=client_to_arbiter tag={tag} type={_payload_type(tag)} bytes={ciphertext_bytes(result)} encrypt_s={time.perf_counter()-started:.6f}", flush=True)
        return result
    _wrap_put(ctx.arbiter, encrypt, ctx, "client_to_arbiter")


def install_runtime_protection(ctx: Any, args: Any) -> None:
    protection = str(getattr(args, "protection", "plain")).lower()
    print(
        f"[PrivacyRuntime] rank={getattr(ctx, 'rank', '?')} protection={protection} "
        f"audit={bool(getattr(args, 'privacy_audit', False))} "
        f"he_backend={getattr(args, 'he_backend', 'unset')} "
        f"dp_post_clip_ratio={float(getattr(args, 'dp_post_clip_ratio', 0.0)):.6f} "
        f"experimental_noise_clip_ratio={float(getattr(args, 'experimental_noise_clip_ratio', 0.0)):.6f} "
        f"experimental_skip_update_clip={bool(getattr(args, 'experimental_skip_update_clip', False))}",
        flush=True,
    )
    if float(getattr(args, "experimental_noise_clip_ratio", 0.0)) > 0 or bool(getattr(args, "experimental_skip_update_clip", False)):
        print(
            "[PrivacyRuntime] WARNING: noise-only clipping/raw-update mode is an experimental "
            "non-standard perturbation and must not be reported as Gaussian DP.",
            flush=True,
        )
    if bool(getattr(args, "privacy_audit", False)) and protection == "plain" and not bool(getattr(ctx, "is_on_arbiter", False)):
        print("[PrivacyRuntime] explicit protected_arbiter_put is required by this FATE Party proxy.", flush=True)
    elif protection == "dp":
        _install_dp(ctx, args)
    elif protection == "he":
        backend = str(getattr(args, "he_backend", "auto"))
        if backend in ("auto", "he_ttp"):
            _install_he_ttp(ctx, args)
        elif backend == "he_sa":
            # Stock FATE FedAVG consumes ``aggregator='secure_aggregate'``
            # from get_setting().  Custom loops must be classified HE-TTP
            # unless their aggregation point is explicitly migrated.
            print("[HESA] Delegating aggregation to FATE secure_aggregate backend.", flush=True)
        else:
            raise ValueError(f"Unsupported HE backend: {backend}")
