"""Packed CKKS backend for practical HE secure aggregation.

Each CKKS ciphertext carries a vector chunk instead of one scalar.  Clients
receive a public-only context; the Arbiter owns the secret context and decrypts
only the homomorphically aggregated update.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch


# FATE's federation deserializer accepts ordinary dict/list/bytes values but
# deliberately rejects arbitrary user classes.  Keep every ciphertext payload
# in this plain, serializable mapping rather than a dataclass.
CKKSPackedTensor = dict[str, Any]
_PACKED_MARKER = "__privacy_ckks_packed__"


def _is_packed(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get(_PACKED_MARKER) == 1


def _ts():
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError(
            "Packed HE requires TenSEAL. In the FATE environment run: pip install tenseal"
        ) from exc
    return ts


def generate_context(poly_modulus_degree: int = 8192, global_scale_bits: int = 40):
    ts = _ts()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        int(poly_modulus_degree),
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = float(2 ** int(global_scale_bits))
    return context


def export_public_context(context: Any) -> bytes:
    return context.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False,
    )


def import_context(payload: bytes):
    return _ts().context_from(payload)


def _slot_count(slot_count: int) -> int:
    # TenSEAL does not expose this uniformly across releases.  CKKS has
    # poly_modulus_degree / 2 slots, so the caller supplies that stable value.
    slots = int(slot_count)
    if slots <= 0:
        raise RuntimeError("CKKS context reports no available slots")
    return slots


def encrypt_tensor(
    tensor: torch.Tensor,
    context: Any,
    progress: Callable[[int], None] | None = None,
    slot_count: int = 4096,
) -> CKKSPackedTensor:
    source = tensor.detach().cpu().contiguous()
    flat = source.to(torch.float64).reshape(-1)
    slots = _slot_count(slot_count)
    chunks: list[bytes] = []
    lengths: list[int] = []
    ts = _ts()
    for start in range(0, flat.numel(), slots):
        chunk = flat[start:start + slots]
        lengths.append(int(chunk.numel()))
        chunks.append(ts.ckks_vector(context, chunk.tolist()).serialize())
        if progress is not None:
            progress(int(chunk.numel()))
    return {
        _PACKED_MARKER: 1,
        "shape": list(source.shape),
        "dtype": str(source.dtype).replace("torch.", ""),
        "chunk_lengths": lengths,
        "ciphertexts": chunks,
    }


def decrypt_tensor(payload: CKKSPackedTensor, context: Any) -> torch.Tensor:
    ts = _ts()
    values: list[float] = []
    for blob, length in zip(payload["ciphertexts"], payload["chunk_lengths"]):
        values.extend(ts.ckks_vector_from(context, blob).decrypt()[:length])
    dtype = getattr(torch, payload["dtype"], torch.float32)
    return torch.tensor(values, dtype=torch.float64).reshape(payload["shape"]).to(dtype)


def encrypt_tree(
    value: Any,
    context: Any,
    progress: Callable[[int], None] | None = None,
    slot_count: int = 4096,
) -> Any:
    if torch.is_tensor(value):
        return encrypt_tensor(value, context, progress, slot_count)
    if isinstance(value, Mapping):
        return {key: encrypt_tree(item, context, progress, slot_count) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(encrypt_tree(item, context, progress, slot_count) for item in value)
    if isinstance(value, list):
        return [encrypt_tree(item, context, progress, slot_count) for item in value]
    return value


def decrypt_tree(value: Any, context: Any) -> Any:
    if _is_packed(value):
        return decrypt_tensor(value, context)
    if isinstance(value, Mapping):
        return {key: decrypt_tree(item, context) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(decrypt_tree(item, context) for item in value)
    if isinstance(value, list):
        return [decrypt_tree(item, context) for item in value]
    return value


def homomorphic_sum_tree(values: list[Any], context: Any) -> Any:
    if not values:
        raise ValueError("Cannot aggregate an empty CKKS payload list")
    first = values[0]
    if _is_packed(first):
        if not all(_is_packed(item) for item in values):
            raise TypeError("CKKS aggregation received mixed encrypted/plain payloads")
        if not all(
            item["shape"] == first["shape"]
            and item["dtype"] == first["dtype"]
            and item["chunk_lengths"] == first["chunk_lengths"]
            for item in values
        ):
            raise ValueError("CKKS aggregation received incompatible packed tensor metadata")
        ts = _ts()
        ciphertexts = []
        for index in range(len(first["ciphertexts"])):
            total = ts.ckks_vector_from(context, first["ciphertexts"][index])
            for item in values[1:]:
                total += ts.ckks_vector_from(context, item["ciphertexts"][index])
            ciphertexts.append(total.serialize())
        return {
            _PACKED_MARKER: 1,
            "shape": list(first["shape"]),
            "dtype": first["dtype"],
            "chunk_lengths": list(first["chunk_lengths"]),
            "ciphertexts": ciphertexts,
        }
    if isinstance(first, Mapping):
        keys = list(first.keys())
        if not all(isinstance(item, Mapping) and list(item.keys()) == keys for item in values):
            raise ValueError("CKKS aggregation received incompatible update dictionaries")
        return {key: homomorphic_sum_tree([item[key] for item in values], context) for key in keys}
    if isinstance(first, (tuple, list)):
        length = len(first)
        if not all(isinstance(item, type(first)) and len(item) == length for item in values):
            raise ValueError("CKKS aggregation received incompatible sequence payloads")
        aggregate = [homomorphic_sum_tree([item[index] for item in values], context) for index in range(length)]
        return tuple(aggregate) if isinstance(first, tuple) else aggregate
    raise TypeError(f"Unsupported CKKS aggregate payload type: {type(first)!r}")


def homomorphic_weighted_sum_tree(values: list[Any], weights: list[float], context: Any) -> Any:
    """Compute ``sum_i weights[i] * Enc(values[i])`` without decryption.

    Weights are public protocol coefficients; only the model payload stays
    encrypted.  This is used by temporal/asynchronous weighted FedAvg.
    """
    if not values or len(values) != len(weights):
        raise ValueError("CKKS weighted aggregation requires aligned non-empty values and weights")
    first = values[0]
    if _is_packed(first):
        if not all(_is_packed(item) for item in values):
            raise TypeError("CKKS weighted aggregation received mixed encrypted/plain payloads")
        if not all(
            item["shape"] == first["shape"] and item["dtype"] == first["dtype"]
            and item["chunk_lengths"] == first["chunk_lengths"] for item in values
        ):
            raise ValueError("CKKS weighted aggregation received incompatible packed tensor metadata")
        ts = _ts()
        ciphertexts = []
        for index in range(len(first["ciphertexts"])):
            total = ts.ckks_vector_from(context, first["ciphertexts"][index]) * float(weights[0])
            for item, weight in zip(values[1:], weights[1:]):
                total += ts.ckks_vector_from(context, item["ciphertexts"][index]) * float(weight)
            ciphertexts.append(total.serialize())
        return {
            _PACKED_MARKER: 1,
            "shape": list(first["shape"]), "dtype": first["dtype"],
            "chunk_lengths": list(first["chunk_lengths"]), "ciphertexts": ciphertexts,
        }
    if isinstance(first, Mapping):
        keys = list(first.keys())
        if not all(isinstance(item, Mapping) and list(item.keys()) == keys for item in values):
            raise ValueError("CKKS weighted aggregation received incompatible update dictionaries")
        return {
            key: homomorphic_weighted_sum_tree([item[key] for item in values], weights, context)
            for key in keys
        }
    if isinstance(first, (tuple, list)):
        length = len(first)
        if not all(isinstance(item, type(first)) and len(item) == length for item in values):
            raise ValueError("CKKS aggregation received incompatible sequence payloads")
        aggregate = [
            homomorphic_weighted_sum_tree([item[index] for item in values], weights, context)
            for index in range(length)
        ]
        return tuple(aggregate) if isinstance(first, tuple) else aggregate
    raise TypeError(f"Unsupported CKKS weighted aggregate payload type: {type(first)!r}")


def ciphertext_bytes(value: Any) -> int:
    if _is_packed(value):
        return sum(len(blob) for blob in value["ciphertexts"])
    if isinstance(value, Mapping):
        return sum(ciphertext_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(ciphertext_bytes(item) for item in value)
    return 0
