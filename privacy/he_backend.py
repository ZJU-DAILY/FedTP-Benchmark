"""Optional Paillier payload backend used by HE-SA and HE-TTP experiments.

The implementation deliberately keeps ciphertexts opaque until the trusted
arbiter decrypts them.  It requires ``phe`` in the Linux/FATE environment;
Plain and DP runs do not depend on that package.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch


SCALE = 1_000_000


_PAILLIER_MARKER = "__privacy_paillier_tensor__"


def _is_paillier_tensor(value: Any) -> bool:
    """Recognise the FATE-serializable Paillier tensor wire format.

    FATE deliberately rejects arbitrary Python classes during inter-process
    deserialization.  Paillier ciphertexts must therefore be sent as ordinary
    dict/list/int values, never as a dataclass instance.
    """
    return isinstance(value, Mapping) and value.get(_PAILLIER_MARKER) == 1


def _paillier():
    try:
        from phe import paillier
    except ImportError as exc:
        raise RuntimeError(
            "HE mode requires the 'phe' package in the FATE runtime. Install it with: pip install phe"
        ) from exc
    return paillier


def generate_keypair(n_length: int):
    return _paillier().generate_paillier_keypair(n_length=n_length)


def export_public_key(public_key: Any) -> dict[str, int]:
    return {"n": int(public_key.n)}


def import_public_key(payload: Mapping[str, int]):
    return _paillier().PaillierPublicKey(n=int(payload["n"]))


def encrypt_tensor(tensor: torch.Tensor, public_key: Any) -> dict[str, Any]:
    source = tensor.detach().cpu().contiguous()
    scaled = torch.round(source.float() * SCALE).to(torch.int64).reshape(-1).tolist()
    encrypted = [int(public_key.encrypt(int(value)).ciphertext()) for value in scaled]
    return {
        _PAILLIER_MARKER: 1,
        "n": int(public_key.n),
        "exponent": 0,
        "shape": list(source.shape),
        "dtype": str(source.dtype).replace("torch.", ""),
        "values": encrypted,
    }


def decrypt_tensor(ciphertext: Mapping[str, Any], private_key: Any) -> torch.Tensor:
    paillier = _paillier()
    public_key = paillier.PaillierPublicKey(n=int(ciphertext["n"]))
    values = [
        private_key.decrypt(paillier.EncryptedNumber(public_key, int(value), int(ciphertext["exponent"])))
        for value in ciphertext["values"]
    ]
    dtype = getattr(torch, str(ciphertext["dtype"]), torch.float32)
    return torch.tensor(values, dtype=torch.float64).div(SCALE).reshape(tuple(ciphertext["shape"])).to(dtype)


def encrypt_tree(value: Any, public_key: Any, progress: Callable[[int], None] | None = None) -> Any:
    if torch.is_tensor(value):
        result = encrypt_tensor(value, public_key)
        if progress is not None:
            progress(int(value.numel()))
        return result
    if isinstance(value, Mapping):
        return {key: encrypt_tree(item, public_key, progress) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(encrypt_tree(item, public_key, progress) for item in value)
    if isinstance(value, list):
        return [encrypt_tree(item, public_key, progress) for item in value]
    return value


def decrypt_tree(value: Any, private_key: Any) -> Any:
    if _is_paillier_tensor(value):
        return decrypt_tensor(value, private_key)
    if isinstance(value, Mapping):
        return {key: decrypt_tree(item, private_key) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(decrypt_tree(item, private_key) for item in value)
    if isinstance(value, list):
        return [decrypt_tree(item, private_key) for item in value]
    return value


def homomorphic_sum_tree(values: list[Any]) -> Any:
    """Add identically structured Paillier payloads without decrypting them.

    This is deliberately limited to the encrypted floating-point update tree
    used by HE secure aggregation.  The secret key is never accepted here.
    """
    if not values:
        raise ValueError("Cannot homomorphically aggregate an empty payload list")
    first = values[0]
    if _is_paillier_tensor(first):
        if not all(_is_paillier_tensor(item) for item in values):
            raise TypeError("HE aggregation received mixed encrypted/plain tensor payloads")
        if not all(
            item["n"] == first["n"] and item["shape"] == first["shape"] and item["dtype"] == first["dtype"]
            for item in values
        ):
            raise ValueError("HE aggregation received incompatible encrypted tensor metadata")
        paillier = _paillier()
        public_key = paillier.PaillierPublicKey(n=int(first["n"]))
        encrypted_rows = [
            [paillier.EncryptedNumber(public_key, int(raw), int(item["exponent"])) for raw in item["values"]]
            for item in values
        ]
        summed = []
        for column in zip(*encrypted_rows):
            total = column[0]
            for encrypted_number in column[1:]:
                total = total + encrypted_number
            summed.append(int(total.ciphertext()))
        return {
            _PAILLIER_MARKER: 1,
            "n": int(first["n"]), "exponent": int(first["exponent"]),
            "shape": list(first["shape"]), "dtype": first["dtype"], "values": summed,
        }
    if isinstance(first, Mapping):
        expected_keys = list(first.keys())
        if not all(isinstance(item, Mapping) and list(item.keys()) == expected_keys for item in values):
            raise ValueError("HE aggregation received incompatible update dictionaries")
        return {key: homomorphic_sum_tree([item[key] for item in values]) for key in expected_keys}
    raise TypeError(f"Unsupported HE aggregate payload type: {type(first)!r}")


def homomorphic_weighted_sum_tree(values: list[Any], integer_weights: list[int]) -> Any:
    """Compute an integer-weighted encrypted sum without per-client decryption.

    Integer weights are used by weighted FedAvg variants: the arbiter decrypts
    only ``sum_i n_i * delta_i`` and divides by ``sum_i n_i`` in plaintext.
    """
    if len(values) != len(integer_weights) or not values:
        raise ValueError("Encrypted payloads and integer weights must be non-empty and aligned")
    first = values[0]
    if _is_paillier_tensor(first):
        if not all(_is_paillier_tensor(item) for item in values):
            raise TypeError("HE weighted aggregation received mixed encrypted/plain payloads")
        if not all(item["n"] == first["n"] and item["shape"] == first["shape"] and item["dtype"] == first["dtype"] for item in values):
            raise ValueError("HE weighted aggregation received incompatible encrypted tensor metadata")
        paillier = _paillier()
        public_key = paillier.PaillierPublicKey(n=int(first["n"]))
        encrypted_rows = [
            [paillier.EncryptedNumber(public_key, int(raw), int(item["exponent"])) for raw in item["values"]]
            for item in values
        ]
        weighted = []
        for column in zip(*encrypted_rows):
            total = column[0] * int(integer_weights[0])
            for encrypted_number, weight in zip(column[1:], integer_weights[1:]):
                total = total + encrypted_number * int(weight)
            weighted.append(int(total.ciphertext()))
        return {
            _PAILLIER_MARKER: 1,
            "n": int(first["n"]), "exponent": int(first["exponent"]),
            "shape": list(first["shape"]), "dtype": first["dtype"], "values": weighted,
        }
    if isinstance(first, Mapping):
        keys = list(first.keys())
        if not all(isinstance(item, Mapping) and list(item.keys()) == keys for item in values):
            raise ValueError("HE weighted aggregation received incompatible update dictionaries")
        return {key: homomorphic_weighted_sum_tree([item[key] for item in values], integer_weights) for key in keys}
    raise TypeError(f"Unsupported HE weighted aggregate payload type: {type(first)!r}")


def ciphertext_bytes(value: Any) -> int:
    if _is_paillier_tensor(value):
        return sum((int(item).bit_length() + 7) // 8 for item in value["values"])
    if isinstance(value, Mapping):
        return sum(ciphertext_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(ciphertext_bytes(item) for item in value)
    return 0
