"""Cheddar-backend twin of test_complex_encode.py.

Runs Geo's complex-lane encoding tests against the Cheddar GPU backend
(EncodeComplex/DecodeComplex/Conjugate/ConjugateNew in
orion/backend/cheddar), plus a conjugation round-trip the lattigo file
doesn't cover. Requires the built _cheddar_native extension and a GPU.
"""

import functools

import torch

from orion.core.orion import scheme
from orion.backend.python.tensors import CipherTensor
from tests import test_complex_encode as base

CONFIG = {
    "ckks_params": dict(base.CONFIG["ckks_params"]),
    "orion": {
        **base.CONFIG["orion"],
        "backend": "cheddar",
        "device": "gpu",
    },
}


def _on_cheddar(fn):
    """Run one of the lattigo-file tests with the config swapped to
    Cheddar. The bodies read the module-global CONFIG at call time, so a
    temporary swap retargets them."""
    @functools.wraps(fn)
    def wrapper():
        original = base.CONFIG
        base.CONFIG = CONFIG
        try:
            fn()
        finally:
            base.CONFIG = original
    return wrapper


test_complex_encode_decode_round_trip = _on_cheddar(
    base.test_complex_encode_decode_round_trip)
test_complex_encrypt_decrypt_round_trip = _on_cheddar(
    base.test_complex_encrypt_decrypt_round_trip)
test_complex_ciphertext_times_real_plaintext = _on_cheddar(
    base.test_complex_ciphertext_times_real_plaintext)
test_complex_ciphertext_times_real_ciphertext = _on_cheddar(
    base.test_complex_ciphertext_times_real_ciphertext)
test_real_encode_is_unaffected = _on_cheddar(
    base.test_real_encode_is_unaffected)


def test_mul_imaginary_unit():
    """MulImaginaryUnit must rotate every slot by 90 degrees (x -> ix),
    key-free and without consuming a level."""
    scheme.init_scheme(CONFIG)

    try:
        values = torch.complex(
            torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, -4.0, 2.5], dtype=torch.float64),
        )

        ctxt = scheme.encrypt(scheme.encode(values))
        out_ids = [
            scheme.backend.MulImaginaryUnitNew(ct_id) for ct_id in ctxt.ids]
        out_ct = CipherTensor(
            scheme, out_ids, ctxt.shape, ctxt.on_shape,
            ctxt.start, ctxt.stride, ctxt.stop, ctxt.is_complex)

        decoded = scheme.decode(scheme.decrypt(out_ct))
        torch.testing.assert_close(
            decoded, values * 1j, atol=1e-2, rtol=1e-2)
    finally:
        scheme.delete_scheme()


def test_conjugate_round_trip():
    """conj(ct) must flip the imaginary lane and leave the real lane."""
    scheme.init_scheme(CONFIG)

    try:
        real = torch.tensor([1.5, -2.25, 3.0, 0.0], dtype=torch.float64)
        imag = torch.tensor([0.5, 4.0, -1.0, 2.0], dtype=torch.float64)
        values = torch.complex(real, imag)

        ctxt = scheme.encrypt(scheme.encode(values))

        conj_ids = [
            scheme.evaluator.conjugate(ct_id, in_place=False)
            for ct_id in ctxt.ids
        ]
        conj_ct = CipherTensor(
            scheme, conj_ids, ctxt.shape, ctxt.on_shape,
            ctxt.start, ctxt.stride, ctxt.stop, ctxt.is_complex)

        decoded = scheme.decode(scheme.decrypt(conj_ct))
        torch.testing.assert_close(
            decoded, values.conj(), atol=1e-2, rtol=1e-2)

        # In-place variant mutates the original ciphertext.
        for ct_id in ctxt.ids:
            scheme.evaluator.conjugate(ct_id, in_place=True)
        decoded_inplace = scheme.decode(scheme.decrypt(ctxt))
        torch.testing.assert_close(
            decoded_inplace, values.conj(), atol=1e-2, rtol=1e-2)
    finally:
        scheme.delete_scheme()
