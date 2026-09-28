"""Complex-lane packing for client uploads.

CKKS slots are complex, but Orion's normal encode path fills only the
real lane of each slot, so a real input vector of W values costs
ceil(W / slots) ciphertexts. Packing a second real vector into the
imaginary lane halves that count.

This module implements the cheap variant of that idea for the *upload*
of an already-plaintext-shaped input (HE-LRM's one-hot selector is the
motivating case): the client pairs ciphertext-sized chunks, and the
server splits the lanes back apart with one conjugation per packed
ciphertext before any layer runs. The technique is ported from the
HyDia encrypted-DB example (`examples/hydia`, `--packing complex`),
which uses it on its database rather than its query.

Pairing is done *per ciphertext chunk*, not by splitting the vector in
half: chunk 2m goes to the real lane and chunk 2m+1 to the imaginary
lane of packed ciphertext m. That keeps every unpacked ciphertext
covering exactly the same slot range it covered before, so the
consuming layer's BSGS diagonals, rotation distances and physical
widths are all untouched -- the server-side result is bit-for-bit the
same ciphertext list the unpacked path would have produced, modulo the
extra conjugation's key-switch noise.

The 0.5 pre-scale is what makes the extraction free: with
    p = 0.5 * (x + i*y)
    x = p + conj(p)        ( = 2*Re(p) )
    y = -i * (p - conj(p)) ( = 2*Im(p) )
both land back at the original magnitude, so no rescale and no
multiplicative level is consumed. Conjugation is one key-switch;
add/sub and multiply-by-i are key-free.

Not stackable with an IVE-style selector, which spends the imaginary
lane on its own query expansion.
"""

import torch

from orion.backend.python.tensors import CipherTensor


def plan_packing(num_elements, num_slots):
    """How many ciphertexts the plain and packed uploads would cost.

    Returns (num_real_cts, num_packed_cts). Packing is only profitable
    at >= 2 ciphertexts; at 1 the pair partner would be pure zero
    padding and the conjugation would be paid for nothing.
    """
    num_real_cts = -(-num_elements // num_slots)  # ceil
    num_packed_cts = -(-num_real_cts // 2)
    return num_real_cts, num_packed_cts


def is_profitable(num_elements, num_slots):
    num_real_cts, _ = plan_packing(num_elements, num_slots)
    return num_real_cts >= 2


def pack_lanes(values, num_slots):
    """Client side: fold a real vector into complex-lane chunk pairs.

    Returns a flat complex tensor of ceil(N/2) * num_slots entries,
    where N = ceil(values.numel() / num_slots), plus N itself so the
    server knows how many real ciphertexts to reconstruct (the last one
    may be a zero-padded partner).
    """
    flat = values.flatten().to(torch.float64)
    num_elements = flat.numel()

    pad = (-num_elements) % num_slots
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float64)])

    chunks = flat.view(-1, num_slots)
    num_real_cts = chunks.shape[0]
    if num_real_cts % 2:
        chunks = torch.cat(
            [chunks, torch.zeros(1, num_slots, dtype=torch.float64)])

    packed = 0.5 * (chunks[0::2] + 1j * chunks[1::2])
    return packed.flatten(), num_real_cts


def unpack_lanes(scheme, packed, num_real_cts, shape, on_shape=None):
    """Server side: conjugation lane-split back to real ciphertexts.

    `packed` is the CipherTensor the client uploaded. Returns a
    CipherTensor holding `num_real_cts` real-lane ciphertexts, in the
    same order and at the same level/scale as an unpacked upload would
    have produced, so it can be fed to a layer unchanged.
    """
    backend = scheme.backend
    real_ids = []

    for packed_id in packed.ids:
        conj = backend.ConjugateNew(packed_id)
        # x = p + conj(p)
        even = backend.AddCiphertextNew(packed_id, conj)
        # y = -i * (p - conj(p)); MulImaginaryUnit is in place, Negate
        # returns a fresh handle despite the name.
        diff = backend.SubCiphertextNew(packed_id, conj)
        backend.MulImaginaryUnit(diff)
        odd = backend.Negate(diff)

        backend.DeleteCiphertext(conj)
        backend.DeleteCiphertext(diff)
        real_ids.extend([even, odd])

    # An odd real-ciphertext count leaves one zero-padded partner; drop
    # it rather than leaking the handle.
    for surplus in real_ids[num_real_cts:]:
        backend.DeleteCiphertext(surplus)
    real_ids = real_ids[:num_real_cts]

    return CipherTensor(scheme, real_ids, shape, on_shape, is_complex=False)
