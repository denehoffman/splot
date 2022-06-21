# AUTO GENERATED ON 2022-03-01 AT 22:37:19
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-kernel-signatures.py
#
# (It is usually run as part of pip install . or localbuild.py.)

# fmt: off

from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_int8,
    c_uint8,
    c_int16,
    c_uint16,
    c_int32,
    c_uint32,
    c_int64,
    c_uint64,
    c_float,
    c_double,
    c_char_p,
)

import numpy as np

from numpy import (
    bool_,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
)

class ERROR(Structure):
    _fields_ = [
        ("str", c_char_p),
        ("filename", c_char_p),
        ("id", c_int64),
        ("attempt", c_int64),
        ("pass_through", c_bool),
    ]


def by_signature(lib):
    out = {}

    f = lib.awkward_BitMaskedArray_to_ByteMaskedArray
    f.argtypes = [POINTER(c_int8), POINTER(c_uint8), c_int64, c_bool, c_bool]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_BitMaskedArray_to_ByteMaskedArray', int8, uint8] = f

    f = lib.awkward_BitMaskedArray_to_IndexedOptionArray64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint8), c_int64, c_bool, c_bool]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_BitMaskedArray_to_IndexedOptionArray', int64, uint8] = f

    f = lib.awkward_ByteMaskedArray_getitem_carry_64
    f.argtypes = [POINTER(c_int8), POINTER(c_int8), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_ByteMaskedArray_getitem_carry', int8, int8, int64] = f

    out['awkward_ByteMaskedArray_getitem_nextcarry', int64, int8] = None

    out['awkward_ByteMaskedArray_getitem_nextcarry_outindex', int64, int64, int8] = None

    f = lib.awkward_ByteMaskedArray_mask8
    f.argtypes = [POINTER(c_int8), POINTER(c_int8), c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_ByteMaskedArray_mask', int8, int8] = f

    out['awkward_ByteMaskedArray_numnull', int64, int8] = None

    f = lib.awkward_ByteMaskedArray_overlay_mask8
    f.argtypes = [POINTER(c_int8), POINTER(c_int8), POINTER(c_int8), c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_ByteMaskedArray_overlay_mask', int8, int8, int8] = f

    out['awkward_ByteMaskedArray_reduce_next_64', int64, int64, int64, int8, int64] = None

    out['awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64', int64, int8] = None

    out['awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64', int64, int8, int64] = None

    f = lib.awkward_ByteMaskedArray_toIndexedOptionArray64
    f.argtypes = [POINTER(c_int64), POINTER(c_int8), c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_ByteMaskedArray_toIndexedOptionArray', int64, int8] = f

    out['awkward_Content_getitem_next_missing_jagged_getmaskstartstop', int64, int64, int64, int64, int64] = None

    f = lib.awkward_Identities32_to_Identities64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_Identities32_to_Identities64', int64, int32] = f

    out['awkward_Identities_extend', int32, int32] = None

    out['awkward_Identities_extend', int64, int64] = None

    out['awkward_Identities_from_IndexedArray', bool_, int32, int32, int32] = None

    out['awkward_Identities_from_IndexedArray', bool_, int32, int32, int64] = None

    out['awkward_Identities_from_IndexedArray', bool_, int32, int32, uint32] = None

    out['awkward_Identities_from_IndexedArray', bool_, int64, int64, int32] = None

    out['awkward_Identities_from_IndexedArray', bool_, int64, int64, int64] = None

    out['awkward_Identities_from_IndexedArray', bool_, int64, int64, uint32] = None

    out['awkward_Identities_from_ListArray', bool_, int32, int32, int32, int32] = None

    out['awkward_Identities_from_ListArray', bool_, int32, int32, int64, int64] = None

    out['awkward_Identities_from_ListArray', bool_, int32, int32, uint32, uint32] = None

    out['awkward_Identities_from_ListArray', bool_, int64, int64, int32, int32] = None

    out['awkward_Identities_from_ListArray', bool_, int64, int64, int64, int64] = None

    out['awkward_Identities_from_ListArray', bool_, int64, int64, uint32, uint32] = None

    out['awkward_Identities_from_ListOffsetArray', int32, int32, int32] = None

    out['awkward_Identities_from_ListOffsetArray', int32, int32, int64] = None

    out['awkward_Identities_from_ListOffsetArray', int32, int32, uint32] = None

    out['awkward_Identities_from_ListOffsetArray', int64, int64, int32] = None

    out['awkward_Identities_from_ListOffsetArray', int64, int64, int64] = None

    out['awkward_Identities_from_ListOffsetArray', int64, int64, uint32] = None

    out['awkward_Identities_from_RegularArray', int32, int32] = None

    out['awkward_Identities_from_RegularArray', int64, int64] = None

    out['awkward_Identities_from_UnionArray', bool_, int32, int32, int8, int32] = None

    out['awkward_Identities_from_UnionArray', bool_, int32, int32, int8, int64] = None

    out['awkward_Identities_from_UnionArray', bool_, int32, int32, int8, uint32] = None

    out['awkward_Identities_from_UnionArray', bool_, int64, int64, int8, int32] = None

    out['awkward_Identities_from_UnionArray', bool_, int64, int64, int8, int64] = None

    out['awkward_Identities_from_UnionArray', bool_, int64, int64, int8, uint32] = None

    f = lib.awkward_Identities32_getitem_carry_64
    f.argtypes = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_Identities_getitem_carry', int32, int32, int64] = f

    f = lib.awkward_Identities64_getitem_carry_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_Identities_getitem_carry', int64, int64, int64] = f

    out['awkward_Index_iscontiguous', bool_, int32] = None

    out['awkward_Index_iscontiguous', bool_, int64] = None

    out['awkward_Index_iscontiguous', bool_, int8] = None

    out['awkward_Index_iscontiguous', bool_, uint32] = None

    out['awkward_Index_iscontiguous', bool_, uint8] = None

    f = lib.awkward_Index32_to_Index64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_Index_to_Index64', int64, int32] = f

    f = lib.awkward_Index8_to_Index64
    f.argtypes = [POINTER(c_int64), POINTER(c_int8), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_Index_to_Index64', int64, int8] = f

    f = lib.awkward_IndexU32_to_Index64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_Index_to_Index64', int64, uint32] = f

    f = lib.awkward_IndexU8_to_Index64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint8), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_Index_to_Index64', int64, uint8] = f

    out['awkward_IndexedArray_fill', int64, int32] = None

    out['awkward_IndexedArray_fill', int64, int64] = None

    out['awkward_IndexedArray_fill', int64, uint32] = None

    f = lib.awkward_IndexedArray_fill_to64_count
    f.argtypes = [POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_IndexedArray_fill_count', int64] = f

    out['awkward_IndexedArray_flatten_nextcarry', int64, int32] = None

    out['awkward_IndexedArray_flatten_nextcarry', int64, int64] = None

    out['awkward_IndexedArray_flatten_nextcarry', int64, uint32] = None

    out['awkward_IndexedArray_flatten_none2empty', int64, int32, int64] = None

    out['awkward_IndexedArray_flatten_none2empty', int64, int64, int64] = None

    out['awkward_IndexedArray_flatten_none2empty', int64, uint32, int64] = None

    out['awkward_IndexedArray_getitem_adjust_outindex', int8, int64, int64, int64, int64] = None

    out['awkward_IndexedArray_getitem_carry', int32, int32, int64] = None

    out['awkward_IndexedArray_getitem_carry', int64, int64, int64] = None

    out['awkward_IndexedArray_getitem_carry', uint32, uint32, int64] = None

    out['awkward_IndexedArray_getitem_nextcarry', int64, int32] = None

    out['awkward_IndexedArray_getitem_nextcarry', int64, int64] = None

    out['awkward_IndexedArray_getitem_nextcarry', int64, uint32] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex', int64, int32, int32] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex', int64, int64, int64] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex', int64, uint32, uint32] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex_mask', int64, int64, int32] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex_mask', int64, int64, int64] = None

    out['awkward_IndexedArray_getitem_nextcarry_outindex_mask', int64, int64, uint32] = None

    out['awkward_IndexedArray_local_preparenext_64', int64, int64, int64, int64] = None

    f = lib.awkward_IndexedArray32_mask8
    f.argtypes = [POINTER(c_int8), POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_IndexedArray_mask', int8, int32] = f

    f = lib.awkward_IndexedArray64_mask8
    f.argtypes = [POINTER(c_int8), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_IndexedArray_mask', int8, int64] = f

    f = lib.awkward_IndexedArrayU32_mask8
    f.argtypes = [POINTER(c_int8), POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_IndexedArray_mask', int8, uint32] = f

    out['awkward_IndexedArray_numnull', int64, int32] = None

    out['awkward_IndexedArray_numnull', int64, int64] = None

    out['awkward_IndexedArray_numnull', int64, uint32] = None

    out['awkward_IndexedArray_numnull_parents', int64, int64, int32] = None

    out['awkward_IndexedArray_numnull_parents', int64, int64, int64] = None

    out['awkward_IndexedArray_numnull_parents', int64, int64, uint32] = None

    out['awkward_IndexedArray_numnull_unique_64', int64] = None

    out['awkward_IndexedArray_index_of_nulls', int64, int32, int64, int64] = None

    out['awkward_IndexedArray_index_of_nulls', int64, int64, int64, int64] = None

    out['awkward_IndexedArray_index_of_nulls', int64, uint32, int64, int64] = None

    f = lib.awkward_IndexedArray32_overlay_mask8_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int8), POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_IndexedArray_overlay_mask', int64, int8, int32] = f

    f = lib.awkward_IndexedArray64_overlay_mask8_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int8), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_IndexedArray_overlay_mask', int64, int8, int64] = f

    f = lib.awkward_IndexedArrayU32_overlay_mask8_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int8), POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_IndexedArray_overlay_mask', int64, int8, uint32] = f

    out['awkward_IndexedArray_reduce_next_64', int64, int64, int64, int32, int64] = None

    out['awkward_IndexedArray_reduce_next_64', int64, int64, int64, int64, int64] = None

    out['awkward_IndexedArray_reduce_next_64', int64, int64, int64, uint32, int64] = None

    f = lib.awkward_IndexedArray_reduce_next_fix_offsets_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_IndexedArray_reduce_next_fix_offsets_64', int64, int64] = f

    out['awkward_IndexedArray_unique_next_index_and_offsets_64', int64, int64, int64, int64] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64', int64, int32] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64', int64, int64] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64', int64, uint32] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64', int64, int32, int64] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64', int64, int64, int64] = None

    out['awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64', int64, uint32, int64] = None

    f = lib.awkward_IndexedArray32_simplify32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64, POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int32, int32] = f

    f = lib.awkward_IndexedArray32_simplify64_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int32, int64] = f

    f = lib.awkward_IndexedArray32_simplifyU32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64, POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int32, uint32] = f

    f = lib.awkward_IndexedArray64_simplify32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int64, int32] = f

    f = lib.awkward_IndexedArray64_simplify64_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int64, int64] = f

    f = lib.awkward_IndexedArray64_simplifyU32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, int64, uint32] = f

    f = lib.awkward_IndexedArrayU32_simplify32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64, POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, uint32, int32] = f

    f = lib.awkward_IndexedArrayU32_simplify64_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, uint32, int64] = f

    f = lib.awkward_IndexedArrayU32_simplifyU32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64, POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_IndexedArray_simplify', int64, uint32, uint32] = f

    f = lib.awkward_IndexedArray32_validity
    f.argtypes = [POINTER(c_int32), c_int64, c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_IndexedArray_validity', int32] = f

    f = lib.awkward_IndexedArray64_validity
    f.argtypes = [POINTER(c_int64), c_int64, c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_IndexedArray_validity', int64] = f

    f = lib.awkward_IndexedArrayU32_validity
    f.argtypes = [POINTER(c_uint32), c_int64, c_int64, c_bool]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_IndexedArray_validity', uint32] = f

    out['awkward_IndexedArray_ranges_next_64', int32, int64, int64, int64, int64, int64] = None

    out['awkward_IndexedArray_ranges_next_64', int64, int64, int64, int64, int64, int64] = None

    out['awkward_IndexedArray_ranges_next_64', uint32, int64, int64, int64, int64, int64] = None

    out['awkward_IndexedArray_ranges_carry_next_64', int32, int64, int64, int64] = None

    out['awkward_IndexedArray_ranges_carry_next_64', int64, int64, int64, int64] = None

    out['awkward_IndexedArray_ranges_carry_next_64', uint32, int64, int64, int64] = None

    out['awkward_IndexedOptionArray_rpad_and_clip_mask_axis1', int64, int8] = None

    out['awkward_ListArray_broadcast_tooffsets', int64, int64, int32, int32] = None

    out['awkward_ListArray_broadcast_tooffsets', int64, int64, int64, int64] = None

    out['awkward_ListArray_broadcast_tooffsets', int64, int64, uint32, uint32] = None

    out['awkward_ListArray_combinations', int64, int64, int64, int32, int32] = None

    out['awkward_ListArray_combinations', int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_combinations', int64, int64, int64, uint32, uint32] = None

    out['awkward_ListArray_combinations_length', int64, int64, int32, int32] = None

    out['awkward_ListArray_combinations_length', int64, int64, int64, int64] = None

    out['awkward_ListArray_combinations_length', int64, int64, uint32, uint32] = None

    out['awkward_ListArray_compact_offsets', int64, int32, int32] = None

    out['awkward_ListArray_compact_offsets', int64, int64, int64] = None

    out['awkward_ListArray_compact_offsets', int64, uint32, uint32] = None

    out['awkward_ListArray_fill', int64, int64, int32, int32] = None

    out['awkward_ListArray_fill', int64, int64, int64, int64] = None

    out['awkward_ListArray_fill', int64, int64, uint32, uint32] = None

    out['awkward_ListArray_getitem_carry', int32, int32, int32, int32, int64] = None

    out['awkward_ListArray_getitem_carry', int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_carry', uint32, uint32, uint32, uint32, int64] = None

    out['awkward_ListArray_getitem_jagged_apply', int64, int64, int64, int64, int64, int32, int32] = None

    out['awkward_ListArray_getitem_jagged_apply', int64, int64, int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_jagged_apply', int64, int64, int64, int64, int64, uint32, uint32] = None

    out['awkward_ListArray_getitem_jagged_carrylen', int64, int64, int64] = None

    out['awkward_ListArray_getitem_jagged_descend', int64, int64, int64, int32, int32] = None

    out['awkward_ListArray_getitem_jagged_descend', int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_jagged_descend', int64, int64, int64, uint32, uint32] = None

    f = lib.awkward_ListArray32_getitem_jagged_expand_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int32), POINTER(c_int32), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'out', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_jagged_expand', int64, int64, int64, int64, int32, int32] = f

    f = lib.awkward_ListArray64_getitem_jagged_expand_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'out', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_jagged_expand', int64, int64, int64, int64, int64, int64] = f

    f = lib.awkward_ListArrayU32_getitem_jagged_expand_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_uint32), POINTER(c_uint32), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'out', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_jagged_expand', int64, int64, int64, int64, uint32, uint32] = f

    out['awkward_ListArray_getitem_jagged_numvalid', int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_jagged_shrink', int64, int64, int64, int64, int64, int64] = None

    f = lib.awkward_ListArray32_getitem_next_array_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int32), POINTER(c_int32), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_next_array', int64, int64, int32, int32, int64] = f

    f = lib.awkward_ListArray64_getitem_next_array_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_next_array', int64, int64, int64, int64, int64] = f

    f = lib.awkward_ListArrayU32_getitem_next_array_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in', 'in']
    out['awkward_ListArray_getitem_next_array', int64, int64, uint32, uint32, int64] = f

    out['awkward_ListArray_getitem_next_array_advanced', int64, int64, int32, int32, int64, int64] = None

    out['awkward_ListArray_getitem_next_array_advanced', int64, int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_next_array_advanced', int64, int64, uint32, uint32, int64, int64] = None

    out['awkward_ListArray_getitem_next_at', int64, int32, int32] = None

    out['awkward_ListArray_getitem_next_at', int64, int64, int64] = None

    out['awkward_ListArray_getitem_next_at', int64, uint32, uint32] = None

    out['awkward_ListArray_getitem_next_range', int32, int64, int32, int32] = None

    out['awkward_ListArray_getitem_next_range', int64, int64, int64, int64] = None

    out['awkward_ListArray_getitem_next_range', uint32, int64, uint32, uint32] = None

    out['awkward_ListArray_getitem_next_range_carrylength', int64, int32, int32] = None

    out['awkward_ListArray_getitem_next_range_carrylength', int64, int64, int64] = None

    out['awkward_ListArray_getitem_next_range_carrylength', int64, uint32, uint32] = None

    out['awkward_ListArray_getitem_next_range_counts', int64, int32] = None

    out['awkward_ListArray_getitem_next_range_counts', int64, int64] = None

    out['awkward_ListArray_getitem_next_range_counts', int64, uint32] = None

    out['awkward_ListArray_getitem_next_range_spreadadvanced', int64, int64, int32] = None

    out['awkward_ListArray_getitem_next_range_spreadadvanced', int64, int64, int64] = None

    out['awkward_ListArray_getitem_next_range_spreadadvanced', int64, int64, uint32] = None

    out['awkward_ListArray_localindex', int64, int32] = None

    out['awkward_ListArray_localindex', int64, int64] = None

    out['awkward_ListArray_localindex', int64, uint32] = None

    out['awkward_ListArray_min_range', int64, int32, int32] = None

    out['awkward_ListArray_min_range', int64, int64, int64] = None

    out['awkward_ListArray_min_range', int64, uint32, uint32] = None

    f = lib.awkward_ListArray32_num_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_ListArray_num', int64, int32, int32] = f

    f = lib.awkward_ListArray64_num_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_ListArray_num', int64, int64, int64] = f

    f = lib.awkward_ListArrayU32_num_64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_ListArray_num', int64, uint32, uint32] = f

    out['awkward_ListArray_rpad_and_clip_length_axis1', int64, int32, int32] = None

    out['awkward_ListArray_rpad_and_clip_length_axis1', int64, int64, int64] = None

    out['awkward_ListArray_rpad_and_clip_length_axis1', int64, uint32, uint32] = None

    out['awkward_ListArray_rpad_axis1', int64, int32, int32, int32, int32] = None

    out['awkward_ListArray_rpad_axis1', int64, int64, int64, int64, int64] = None

    out['awkward_ListArray_rpad_axis1', int64, uint32, uint32, uint32, uint32] = None

    f = lib.awkward_ListArray32_validity
    f.argtypes = [POINTER(c_int32), POINTER(c_int32), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_ListArray_validity', int32, int32] = f

    f = lib.awkward_ListArray64_validity
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_ListArray_validity', int64, int64] = f

    f = lib.awkward_ListArrayU32_validity
    f.argtypes = [POINTER(c_uint32), POINTER(c_uint32), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in']
    out['awkward_ListArray_validity', uint32, uint32] = f

    out['awkward_ListOffsetArray_compact_offsets', int64, int32] = None

    out['awkward_ListOffsetArray_compact_offsets', int64, int64] = None

    out['awkward_ListOffsetArray_compact_offsets', int64, uint32] = None

    f = lib.awkward_ListOffsetArray32_flatten_offsets_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_ListOffsetArray_flatten_offsets', int64, int32, int64] = f

    f = lib.awkward_ListOffsetArray64_flatten_offsets_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_ListOffsetArray_flatten_offsets', int64, int64, int64] = f

    f = lib.awkward_ListOffsetArrayU32_flatten_offsets_64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_ListOffsetArray_flatten_offsets', int64, uint32, int64] = f

    out['awkward_ListOffsetArray_getitem_adjust_offsets', int64, int64, int64, int64] = None

    out['awkward_ListOffsetArray_getitem_adjust_offsets_index', int64, int64, int64, int64, int64, int8] = None

    out['awkward_ListOffsetArray_local_preparenext_64', int64, int64] = None

    f = lib.awkward_ListOffsetArray_reduce_global_startstop_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in']
    out['awkward_ListOffsetArray_reduce_global_startstop_64', int64, int64, int64] = f

    out['awkward_ListOffsetArray_reduce_local_nextparents_64', int64, int64] = None

    out['awkward_ListOffsetArray_reduce_local_outoffsets_64', int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_findgaps_64', int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64', int64, int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64', int64, int64, int64, int64, int64, int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64', int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64', int64, int64, int64, int64] = None

    out['awkward_ListOffsetArray_reduce_nonlocal_preparenext_64', int64, int64, int64, int64, int64, int64, int64] = None

    out['awkward_ListOffsetArray_rpad_and_clip_axis1', int64, int32] = None

    out['awkward_ListOffsetArray_rpad_and_clip_axis1', int64, int64] = None

    out['awkward_ListOffsetArray_rpad_and_clip_axis1', int64, uint32] = None

    out['awkward_ListOffsetArray_rpad_axis1', int64, int32] = None

    out['awkward_ListOffsetArray_rpad_axis1', int64, int64] = None

    out['awkward_ListOffsetArray_rpad_axis1', int64, uint32] = None

    out['awkward_ListOffsetArray_rpad_length_axis1', int32, int32, int64] = None

    out['awkward_ListOffsetArray_rpad_length_axis1', int64, int64, int64] = None

    out['awkward_ListOffsetArray_rpad_length_axis1', uint32, uint32, int64] = None

    out['awkward_ListOffsetArray_toRegularArray', int64, int32] = None

    out['awkward_ListOffsetArray_toRegularArray', int64, int64] = None

    out['awkward_ListOffsetArray_toRegularArray', int64, uint32] = None

    out['awkward_MaskedArray_getitem_next_jagged_project', int32, int64, int64, int64, int64] = None

    out['awkward_MaskedArray_getitem_next_jagged_project', int64, int64, int64, int64, int64] = None

    out['awkward_MaskedArray_getitem_next_jagged_project', uint32, int64, int64, int64, int64] = None

    out['awkward_NumpyArray_copy', uint8, uint8] = None

    out['awkward_NumpyArray_contiguous_copy', uint8, uint8, int64] = None

    out['awkward_NumpyArray_contiguous_copy_from_many', uint8, uint8, int64, int64] = None

    f = lib.awkward_NumpyArray_contiguous_init_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_NumpyArray_contiguous_init', int64] = f

    f = lib.awkward_NumpyArray_contiguous_next_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_contiguous_next', int64, int64] = f

    out['awkward_NumpyArray_fill', int8, int8] = None

    out['awkward_NumpyArray_fill', int8, int16] = None

    out['awkward_NumpyArray_fill', int8, int32] = None

    out['awkward_NumpyArray_fill', int8, int64] = None

    out['awkward_NumpyArray_fill', int8, uint8] = None

    out['awkward_NumpyArray_fill', int8, uint16] = None

    out['awkward_NumpyArray_fill', int8, uint32] = None

    out['awkward_NumpyArray_fill', int8, uint64] = None

    out['awkward_NumpyArray_fill', int8, float32] = None

    out['awkward_NumpyArray_fill', int8, float64] = None

    out['awkward_NumpyArray_fill', int16, int8] = None

    out['awkward_NumpyArray_fill', int16, int16] = None

    out['awkward_NumpyArray_fill', int16, int32] = None

    out['awkward_NumpyArray_fill', int16, int64] = None

    out['awkward_NumpyArray_fill', int16, uint8] = None

    out['awkward_NumpyArray_fill', int16, uint16] = None

    out['awkward_NumpyArray_fill', int16, uint32] = None

    out['awkward_NumpyArray_fill', int16, uint64] = None

    out['awkward_NumpyArray_fill', int16, float32] = None

    out['awkward_NumpyArray_fill', int16, float64] = None

    out['awkward_NumpyArray_fill', int32, int8] = None

    out['awkward_NumpyArray_fill', int32, int16] = None

    out['awkward_NumpyArray_fill', int32, int32] = None

    out['awkward_NumpyArray_fill', int32, int64] = None

    out['awkward_NumpyArray_fill', int32, uint8] = None

    out['awkward_NumpyArray_fill', int32, uint16] = None

    out['awkward_NumpyArray_fill', int32, uint32] = None

    out['awkward_NumpyArray_fill', int32, uint64] = None

    out['awkward_NumpyArray_fill', int32, float32] = None

    out['awkward_NumpyArray_fill', int32, float64] = None

    out['awkward_NumpyArray_fill', int64, int8] = None

    out['awkward_NumpyArray_fill', int64, int16] = None

    out['awkward_NumpyArray_fill', int64, int32] = None

    out['awkward_NumpyArray_fill', int64, int64] = None

    out['awkward_NumpyArray_fill', int64, uint8] = None

    out['awkward_NumpyArray_fill', int64, uint16] = None

    out['awkward_NumpyArray_fill', int64, uint32] = None

    out['awkward_NumpyArray_fill', int64, uint64] = None

    out['awkward_NumpyArray_fill', int64, float32] = None

    out['awkward_NumpyArray_fill', int64, float64] = None

    out['awkward_NumpyArray_fill', uint8, int8] = None

    out['awkward_NumpyArray_fill', uint8, int16] = None

    out['awkward_NumpyArray_fill', uint8, int32] = None

    out['awkward_NumpyArray_fill', uint8, int64] = None

    out['awkward_NumpyArray_fill', uint8, uint8] = None

    out['awkward_NumpyArray_fill', uint8, uint16] = None

    out['awkward_NumpyArray_fill', uint8, uint32] = None

    out['awkward_NumpyArray_fill', uint8, uint64] = None

    out['awkward_NumpyArray_fill', uint8, float32] = None

    out['awkward_NumpyArray_fill', uint8, float64] = None

    out['awkward_NumpyArray_fill', uint16, int8] = None

    out['awkward_NumpyArray_fill', uint16, int16] = None

    out['awkward_NumpyArray_fill', uint16, int32] = None

    out['awkward_NumpyArray_fill', uint16, int64] = None

    out['awkward_NumpyArray_fill', uint16, uint8] = None

    out['awkward_NumpyArray_fill', uint16, uint16] = None

    out['awkward_NumpyArray_fill', uint16, uint32] = None

    out['awkward_NumpyArray_fill', uint16, uint64] = None

    out['awkward_NumpyArray_fill', uint16, float32] = None

    out['awkward_NumpyArray_fill', uint16, float64] = None

    out['awkward_NumpyArray_fill', uint32, int8] = None

    out['awkward_NumpyArray_fill', uint32, int16] = None

    out['awkward_NumpyArray_fill', uint32, int32] = None

    out['awkward_NumpyArray_fill', uint32, int64] = None

    out['awkward_NumpyArray_fill', uint32, uint8] = None

    out['awkward_NumpyArray_fill', uint32, uint16] = None

    out['awkward_NumpyArray_fill', uint32, uint32] = None

    out['awkward_NumpyArray_fill', uint32, uint64] = None

    out['awkward_NumpyArray_fill', uint32, float32] = None

    out['awkward_NumpyArray_fill', uint32, float64] = None

    out['awkward_NumpyArray_fill', uint64, int8] = None

    out['awkward_NumpyArray_fill', uint64, int16] = None

    out['awkward_NumpyArray_fill', uint64, int32] = None

    out['awkward_NumpyArray_fill', uint64, int64] = None

    out['awkward_NumpyArray_fill', uint64, uint8] = None

    out['awkward_NumpyArray_fill', uint64, uint16] = None

    out['awkward_NumpyArray_fill', uint64, uint32] = None

    out['awkward_NumpyArray_fill', uint64, uint64] = None

    out['awkward_NumpyArray_fill', uint64, float32] = None

    out['awkward_NumpyArray_fill', uint64, float64] = None

    out['awkward_NumpyArray_fill', float32, int8] = None

    out['awkward_NumpyArray_fill', float32, int16] = None

    out['awkward_NumpyArray_fill', float32, int32] = None

    out['awkward_NumpyArray_fill', float32, int64] = None

    out['awkward_NumpyArray_fill', float32, uint8] = None

    out['awkward_NumpyArray_fill', float32, uint16] = None

    out['awkward_NumpyArray_fill', float32, uint32] = None

    out['awkward_NumpyArray_fill', float32, uint64] = None

    out['awkward_NumpyArray_fill', float32, float32] = None

    out['awkward_NumpyArray_fill', float32, float64] = None

    out['awkward_NumpyArray_fill', float64, int8] = None

    out['awkward_NumpyArray_fill', float64, int16] = None

    out['awkward_NumpyArray_fill', float64, int32] = None

    out['awkward_NumpyArray_fill', float64, int64] = None

    out['awkward_NumpyArray_fill', float64, uint8] = None

    out['awkward_NumpyArray_fill', float64, uint16] = None

    out['awkward_NumpyArray_fill', float64, uint32] = None

    out['awkward_NumpyArray_fill', float64, uint64] = None

    out['awkward_NumpyArray_fill', float64, float32] = None

    out['awkward_NumpyArray_fill', float64, float64] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, bool_] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, int8] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, int16] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, int32] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, int64] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, uint8] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, uint16] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, uint32] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, uint64] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, float32] = None

    out['awkward_NumpyArray_fill_tocomplex', float32, float64] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, bool_] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, int8] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, int16] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, int32] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, int64] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, uint8] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, uint16] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, uint32] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, uint64] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, float32] = None

    out['awkward_NumpyArray_fill_tocomplex', float64, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', bool_, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', bool_, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', int8, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', int8, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', int16, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', int16, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', int32, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', int32, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', int64, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', int64, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint8, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint8, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint16, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint16, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint32, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint32, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint64, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', uint64, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', float32, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', float32, float64] = None

    out['awkward_NumpyArray_fill_fromcomplex', float64, float32] = None

    out['awkward_NumpyArray_fill_fromcomplex', float64, float64] = None

    out['awkward_NumpyArray_fill_frombool', bool_, bool_] = None

    out['awkward_NumpyArray_fill_frombool', int8, bool_] = None

    out['awkward_NumpyArray_fill_frombool', int16, bool_] = None

    out['awkward_NumpyArray_fill_frombool', int32, bool_] = None

    out['awkward_NumpyArray_fill_frombool', int64, bool_] = None

    out['awkward_NumpyArray_fill_frombool', uint8, bool_] = None

    out['awkward_NumpyArray_fill_frombool', uint16, bool_] = None

    out['awkward_NumpyArray_fill_frombool', uint32, bool_] = None

    out['awkward_NumpyArray_fill_frombool', uint64, bool_] = None

    out['awkward_NumpyArray_fill_frombool', float32, bool_] = None

    out['awkward_NumpyArray_fill_frombool', float64, bool_] = None

    f = lib.awkward_NumpyArray_fill_tobool_fromint8
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_int8), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, int8] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromint16
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_int16), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, int16] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromint32
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, int32] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromint64
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, int64] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromuint8
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_uint8), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, uint8] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromuint16
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_uint16), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, uint16] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromuint32
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, uint32] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromuint64
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_uint64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, uint64] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromfloat32
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_float), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, float32] = f

    f = lib.awkward_NumpyArray_fill_tobool_fromfloat64
    f.argtypes = [POINTER(c_bool), c_int64, POINTER(c_double), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_fill_tobool', bool_, float64] = f

    out['awkward_NumpyArray_fill_scaled', int64, int64] = None

    out['awkward_NumpyArray_rearrange_shifted', int64, int64, int64, int64, int64] = None

    out['awkward_NumpyArray_getitem_boolean_nonzero', int64, int8] = None

    out['awkward_NumpyArray_getitem_boolean_numtrue', int64, int8] = None

    f = lib.awkward_NumpyArray_getitem_next_array_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_getitem_next_array', int64, int64, int64, int64] = f

    f = lib.awkward_NumpyArray_getitem_next_array_advanced_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_getitem_next_array_advanced', int64, int64, int64, int64] = f

    f = lib.awkward_NumpyArray_getitem_next_at_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_getitem_next_at', int64, int64] = f

    out['awkward_NumpyArray_getitem_next_null', uint8, uint8, int64] = None

    f = lib.awkward_NumpyArray_getitem_next_range_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_getitem_next_range', int64, int64] = f

    f = lib.awkward_NumpyArray_getitem_next_range_advanced_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_getitem_next_range_advanced', int64, int64, int64, int64] = f

    f = lib.awkward_NumpyArray_reduce_adjust_starts_64
    f.argtypes = [POINTER(c_int64), c_int64, POINTER(c_int64), POINTER(c_int64)]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_reduce_adjust_starts_64', int64, int64, int64] = f

    f = lib.awkward_NumpyArray_reduce_adjust_starts_shifts_64
    f.argtypes = [POINTER(c_int64), c_int64, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64)]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_NumpyArray_reduce_adjust_starts_shifts_64', int64, int64, int64, int64] = f

    f = lib.awkward_NumpyArray_reduce_mask_ByteMaskedArray_64
    f.argtypes = [POINTER(c_int8), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_NumpyArray_reduce_mask_ByteMaskedArray_64', int8, int64] = f

    out['awkward_ListOffsetArray_argsort_strings', int64, int64, uint8, int64, int64] = None

    out['awkward_NumpyArray_sort_asstrings_uint8', uint8, uint8, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', int8, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', int16, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', int32, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', int64, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', uint8, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', uint16, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', uint32, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', uint64, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', float32, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_ranges', float64, int64, int64, int64] = None

    out['awkward_NumpyArray_unique_strings', uint8, int64, int64, int64] = None

    out['awkward_NumpyArray_subrange_equal', bool_, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', int8, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', int16, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', int32, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', int64, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', uint8, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', uint16, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', uint32, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', uint64, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', float32, int64, int64, bool_] = None

    out['awkward_NumpyArray_subrange_equal', float64, int64, int64, bool_] = None

    f = lib.awkward_RegularArray_broadcast_tooffsets_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in']
    out['awkward_RegularArray_broadcast_tooffsets', int64] = f

    out['awkward_RegularArray_broadcast_tooffsets_size1', int64, int64] = None

    out['awkward_RegularArray_combinations_64', int64, int64, int64] = None

    f = lib.awkward_RegularArray_compact_offsets64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_RegularArray_compact_offsets', int64] = f

    f = lib.awkward_RegularArray_getitem_carry_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_carry', int64, int64] = f

    f = lib.awkward_RegularArray_getitem_jagged_expand_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_jagged_expand', int64, int64, int64] = f

    f = lib.awkward_RegularArray_getitem_next_array_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_next_array', int64, int64, int64] = f

    f = lib.awkward_RegularArray_getitem_next_array_advanced_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_next_array_advanced', int64, int64, int64, int64] = f

    out['awkward_RegularArray_getitem_next_array_regularize', int64, int64] = None

    f = lib.awkward_RegularArray_getitem_next_at_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_next_at', int64] = f

    f = lib.awkward_RegularArray_getitem_next_range_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64, c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_next_range', int64] = f

    f = lib.awkward_RegularArray_getitem_next_range_spreadadvanced_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_RegularArray_getitem_next_range_spreadadvanced', int64, int64] = f

    f = lib.awkward_RegularArray_localindex_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_RegularArray_localindex', int64] = f

    f = lib.awkward_RegularArray_num_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_RegularArray_num', int64] = f

    out['awkward_RegularArray_rpad_and_clip_axis1', int64] = None

    out['awkward_UnionArray_fillindex', int64, int32] = None

    out['awkward_UnionArray_fillindex', int64, int64] = None

    out['awkward_UnionArray_fillindex', int64, uint32] = None

    out['awkward_UnionArray_fillindex_count', int64] = None

    f = lib.awkward_UnionArray_fillna_from32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_UnionArray_fillna', int64, int32] = f

    f = lib.awkward_UnionArray_fillna_from64_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_UnionArray_fillna', int64, int64] = f

    f = lib.awkward_UnionArray_fillna_fromU32_to64
    f.argtypes = [POINTER(c_int64), POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in']
    out['awkward_UnionArray_fillna', int64, uint32] = f

    out['awkward_UnionArray_filltags', int8, int8] = None

    out['awkward_UnionArray_filltags_const', int8] = None

    out['awkward_UnionArray_flatten_combine', int8, int64, int64, int8, int32, int64] = None

    out['awkward_UnionArray_flatten_combine', int8, int64, int64, int8, int64, int64] = None

    out['awkward_UnionArray_flatten_combine', int8, int64, int64, int8, uint32, int64] = None

    out['awkward_UnionArray_flatten_length', int64, int8, int32, int64] = None

    out['awkward_UnionArray_flatten_length', int64, int8, int64, int64] = None

    out['awkward_UnionArray_flatten_length', int64, int8, uint32, int64] = None

    out['awkward_UnionArray_nestedfill_tags_index', int8, int32, int64, int64] = None

    out['awkward_UnionArray_nestedfill_tags_index', int8, int64, int64, int64] = None

    out['awkward_UnionArray_nestedfill_tags_index', int8, uint32, int64, int64] = None

    out['awkward_UnionArray_project', int64, int64, int8, int32] = None

    out['awkward_UnionArray_project', int64, int64, int8, int64] = None

    out['awkward_UnionArray_project', int64, int64, int8, uint32] = None

    out['awkward_UnionArray_regular_index', int32, int32, int8] = None

    out['awkward_UnionArray_regular_index', int64, int64, int8] = None

    out['awkward_UnionArray_regular_index', uint32, uint32, int8] = None

    out['awkward_UnionArray_regular_index_getsize', int64, int8] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int32, int8, int32] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int32, int8, int64] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int32, int8, uint32] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int64, int8, int32] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int64, int8, int64] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, int64, int8, uint32] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, uint32, int8, int32] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, uint32, int8, int64] = None

    out['awkward_UnionArray_simplify', int8, int64, int8, uint32, int8, uint32] = None

    out['awkward_UnionArray_simplify_one', int8, int64, int8, int32] = None

    out['awkward_UnionArray_simplify_one', int8, int64, int8, int64] = None

    out['awkward_UnionArray_simplify_one', int8, int64, int8, uint32] = None

    f = lib.awkward_UnionArray8_32_validity
    f.argtypes = [POINTER(c_int8), POINTER(c_int32), c_int64, c_int64, POINTER(c_int64)]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in', 'in']
    out['awkward_UnionArray_validity', int8, int32, int64] = f

    f = lib.awkward_UnionArray8_64_validity
    f.argtypes = [POINTER(c_int8), POINTER(c_int64), c_int64, c_int64, POINTER(c_int64)]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in', 'in']
    out['awkward_UnionArray_validity', int8, int64, int64] = f

    f = lib.awkward_UnionArray8_U32_validity
    f.argtypes = [POINTER(c_int8), POINTER(c_uint32), c_int64, c_int64, POINTER(c_int64)]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in', 'in', 'in']
    out['awkward_UnionArray_validity', int8, uint32, int64] = f

    out['awkward_argsort', int64, bool_, int64] = None

    out['awkward_argsort', int64, int8, int64] = None

    out['awkward_argsort', int64, int16, int64] = None

    out['awkward_argsort', int64, int32, int64] = None

    out['awkward_argsort', int64, int64, int64] = None

    out['awkward_argsort', int64, uint8, int64] = None

    out['awkward_argsort', int64, uint16, int64] = None

    out['awkward_argsort', int64, uint32, int64] = None

    out['awkward_argsort', int64, uint64, int64] = None

    out['awkward_argsort', int64, float32, int64] = None

    out['awkward_argsort', int64, float64, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, bool_, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, int8, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, int16, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, int32, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, int64, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, uint8, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, uint16, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, uint32, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, uint64, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, float32, int64] = None

    out['awkward_ListOffsetArray_argsort', int64, float64, int64] = None

    out['awkward_ListArray_argsort', int64, bool_, int64, int64] = None

    out['awkward_ListArray_argsort', int64, int8, int64, int64] = None

    out['awkward_ListArray_argsort', int64, int16, int64, int64] = None

    out['awkward_ListArray_argsort', int64, int32, int64, int64] = None

    out['awkward_ListArray_argsort', int64, int64, int64, int64] = None

    out['awkward_ListArray_argsort', int64, uint8, int64, int64] = None

    out['awkward_ListArray_argsort', int64, uint16, int64, int64] = None

    out['awkward_ListArray_argsort', int64, uint32, int64, int64] = None

    out['awkward_ListArray_argsort', int64, uint64, int64, int64] = None

    out['awkward_ListArray_argsort', int64, float32, int64, int64] = None

    out['awkward_ListArray_argsort', int64, float64, int64, int64] = None

    out['awkward_quick_argsort', int64, bool_, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, int8, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, int16, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, int32, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, int64, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, uint8, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, uint16, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, uint32, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, uint64, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, float32, int64, int64, int64] = None

    out['awkward_quick_argsort', int64, float64, int64, int64, int64] = None

    f = lib.awkward_carry_arange32
    f.argtypes = [POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_carry_arange', int32] = f

    f = lib.awkward_carry_arange64
    f.argtypes = [POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_carry_arange', int64] = f

    f = lib.awkward_carry_arangeU32
    f.argtypes = [POINTER(c_uint32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_carry_arange', uint32] = f

    f = lib.awkward_combinations_64
    f.argtypes = [POINTER(c_int64), c_int64, c_bool, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_combinations', int64] = f

    f = lib.awkward_content_reduce_zeroparents_64
    f.argtypes = [POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_content_reduce_zeroparents_64', int64] = f

    f = lib.awkward_Index32_carry_64
    f.argtypes = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_index_carry', int32, int32, int64] = f

    f = lib.awkward_Index64_carry_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_index_carry', int64, int64, int64] = f

    f = lib.awkward_Index8_carry_64
    f.argtypes = [POINTER(c_int8), POINTER(c_int8), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_index_carry', int8, int8, int64] = f

    f = lib.awkward_IndexU32_carry_64
    f.argtypes = [POINTER(c_uint32), POINTER(c_uint32), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_index_carry', uint32, uint32, int64] = f

    f = lib.awkward_IndexU8_carry_64
    f.argtypes = [POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_index_carry', uint8, uint8, int64] = f

    f = lib.awkward_Index32_carry_nocheck_64
    f.argtypes = [POINTER(c_int32), POINTER(c_int32), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_index_carry_nocheck', int32, int32, int64] = f

    f = lib.awkward_Index64_carry_nocheck_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_index_carry_nocheck', int64, int64, int64] = f

    f = lib.awkward_Index8_carry_nocheck_64
    f.argtypes = [POINTER(c_int8), POINTER(c_int8), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_index_carry_nocheck', int8, int8, int64] = f

    f = lib.awkward_IndexU32_carry_nocheck_64
    f.argtypes = [POINTER(c_uint32), POINTER(c_uint32), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_index_carry_nocheck', uint32, uint32, int64] = f

    f = lib.awkward_IndexU8_carry_nocheck_64
    f.argtypes = [POINTER(c_uint8), POINTER(c_uint8), POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in']
    out['awkward_index_carry_nocheck', uint8, uint8, int64] = f

    out['awkward_index_rpad_and_clip_axis0', int64] = None

    out['awkward_index_rpad_and_clip_axis1', int64, int64] = None

    out['awkward_Index_nones_as_index', int64] = None

    f = lib.awkward_localindex_64
    f.argtypes = [POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_localindex', int64] = f

    f = lib.awkward_missing_repeat_64
    f.argtypes = [POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in', 'in', 'in', 'in']
    out['awkward_missing_repeat', int64, int64] = f

    f = lib.awkward_new_Identities32
    f.argtypes = [POINTER(c_int32), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_new_Identities', int32] = f

    f = lib.awkward_new_Identities64
    f.argtypes = [POINTER(c_int64), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_new_Identities', int64] = f

    out['awkward_reduce_argmax', int64, int8, int64] = None

    out['awkward_reduce_argmax', int64, int16, int64] = None

    out['awkward_reduce_argmax', int64, int32, int64] = None

    out['awkward_reduce_argmax', int64, int64, int64] = None

    out['awkward_reduce_argmax', int64, uint8, int64] = None

    out['awkward_reduce_argmax', int64, uint16, int64] = None

    out['awkward_reduce_argmax', int64, uint32, int64] = None

    out['awkward_reduce_argmax', int64, uint64, int64] = None

    out['awkward_reduce_argmax', int64, float32, int64] = None

    out['awkward_reduce_argmax', int64, float64, int64] = None

    out['awkward_reduce_argmax_complex', int64, float32, int64] = None

    out['awkward_reduce_argmax_complex', int64, float64, int64] = None

    out['awkward_reduce_argmax_bool_64', int64, bool_, int64] = None

    out['awkward_reduce_argmin', int64, int8, int64] = None

    out['awkward_reduce_argmin', int64, int16, int64] = None

    out['awkward_reduce_argmin', int64, int32, int64] = None

    out['awkward_reduce_argmin', int64, int64, int64] = None

    out['awkward_reduce_argmin', int64, uint8, int64] = None

    out['awkward_reduce_argmin', int64, uint16, int64] = None

    out['awkward_reduce_argmin', int64, uint32, int64] = None

    out['awkward_reduce_argmin', int64, uint64, int64] = None

    out['awkward_reduce_argmin', int64, float32, int64] = None

    out['awkward_reduce_argmin', int64, float64, int64] = None

    out['awkward_reduce_argmin_bool_64', int64, bool_, int64] = None

    out['awkward_reduce_argmin_complex', int64, float32, int64] = None

    out['awkward_reduce_argmin_complex', int64, float64, int64] = None

    out['awkward_reduce_count_64', int64, int64] = None

    out['awkward_reduce_countnonzero', int64, bool_, int64] = None

    out['awkward_reduce_countnonzero', int64, int8, int64] = None

    out['awkward_reduce_countnonzero', int64, int16, int64] = None

    out['awkward_reduce_countnonzero', int64, int32, int64] = None

    out['awkward_reduce_countnonzero', int64, int64, int64] = None

    out['awkward_reduce_countnonzero', int64, uint8, int64] = None

    out['awkward_reduce_countnonzero', int64, uint16, int64] = None

    out['awkward_reduce_countnonzero', int64, uint32, int64] = None

    out['awkward_reduce_countnonzero', int64, uint64, int64] = None

    out['awkward_reduce_countnonzero', int64, float32, int64] = None

    out['awkward_reduce_countnonzero', int64, float64, int64] = None

    out['awkward_reduce_countnonzero_complex', int64, float32, int64] = None

    out['awkward_reduce_countnonzero_complex', int64, float64, int64] = None

    out['awkward_reduce_max', int8, int8, int64] = None

    out['awkward_reduce_max', int16, int16, int64] = None

    out['awkward_reduce_max', int32, int32, int64] = None

    out['awkward_reduce_max', int64, int64, int64] = None

    out['awkward_reduce_max', uint8, uint8, int64] = None

    out['awkward_reduce_max', uint16, uint16, int64] = None

    out['awkward_reduce_max', uint32, uint32, int64] = None

    out['awkward_reduce_max', uint64, uint64, int64] = None

    out['awkward_reduce_max', float32, float32, int64] = None

    out['awkward_reduce_max', float64, float64, int64] = None

    out['awkward_reduce_max_complex', float32, float32, int64] = None

    out['awkward_reduce_max_complex', float64, float64, int64] = None

    out['awkward_reduce_min', int8, int8, int64] = None

    out['awkward_reduce_min', int16, int16, int64] = None

    out['awkward_reduce_min', int32, int32, int64] = None

    out['awkward_reduce_min', int64, int64, int64] = None

    out['awkward_reduce_min', uint8, uint8, int64] = None

    out['awkward_reduce_min', uint16, uint16, int64] = None

    out['awkward_reduce_min', uint32, uint32, int64] = None

    out['awkward_reduce_min', uint64, uint64, int64] = None

    out['awkward_reduce_min', float32, float32, int64] = None

    out['awkward_reduce_min', float64, float64, int64] = None

    out['awkward_reduce_min_complex', float32, float32, int64] = None

    out['awkward_reduce_min_complex', float64, float64, int64] = None

    out['awkward_reduce_prod', int32, int8, int64] = None

    out['awkward_reduce_prod', int32, int16, int64] = None

    out['awkward_reduce_prod', int32, int32, int64] = None

    out['awkward_reduce_prod', int64, int8, int64] = None

    out['awkward_reduce_prod', int64, int16, int64] = None

    out['awkward_reduce_prod', int64, int32, int64] = None

    out['awkward_reduce_prod', int64, int64, int64] = None

    out['awkward_reduce_prod', uint32, uint8, int64] = None

    out['awkward_reduce_prod', uint32, uint16, int64] = None

    out['awkward_reduce_prod', uint32, uint32, int64] = None

    out['awkward_reduce_prod', uint64, uint8, int64] = None

    out['awkward_reduce_prod', uint64, uint16, int64] = None

    out['awkward_reduce_prod', uint64, uint32, int64] = None

    out['awkward_reduce_prod', uint64, uint64, int64] = None

    out['awkward_reduce_prod', float32, float32, int64] = None

    out['awkward_reduce_prod', float64, float64, int64] = None

    out['awkward_reduce_prod_complex', float32, float32, int64] = None

    out['awkward_reduce_prod_complex', float64, float64, int64] = None

    out['awkward_reduce_prod_bool', bool_, bool_, int64] = None

    out['awkward_reduce_prod_bool', bool_, int8, int64] = None

    out['awkward_reduce_prod_bool', bool_, int16, int64] = None

    out['awkward_reduce_prod_bool', bool_, int32, int64] = None

    out['awkward_reduce_prod_bool', bool_, int64, int64] = None

    out['awkward_reduce_prod_bool', bool_, uint8, int64] = None

    out['awkward_reduce_prod_bool', bool_, uint16, int64] = None

    out['awkward_reduce_prod_bool', bool_, uint32, int64] = None

    out['awkward_reduce_prod_bool', bool_, uint64, int64] = None

    out['awkward_reduce_prod_bool', bool_, float32, int64] = None

    out['awkward_reduce_prod_bool', bool_, float64, int64] = None

    out['awkward_reduce_prod_bool_complex', bool_, float32, int64] = None

    out['awkward_reduce_prod_bool_complex', bool_, float64, int64] = None

    out['awkward_reduce_prod_int32_bool_64', int32, bool_, int64] = None

    out['awkward_reduce_prod_int64_bool_64', int64, bool_, int64] = None

    out['awkward_reduce_sum', int32, int8, int64] = None

    out['awkward_reduce_sum', int32, int16, int64] = None

    out['awkward_reduce_sum', int32, int32, int64] = None

    out['awkward_reduce_sum', int64, int8, int64] = None

    out['awkward_reduce_sum', int64, int16, int64] = None

    out['awkward_reduce_sum', int64, int32, int64] = None

    out['awkward_reduce_sum', int64, int64, int64] = None

    out['awkward_reduce_sum', uint32, uint8, int64] = None

    out['awkward_reduce_sum', uint32, uint16, int64] = None

    out['awkward_reduce_sum', uint32, uint32, int64] = None

    out['awkward_reduce_sum', uint64, uint8, int64] = None

    out['awkward_reduce_sum', uint64, uint16, int64] = None

    out['awkward_reduce_sum', uint64, uint32, int64] = None

    out['awkward_reduce_sum', uint64, uint64, int64] = None

    out['awkward_reduce_sum', float32, float32, int64] = None

    out['awkward_reduce_sum', float64, float64, int64] = None

    out['awkward_reduce_sum_complex', float32, float32, int64] = None

    out['awkward_reduce_sum_complex', float64, float64, int64] = None

    out['awkward_reduce_sum_bool', bool_, bool_, int64] = None

    out['awkward_reduce_sum_bool', bool_, int8, int64] = None

    out['awkward_reduce_sum_bool', bool_, int16, int64] = None

    out['awkward_reduce_sum_bool', bool_, int32, int64] = None

    out['awkward_reduce_sum_bool', bool_, int64, int64] = None

    out['awkward_reduce_sum_bool', bool_, uint8, int64] = None

    out['awkward_reduce_sum_bool', bool_, uint16, int64] = None

    out['awkward_reduce_sum_bool', bool_, uint32, int64] = None

    out['awkward_reduce_sum_bool', bool_, uint64, int64] = None

    out['awkward_reduce_sum_bool', bool_, float32, int64] = None

    out['awkward_reduce_sum_bool', bool_, float64, int64] = None

    out['awkward_reduce_sum_bool_complex', bool_, float32, int64] = None

    out['awkward_reduce_sum_bool_complex', bool_, float64, int64] = None

    out['awkward_reduce_sum_int32_bool_64', int32, bool_, int64] = None

    out['awkward_reduce_sum_int64_bool_64', int64, bool_, int64] = None

    f = lib.awkward_regularize_arrayslice_64
    f.argtypes = [POINTER(c_int64), c_int64, c_int64]
    f.restype = ERROR
    f.dir = ['in', 'in', 'in']
    out['awkward_regularize_arrayslice', int64] = f

    out['awkward_slicearray_ravel', int64, int64, int64, int64] = None

    out['awkward_slicemissing_check_same', bool_, int8, int64] = None

    out['awkward_quick_sort', bool_, int64, int64, int64, int64] = None

    out['awkward_quick_sort', int8, int64, int64, int64, int64] = None

    out['awkward_quick_sort', int16, int64, int64, int64, int64] = None

    out['awkward_quick_sort', int32, int64, int64, int64, int64] = None

    out['awkward_quick_sort', int64, int64, int64, int64, int64] = None

    out['awkward_quick_sort', uint8, int64, int64, int64, int64] = None

    out['awkward_quick_sort', uint16, int64, int64, int64, int64] = None

    out['awkward_quick_sort', uint32, int64, int64, int64, int64] = None

    out['awkward_quick_sort', uint64, int64, int64, int64, int64] = None

    out['awkward_quick_sort', float32, int64, int64, int64, int64] = None

    out['awkward_quick_sort', float64, int64, int64, int64, int64] = None

    out['awkward_sort', bool_, bool_, int64] = None

    out['awkward_sort', int8, int8, int64] = None

    out['awkward_sort', int16, int16, int64] = None

    out['awkward_sort', int32, int32, int64] = None

    out['awkward_sort', int64, int64, int64] = None

    out['awkward_sort', uint8, uint8, int64] = None

    out['awkward_sort', uint16, uint16, int64] = None

    out['awkward_sort', uint32, uint32, int64] = None

    out['awkward_sort', uint64, uint64, int64] = None

    out['awkward_sort', float32, float32, int64] = None

    out['awkward_sort', float64, float64, int64] = None

    out['awkward_sort_ascending', bool_, bool_, int64] = None

    out['awkward_sort_ascending', int8, int8, int64] = None

    out['awkward_sort_ascending', int16, int16, int64] = None

    out['awkward_sort_ascending', int32, int32, int64] = None

    out['awkward_sort_ascending', int64, int64, int64] = None

    out['awkward_sort_ascending', uint8, uint8, int64] = None

    out['awkward_sort_ascending', uint16, uint16, int64] = None

    out['awkward_sort_ascending', uint32, uint32, int64] = None

    out['awkward_sort_ascending', uint64, uint64, int64] = None

    out['awkward_sort_ascending', float32, float32, int64] = None

    out['awkward_sort_ascending', float64, float64, int64] = None

    out['awkward_unique', bool_, int64] = None

    out['awkward_unique', int8, int64] = None

    out['awkward_unique', int16, int64] = None

    out['awkward_unique', int32, int64] = None

    out['awkward_unique', int64, int64] = None

    out['awkward_unique', uint8, int64] = None

    out['awkward_unique', uint16, int64] = None

    out['awkward_unique', uint32, int64] = None

    out['awkward_unique', uint64, int64] = None

    out['awkward_unique', float32, int64] = None

    out['awkward_unique', float64, int64] = None

    out['awkward_unique_copy', bool_, bool_, int64] = None

    out['awkward_unique_copy', int8, int8, int64] = None

    out['awkward_unique_copy', int16, int16, int64] = None

    out['awkward_unique_copy', int32, int32, int64] = None

    out['awkward_unique_copy', int64, int64, int64] = None

    out['awkward_unique_copy', uint8, uint8, int64] = None

    out['awkward_unique_copy', uint16, uint16, int64] = None

    out['awkward_unique_copy', uint32, uint32, int64] = None

    out['awkward_unique_copy', uint64, uint64, int64] = None

    out['awkward_unique_copy', float32, float32, int64] = None

    out['awkward_unique_copy', float64, float64, int64] = None

    out['awkward_unique_offsets', int8, int64, int64] = None

    out['awkward_unique_offsets', int16, int64, int64] = None

    out['awkward_unique_offsets', int32, int64, int64] = None

    out['awkward_unique_offsets', int64, int64, int64] = None

    out['awkward_unique_ranges', bool_, int64, int64] = None

    out['awkward_unique_ranges', int8, int64, int64] = None

    out['awkward_unique_ranges', int16, int64, int64] = None

    out['awkward_unique_ranges', int32, int64, int64] = None

    out['awkward_unique_ranges', int64, int64, int64] = None

    out['awkward_unique_ranges', uint8, int64, int64] = None

    out['awkward_unique_ranges', uint16, int64, int64] = None

    out['awkward_unique_ranges', uint32, int64, int64] = None

    out['awkward_unique_ranges', uint64, int64, int64] = None

    out['awkward_unique_ranges', float32, int64, int64] = None

    out['awkward_unique_ranges', float64, int64, int64] = None

    out['awkward_sorting_ranges', int64, int64] = None

    out['awkward_sorting_ranges_length', int64, int64] = None

    out['awkward_one_mask', int8] = None

    f = lib.awkward_zero_mask8
    f.argtypes = [POINTER(c_int8), c_int64]
    f.restype = ERROR
    f.dir = ['out', 'in']
    out['awkward_zero_mask', int8] = f

    return out
