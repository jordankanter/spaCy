from cymem.cymem cimport Pool
from libc.stdint cimport int64_t, uint32_t
from libcpp.set cimport set
from libcpp.vector cimport vector
from murmurhash.mrmr cimport hash64
from preshed.maps cimport PreshMap

from .typedefs cimport attr_t, hash_t

ctypedef union Utf8Str:
    unsigned char[8] s
    unsigned char* p


cdef class StringStore:
    cdef Pool mem
    cdef vector[hash_t] _keys
    cdef PreshMap _map

    cdef const Utf8Str* intern_unicode(self, str py_string, bint allow_transient)
    cdef const Utf8Str* _intern_utf8(self, char* utf8_string, int length, hash_t* precalculated_hash, bint allow_transient)
    cdef vector[hash_t] _transient_keys
    cdef Pool _non_temp_mem
