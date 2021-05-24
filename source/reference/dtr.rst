.. py:module:: megengine.dtr
.. currentmodule:: megengine.dtr

=============
megengine.dtr
=============

.. autosummary::
   :toctree: api
   :nosignatures:

   enable
   disable

.. py:function:: eviction_threshold(mod, value: Union[int, str])

   Get or set the eviction threshold in bytes. It can also be set to a string,
    whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.

    .. note::

       When GPU memory usage exceeds this value, DTR will heuristically select
       and evict resident tensors until the amount of used memory falls below
       this threshold.

    Examples:

    .. code-block::

        import megengine as mge
        mge.dtr.eviction_threshold = "2GB"

.. py:function:: evictee_minimum_size(mod)

   Get or set the memory threshold of tensors in bytes. It can also be set to a
    string, whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.

    .. note::

       Only tensors whose size exceeds this threshold will be added to the
       candidate set. A tensor that is not added to the candidate set will
       never be evicted during its lifetime.

    Examples:

    .. code-block::

        import megengine as mge
        mge.dtr.evictee_minimum_size = "2MB"


