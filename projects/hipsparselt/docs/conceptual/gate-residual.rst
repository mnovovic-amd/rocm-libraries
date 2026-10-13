.. meta::
   :description: Gate residual epilogue in hipSPARSELt
   :keywords: hipSPARSELt, ROCm, gate residual, epilogue, SpMM

.. _gate-residual:

**********************************************************************
Gate residual epilogue
**********************************************************************

The gate residual epilogue applies an element-wise gate matrix :math:`G` to the result of the
sparse matrix multiplication, fused into the same kernel:

.. math::

   D_{ij} = G_{ij} \cdot X_{ij} + G_{ij}

where :math:`X` is the result of the matrix multiplication and the other epilogue operations,
before it is converted to the data type of ``D``.

Operation order
===============

When the gate residual is enabled, the epilogue is executed in the following order:

1. Sparse-dense matrix multiplication
2. Scaling by alpha/beta
3. Bias addition (if configured)
4. Activation function (if configured)
5. Gate residual: :math:`X \leftarrow G \cdot X + G`
6. Conversion to the data type of ``D``

All steps up to and including the gate residual are computed in ``FP32``. The gate values are
converted to ``FP32`` before they are applied.

Gate matrix requirements
========================

The gate matrix is described by a dense matrix descriptor with the following properties:

* The number of rows and columns must match matrix ``D``.
* The memory order must match matrix ``D``.
* The leading dimension is independent of ``D``.
* For batched operations, the gate matrix has its own batch stride. A batch stride of ``0``
  uses the same gate matrix for every batch.

Supported data types
====================

The gate residual is supported on gfx942 and gfx950 for the following data types:

.. list-table::
   :header-rows: 1

   * - Input A/B
     - Output D
     - Gate G
   * - ``FP16``
     - ``FP16``
     - ``FP16``
   * - ``BF16``
     - ``BF16``
     - ``BF16``
   * - ``FP8``
     - ``FP32``
     - ``FP32`` or ``FP8``
   * - ``BF8``
     - ``FP32``
     - ``FP32`` or ``BF8``

On gfx942, ``FP8`` and ``BF8`` refer to the ``E4M3_FNUZ`` and ``E5M2_FNUZ`` formats.
``INT8`` inputs are not supported.

Configuration
=============

Create a dense descriptor for the gate matrix, then set the gate descriptor and the gate pointer on the
matmul descriptor:

.. code-block:: cpp

    // Describe the gate matrix: same size and memory order as matrix D
    hipsparseLtMatDescriptor_t matGate;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtDenseDescriptorInit(
        &handle, &matGate, row_d, col_d, ldg, 16, HIP_R_16F, HIPSPARSE_ORDER_COL));

    // Attach the gate descriptor and the gate matrix to the matmul descriptor
    hipsparseLtMatDescriptor_t* gate_desc = &matGate;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        HIPSPARSELT_MATMUL_GATE_RESIDUAL_DESC,
        &gate_desc,
        sizeof(gate_desc)));

    void* gate_ptr = d_gate;
    CHECK_HIPSPARSELT_ERROR(hipsparseLtMatmulDescSetAttribute(
        &handle,
        &matmul,
        HIPSPARSELT_MATMUL_GATE_RESIDUAL_MAT_POINTER,
        &gate_ptr,
        sizeof(void*)));

For batched operations, set ``HIPSPARSELT_MAT_NUM_BATCHES`` and ``HIPSPARSELT_MAT_BATCH_STRIDE`` on the
gate descriptor in the same way as for the other matrices.

Setting the gate pointer to ``nullptr`` disables the gate residual, and the result is the same as without
a gate matrix.
