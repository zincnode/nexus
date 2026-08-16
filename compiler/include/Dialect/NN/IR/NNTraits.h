//===-------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NN-specific semantic traits. These are documentation-only markers grouping
// ops into semantic categories (element-wise, convolution, pooling, ...).
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_NN_IR_NNTRAITS_H
#define DIALECT_NN_IR_NNTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace nxs {
namespace OpTrait {

// The op applies pointwise to each element of its input(s), e.g. add, mul,
// relu, silu.
template <typename ConcreteType>
class Elementwise
    : public ::mlir::OpTrait::TraitBase<ConcreteType, Elementwise> {};

// The op reduces its input along a dimension, e.g. softmax.
template <typename ConcreteType>
class Reduction
    : public ::mlir::OpTrait::TraitBase<ConcreteType, Reduction> {};

// The op computes a matrix product or affine transformation, e.g. matmul,
// linear.
template <typename ConcreteType>
class LinearAlgebra
    : public ::mlir::OpTrait::TraitBase<ConcreteType, LinearAlgebra> {};

// The op performs a convolution, e.g. conv2d.
template <typename ConcreteType>
class Convolution
    : public ::mlir::OpTrait::TraitBase<ConcreteType, Convolution> {};

// The op normalizes its input, e.g. batch_norm.
template <typename ConcreteType>
class Normalization
    : public ::mlir::OpTrait::TraitBase<ConcreteType, Normalization> {};

// The op downsamples its input over local windows, e.g. max_pool2d,
// adaptive_avg_pool2d.
template <typename ConcreteType>
class Pooling : public ::mlir::OpTrait::TraitBase<ConcreteType, Pooling> {};

// The op resamples its input to a different resolution, e.g. upsample.
template <typename ConcreteType>
class Resampling
    : public ::mlir::OpTrait::TraitBase<ConcreteType, Resampling> {};

// The op manipulates the shape of its input, e.g. flatten, reshape,
// transpose, concat, slice.
template <typename ConcreteType>
class ShapeManipulation
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ShapeManipulation> {};

} // namespace OpTrait
} // namespace nxs
} // namespace mlir

#endif // DIALECT_NN_IR_NNTRAITS_H
