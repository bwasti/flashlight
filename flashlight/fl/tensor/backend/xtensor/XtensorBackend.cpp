/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/xtensor/XtensorBackend.h"

#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"

#define FL_XTENSOR_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(        \
      "XtensorBackend::" + std::string(__func__) + " - unimplemented.");

namespace fl {

XtensorBackend::XtensorBackend() {
  // Set up state
}

XtensorBackend& XtensorBackend::getInstance() {
  static XtensorBackend instance;
  return instance;
}

TensorBackendType XtensorBackend::backendType() const {
  // Implementers of a backend should create their own option in the
  // TensorBackendType enum and return it here.
  return TensorBackendType::Xtensor;
}

/* -------------------------- Compute Functions -------------------------- */

void XtensorBackend::eval(const Tensor& /* tensor */) {
  // Launch computation for a given tensor. Can be a noop for non-async
  // runtimes.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

bool XtensorBackend::supportsDataType(const fl::dtype& /* dtype */) const {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::getMemMgrInfo(
    const char* /* msg */,
    const int /* deviceId */,
    std::ostream* /* ostream */) {
  // Can be a noop if no memory manager is implemented.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::setMemMgrLogStream(std::ostream* /* stream */) {
  // Can be a noop if no memory manager is implemented.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::setMemMgrLoggingEnabled(const bool /* enabled */) {
  // Can be a noop if no memory manager is implemented.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::setMemMgrFlushInterval(const size_t /* interval */) {
  // Can be a noop if no memory manager is implemented.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/* -------------------------- Rand Functions -------------------------- */

void XtensorBackend::setSeed(const int /* seed */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::randn(const Shape& /* shape */, dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::rand(const Shape& /* shape */, dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                           \
  Tensor XtensorBackend::fromScalar(TYPE /* value */, const dtype /* type */) {   \
    throw std::invalid_argument(                                               \
        "XtensorBackend::fromScalar - not implemented for type " +                \
        std::string(#TYPE));                                                   \
  }                                                                            \
  Tensor XtensorBackend::full(                                                    \
      const Shape& /* shape */, TYPE /* value */, const dtype /* type */) {    \
    throw std::invalid_argument(                                               \
        "XtensorBackend::full - not implemented for type " + std::string(#TYPE)); \
  }
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
FL_XTENSOR_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);

Tensor XtensorBackend::identity(const Dim /* dim */, const dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::arange(
    const Shape& /* shape */,
    const Dim /* seqDim */,
    const dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::iota(
    const Shape& /* dims */,
    const Shape& /* tileDims */,
    const dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/************************ Shaping and Indexing *************************/
Tensor XtensorBackend::reshape(
    const Tensor& /* tensor */,
    const Shape& /* shape */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::transpose(
    const Tensor& /* tensor */,
    const Shape& /* axes */ /* = {} */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::tile(const Tensor& /* tensor */, const Shape& /* shape */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::concatenate(
    const std::vector<Tensor>& /* tensors */,
    const unsigned /* axis */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::nonzero(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::pad(
    const Tensor& /* input */,
    const std::vector<std::pair<int, int>>& /* padWidths */,
    const PadType /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/************************** Unary Operators ***************************/

Tensor XtensorBackend::exp(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::log(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::negative(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::logicalNot(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::log1p(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sin(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::cos(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sqrt(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::tanh(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::floor(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::ceil(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::rint(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::absolute(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sigmoid(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::erf(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::flip(const Tensor& /* tensor */, const unsigned /* dim */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::clip(
    const Tensor& /* tensor */,
    const Tensor& /* low */,
    const Tensor& /* high */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::roll(
    const Tensor& /* tensor */,
    const int /* shift */,
    const unsigned /* axis */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::isnan(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::isinf(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sign(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::tril(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::triu(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::where(
    const Tensor& /* condition */,
    const Tensor& /* x */,
    const Tensor& /* y */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::topk(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* k */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::sort(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::argsort(
    const Tensor& /* input */,
    const Dim /* axis */,
    const SortMode /* sortMode */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/************************** Binary Operators ***************************/
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                            \
  Tensor XtensorBackend::FUNC(const Tensor& /* a */, TYPE /* rhs */) {         \
    throw std::runtime_error(                                               \
        "XtensorBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                \
  }                                                                         \
  Tensor XtensorBackend::FUNC(TYPE /* lhs */, const Tensor& /* a */) {         \
    throw std::runtime_error(                                               \
        "XtensorBackend::" + std::string(#FUNC) + " unimplemented for type " + \
        std::string(#TYPE));                                                \
  }

#define FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP)                   \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const bool&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const int&);                \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned&);           \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const char&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned char&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long long&);          \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long long&); \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const double&);             \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const float&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const short&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                                          \
  Tensor XtensorBackend::FUNC(const Tensor& /* lhs */, const Tensor& /* rhs */) { \
    throw std::runtime_error(                                                  \
        "XtensorBackend::" + std::string(#FUNC) +                                 \
        " unimplemented for two-Tensor inputs.");                              \
  }                                                                            \
  FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP);

// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
FL_AF_BINARY_OP_DEF(+, add);
FL_AF_BINARY_OP_DEF(-, sub);
FL_AF_BINARY_OP_DEF(*, mul);
FL_AF_BINARY_OP_DEF(/, div);
FL_AF_BINARY_OP_DEF(==, eq);
FL_AF_BINARY_OP_DEF(!=, neq);
FL_AF_BINARY_OP_DEF(<, lessThan);
FL_AF_BINARY_OP_DEF(<=, lessThanEqual);
FL_AF_BINARY_OP_DEF(>, greaterThan);
FL_AF_BINARY_OP_DEF(>=, greaterThanEqual);
FL_AF_BINARY_OP_DEF(||, logicalOr);
FL_AF_BINARY_OP_DEF(&&, logicalAnd);
FL_AF_BINARY_OP_DEF(%, mod);
FL_AF_BINARY_OP_DEF(&, bitwiseAnd);
FL_AF_BINARY_OP_DEF(|, bitwiseOr);
FL_AF_BINARY_OP_DEF(^, bitwiseXor);
FL_AF_BINARY_OP_DEF(<<, lShift);
FL_AF_BINARY_OP_DEF(>>, rShift);
#undef FL_AF_BINARY_OP_DEF
#undef FL_AF_BINARY_OP_TYPE_DEF
#undef FL_AF_BINARY_OP_LITERALS_DEF

Tensor XtensorBackend::minimum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::maximum(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::power(const Tensor& /* lhs */, const Tensor& /* rhs */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/************************** BLAS ***************************/

Tensor XtensorBackend::matmul(
    const Tensor& /* lhs */,
    const Tensor& /* rhs */,
    MatrixProperty /* lhsProp */,
    MatrixProperty /* rhsProp */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/************************** Reductions ***************************/

Tensor XtensorBackend::amin(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::amax(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::min(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::max(
    Tensor& /* values */,
    Tensor& /* indices */,
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::sum(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::cumsum(
    const Tensor& /* input */,
    const unsigned /* axis */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::argmax(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::argmin(
    const Tensor& /* input */,
    const unsigned /* axis */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::mean(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::median(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::var(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* bias */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::std(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::norm(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    double /* p */ /* = 2 */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::countNonzero(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::any(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorBackend::all(
    const Tensor& /* input */,
    const std::vector<int>& /* axes */,
    const bool /* keepDims */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorBackend::print(const Tensor& /* tensor */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

} // namespace fl
