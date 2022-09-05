/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/xtensor/XtensorTensor.h"

#include "flashlight/fl/tensor/backend/xtensor/XtensorBackend.h"

#define FL_XTENSOR_BACKEND_UNIMPLEMENTED \
  throw std::invalid_argument(           \
      "XtensorTensor::" + std::string(__func__) + " - unimplemented.");

namespace fl {

#define X(D, T, extra) \
  template <>          \
  const dtype dtypeFrom<T>::value = dtype::D;
MAP_TYPE(X, 0)
#undef X

XtensorTensor::XtensorTensor() {
  GENERIC_XARRAY(type_, {
    const auto& xshape = array.shape();
    std::vector<Dim> flshape(xshape.begin(), xshape.end());
    shape_ = Shape(flshape);
  })
}

XtensorTensor::XtensorTensor(
    const Shape& /* shape */,
    fl::dtype /* type */,
    const void* /* ptr */,
    Location /* memoryLocation */) {}

XtensorTensor::XtensorTensor(
    const Dim /* nRows */,
    const Dim /* nCols */,
    const Tensor& /* values */,
    const Tensor& /* rowIdx */,
    const Tensor& /* colIdx */,
    StorageType /* storageType */) {}

std::unique_ptr<TensorAdapterBase> XtensorTensor::clone() const {
  GENERIC_XARRAY(type_, {
    return std::unique_ptr<XtensorTensor>(new XtensorTensor(array));
  })
}

Tensor XtensorTensor::copy() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::shallowCopy() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

TensorBackendType XtensorTensor::backendType() const {
  return TensorBackendType::Xtensor;
}

TensorBackend& XtensorTensor::backend() const {
  return XtensorBackend::getInstance();
}

const Shape& XtensorTensor::shape() {
  return shape_;
}

fl::dtype XtensorTensor::type() const {
  return type_;
}

fl::dtype XtensorTensor::type() {
  return type_;
}

bool XtensorTensor::isSparse() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Location XtensorTensor::location() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorTensor::scalar(void* out) {
  GENERIC_XARRAY(type_, {
    memcpy(out, array.data(), sizeof(array_type));
  });
}

void XtensorTensor::device(void** /* out */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void XtensorTensor::host(void* out) {
  GENERIC_XARRAY(type_, {
    memcpy(out, array.data(), array.size() * sizeof(array_type));
  })
}

void XtensorTensor::unlock() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

bool XtensorTensor::isLocked() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

bool XtensorTensor::isContiguous() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Shape XtensorTensor::strides() {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

const Stream& XtensorTensor::stream() const {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::astype(const dtype /* type */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::index(const std::vector<Index>& /* indices */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::flatten() const {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::flat(const Index& /* idx */) const {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

Tensor XtensorTensor::asContiguousTensor() {
  return toTensor<XtensorTensor>(*this);
}

void XtensorTensor::setContext(void* /* context */) {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

void* XtensorTensor::getContext() {
  // Used to store arbitrary data on a Tensor - can be a noop.
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

std::string XtensorTensor::toString() {
  std::stringstream buffer;
  switch (type_) {
    case dtype::f32:
      buffer << xarray<float>().data()[0];
      break;
    default:
      break;
  }
  return buffer.str();
}

std::ostream& XtensorTensor::operator<<(std::ostream& /* ostr */) {
  FL_XTENSOR_BACKEND_UNIMPLEMENTED;
}

/******************** Assignment Operators ********************/
#define FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, TYPE)           \
  void XtensorTensor::OP(const TYPE& /* val */) {             \
    throw std::invalid_argument(                              \
        "XtensorTensor::" + std::string(#OP) + " for type " + \
        std::string(#TYPE));                                  \
  }

#define FL_XTENSOR_BACKEND_ASSIGN_OP(OP)                 \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, Tensor);         \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, double);         \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, float);          \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, int);            \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, unsigned);       \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, bool);           \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, char);           \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, unsigned char);  \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, short);          \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, unsigned short); \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, long);           \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, unsigned long);  \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, long long);      \
  FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE(OP, unsigned long long);

FL_XTENSOR_BACKEND_ASSIGN_OP(assign); // =
FL_XTENSOR_BACKEND_ASSIGN_OP(inPlaceAdd); // +=
FL_XTENSOR_BACKEND_ASSIGN_OP(inPlaceSubtract); // -=
FL_XTENSOR_BACKEND_ASSIGN_OP(inPlaceMultiply); // *=
FL_XTENSOR_BACKEND_ASSIGN_OP(inPlaceDivide); // /=
#undef FL_XTENSOR_BACKEND_ASSIGN_OP_TYPE
#undef FL_XTENSOR_BACKEND_ASSIGN_OP

} // namespace fl
