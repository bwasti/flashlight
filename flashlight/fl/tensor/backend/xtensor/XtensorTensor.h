/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

#define MAP_TYPE(X, extra) \
  X(b8, bool, extra)       \
  X(s32, int32_t, extra)   \
  X(u32, uint32_t, extra)  \
  X(f32, float, extra)     \
  X(f64, double, extra)


#define FUNC_WRAP_XARRAY(D, T, func) \
  case dtype::D: {\
    using array_type = T; \
    const auto& array = xarray<array_type>(); \
    func\
    break; \
  }
#define GENERIC_XARRAY(type, func) do { \
  switch(type) { \
    MAP_TYPE(FUNC_WRAP_XARRAY, func) \
    default: \
      break; \
  } \
} while (0);

#define FUNC_WRAP_TYPE(D, T, func) \
  case dtype::D: {\
    using array_type = T; \
    func\
    break; \
  }
#define GENERIC_TYPE(type, func) do { \
  switch(type) { \
    MAP_TYPE(FUNC_WRAP_TYPE, func) \
    default: \
      break; \
  } \
} while (0);

template <typename T>
struct dtypeFrom {
  static const dtype value;
};

namespace detail {

struct ErasedXarray {
  virtual ~ErasedXarray() = default;
};

template <typename T>
struct TypedXarray : public ErasedXarray {
  xt::xshared_expression<xt::xarray<T>> array_;
  TypedXarray(xt::xarray<T>&& array)
      : array_(xt::make_xshared(std::move(array))) {}
  ~TypedXarray() {}
};

} // namespace detail

/**
 * A stub Tensor implementation to make it easy to get started with the
 * Flashlight Tensor API.
 *
 * This stub can be copied, renamed, and implemented as needed.
 */
class XtensorTensor : public TensorAdapterBase {
  // Would also recommend making this a variant/union type with an xarray
  // since that's the thing xtensor uses to think about intermediate JIT
  // expressions. The semantics are very similar to ArrayFire, so there's also
  // an xt::eval(...) which takes an xarray and does the same sort of
  // computation launching. eval() is on the XtensorBackend.
  //
  // Since Xtensor doesn't have a notion of stream (afaik), you can basically
  // return an empty trivial stream.

 public:
  std::shared_ptr<detail::ErasedXarray> array_;
  fl::dtype type_;
  fl::Shape shape_;
  template <typename T>
  XtensorTensor(xt::xarray<T>&& array)
      : array_(std::make_shared<detail::TypedXarray<T>>(std::move(array))) {
    type_ = dtypeFrom<T>::value;
  }
  template <typename F, typename R, typename S>
  XtensorTensor(xt::xgenerator<F, R, S>&& array)
      : array_(std::make_shared<detail::TypedXarray<R>>(std::move(array))) {
    type_ = dtypeFrom<R>::value;
  }
  template <typename T>
  XtensorTensor(const xt::xshared_expression<xt::xarray<T>>& array)
      : array_(std::make_shared<detail::TypedXarray<T>>(array)) {
    type_ = dtypeFrom<T>::value;
  }
  template <typename F, typename ...CT>
  XtensorTensor(xt::xfunction<F, CT...>&& array)
      : array_(std::make_shared<detail::TypedXarray<typename xt::xfunction<F, CT...>::value_type>>(std::move(array))) {
    type_ = dtypeFrom<typename xt::xfunction<F, CT...>::value_type>::value;
  }
  template <typename F, typename CT, typename X, typename O>
  XtensorTensor(xt::xreducer<F, CT, X, O>&& array)
      : array_(std::make_shared<detail::TypedXarray<typename xt::xreducer<F, CT, X, O>::value_type>>(std::move(array))) {
    type_ = dtypeFrom<typename xt::xreducer<F, CT, X, O>::value_type>::value;
  }

  XtensorTensor();

  template <typename T>
  const xt::xshared_expression<xt::xarray<T>>& xarray() const {
    return std::dynamic_pointer_cast<detail::TypedXarray<T>>(array_)->array_;
  }

#define X(D, T, OP) \
  case dtype::D:    \
    return XtensorTensor(xarray<T>() OP rhs.xarray<T>());

#define F(OP)                                            \
  XtensorTensor operator OP(const XtensorTensor& rhs) {  \
    switch (type_) {                                     \
      MAP_TYPE(X, OP)                                    \
      default:                                           \
        throw std::runtime_error(                        \
            "XtensorTensor doesn't support this type:" + \
            dtypeToString(type_));                       \
    }                                                    \
  }

  F(+)
  F(-)
  F(*)
  F(/)
  F(<)
  F(<=)
  F(>)
  F(>=)
// F(==)
// F(!=)
// F(||)
// F(&&)
// F(%)
// F(&)
// F(|)
// F(^)
// F(<<)
// F(>>)
#undef X
#undef F

  /**
   * Construct a XtensorTensor using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  XtensorTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);

  // Constructor for a sparse XtensorTensor. Can throw if unimplemented.
  XtensorTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  ~XtensorTensor() override = default;
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  TensorBackend& backend() const override;
  Tensor copy() override;
  Tensor shallowCopy() override;
  const Shape& shape() override;
  dtype type() const;
  dtype type() override;
  bool isSparse() override;
  Location location() override;
  void scalar(void* out) override;
  void device(void** out) override;
  void host(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() override;
  Shape strides() override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override;
  void* getContext() override;
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) void OP(const TYPE& val) override;

#define ASSIGN_OP(OP)                 \
  ASSIGN_OP_TYPE(OP, Tensor);         \
  ASSIGN_OP_TYPE(OP, double);         \
  ASSIGN_OP_TYPE(OP, float);          \
  ASSIGN_OP_TYPE(OP, int);            \
  ASSIGN_OP_TYPE(OP, unsigned);       \
  ASSIGN_OP_TYPE(OP, bool);           \
  ASSIGN_OP_TYPE(OP, char);           \
  ASSIGN_OP_TYPE(OP, unsigned char);  \
  ASSIGN_OP_TYPE(OP, short);          \
  ASSIGN_OP_TYPE(OP, unsigned short); \
  ASSIGN_OP_TYPE(OP, long);           \
  ASSIGN_OP_TYPE(OP, unsigned long);  \
  ASSIGN_OP_TYPE(OP, long long);      \
  ASSIGN_OP_TYPE(OP, unsigned long long);

  ASSIGN_OP(assign); // =
  ASSIGN_OP(inPlaceAdd); // +=
  ASSIGN_OP(inPlaceSubtract); // -=
  ASSIGN_OP(inPlaceMultiply); // *=
  ASSIGN_OP(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

} // namespace fl
