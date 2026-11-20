// Copyright 2017 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_ATTRIBUTES_ATTRIBUTE_QUANTIZATION_TRANSFORM_H_
#define DRACO_ATTRIBUTES_ATTRIBUTE_QUANTIZATION_TRANSFORM_H_

#include <vector>

#include "draco/attributes/attribute_transform.h"
#include "draco/attributes/point_attribute.h"
#include "draco/core/encoder_buffer.h"

namespace draco {

// Attribute transform for quantized attributes. The transform supports both
// single precision (DT_FLOAT32) and double precision (DT_FLOAT64) attributes.
// The precision of the source attribute determines the precision of the
// quantization parameters (minimum values and range) that are stored in the
// bitstream: double precision attributes use double precision parameters so
// that the dequantized values satisfy the same error contract as the single
// precision path, i.e. the absolute error of each dequantized component is
// bounded by range / (2 * max_quantized_value).
class AttributeQuantizationTransform : public AttributeTransform {
 public:
  AttributeQuantizationTransform()
      : quantization_bits_(-1),
        data_type_(DT_FLOAT32),
        range_(0.f),
        range_d_(0.0) {}
  // Return attribute transform type.
  AttributeTransformType Type() const override {
    return ATTRIBUTE_QUANTIZATION_TRANSFORM;
  }
  // Try to init transform from attribute.
  bool InitFromAttribute(const PointAttribute& attribute) override;
  // Copy parameter values into the provided AttributeTransformData instance.
  void CopyToAttributeTransformData(
      AttributeTransformData* out_data) const override;

  bool TransformAttribute(const PointAttribute& attribute,
                          const std::vector<PointIndex>& point_ids,
                          PointAttribute* target_attribute) override;

  bool InverseTransformAttribute(const PointAttribute& attribute,
                                 PointAttribute* target_attribute) override;

  // Sets the quantization parameters for a single precision attribute.
  bool SetParameters(int quantization_bits, const float* min_values,
                     int num_components, float range);
  // Sets the quantization parameters for a double precision attribute.
  bool SetParameters(int quantization_bits, const double* min_values,
                     int num_components, double range);

  bool ComputeParameters(const PointAttribute& attribute,
                         const int quantization_bits);

  // Encode relevant parameters into buffer.
  bool EncodeParameters(EncoderBuffer* encoder_buffer) const override;

  bool DecodeParameters(const PointAttribute& attribute,
                        DecoderBuffer* decoder_buffer) override;

  int32_t quantization_bits() const { return quantization_bits_; }
  float min_value(int axis) const { return min_values_[axis]; }
  const std::vector<float>& min_values() const { return min_values_; }
  float range() const { return range_; }
  // Double precision views of the quantization parameters. These are the
  // authoritative values when data_type() == DT_FLOAT64.
  double min_value_d(int axis) const { return min_values_d_[axis]; }
  const std::vector<double>& min_values_d() const { return min_values_d_; }
  double range_d() const { return range_d_; }
  // Data type of the attribute this transform operates on. It determines the
  // precision of the quantization parameters stored in the bitstream. The
  // decoder must set this to the data type of the decoded (dequantized)
  // attribute before calling DecodeParameters().
  DataType data_type() const { return data_type_; }
  void set_data_type(DataType data_type) { data_type_ = data_type; }
  bool is_initialized() const { return quantization_bits_ != -1; }

 protected:
  // Create portable attribute using 1:1 mapping between points in the input and
  // output attribute.
  void GeneratePortableAttribute(const PointAttribute& attribute,
                                 int num_points,
                                 PointAttribute* target_attribute) const;

  // Create portable attribute using custom mapping between input and output
  // points.
  void GeneratePortableAttribute(const PointAttribute& attribute,
                                 const std::vector<PointIndex>& point_ids,
                                 int num_points,
                                 PointAttribute* target_attribute) const;

  DataType GetTransformedDataType(
      const PointAttribute& attribute) const override {
    return DT_UINT32;
  }
  int GetTransformedNumComponents(
      const PointAttribute& attribute) const override {
    return attribute.num_components();
  }

  static bool IsQuantizationValid(int quantization_bits);

 private:
  // Computes the quantization parameters from a double precision attribute.
  bool ComputeParametersFromDoubleAttribute(const PointAttribute& attribute,
                                            int quantization_bits);

  // Quantizes a single value of a double precision attribute into an integer
  // in the range [0, max_quantized_value]. The computation is done in a way
  // that avoids overflow/underflow for degenerate, very large, or very small
  // ranges.
  int32_t QuantizeDoubleValue(double value, double min_value,
                              double max_quantized_value) const;

  // Synchronizes the non-authoritative parameter representation with the
  // authoritative one (the one matching |data_type_|).
  void SyncDoubleParametersFromFloat();
  void SyncFloatParametersFromDouble();

  int32_t quantization_bits_;

  // Data type of the attribute the transform operates on (either DT_FLOAT32
  // or DT_FLOAT64).
  DataType data_type_;

  // Minimal dequantized value for each component of the attribute. This
  // single precision representation is authoritative for DT_FLOAT32
  // attributes and holds a truncated view of the double precision parameters
  // for DT_FLOAT64 attributes.
  std::vector<float> min_values_;

  // Bounds of the dequantized attribute (max delta over all components).
  // Authoritative for DT_FLOAT32 attributes (see |min_values_|).
  float range_;

  // Double precision quantization parameters. Authoritative for DT_FLOAT64
  // attributes and exact copies of the single precision parameters for
  // DT_FLOAT32 attributes.
  std::vector<double> min_values_d_;
  double range_d_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTE_DEQUANTIZATION_TRANSFORM_H_
