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
#include "draco/attributes/attribute_quantization_transform.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "draco/attributes/attribute_transform_type.h"
#include "draco/core/quantization_utils.h"

namespace draco {

bool AttributeQuantizationTransform::InitFromAttribute(
    const PointAttribute& attribute) {
  const AttributeTransformData* const transform_data =
      attribute.GetAttributeTransformData();
  if (!transform_data ||
      transform_data->transform_type() != ATTRIBUTE_QUANTIZATION_TRANSFORM) {
    return false;  // Wrong transform type.
  }
  int32_t byte_offset = 0;
  quantization_bits_ = transform_data->GetParameterValue<int32_t>(byte_offset);
  byte_offset += 4;
  data_type_ =
      static_cast<DataType>(transform_data->GetParameterValue<int32_t>(
          byte_offset));
  byte_offset += 4;
  const int num_components = attribute.num_components();
  if (data_type_ == DT_FLOAT64) {
    min_values_d_.resize(num_components);
    for (int i = 0; i < num_components; ++i) {
      min_values_d_[i] = transform_data->GetParameterValue<double>(byte_offset);
      byte_offset += 8;
    }
    range_d_ = transform_data->GetParameterValue<double>(byte_offset);
    SyncFloatParametersFromDouble();
  } else {
    data_type_ = DT_FLOAT32;
    min_values_.resize(num_components);
    for (int i = 0; i < num_components; ++i) {
      min_values_[i] = transform_data->GetParameterValue<float>(byte_offset);
      byte_offset += 4;
    }
    range_ = transform_data->GetParameterValue<float>(byte_offset);
    SyncDoubleParametersFromFloat();
  }
  return true;
}

// Copy parameter values into the provided AttributeTransformData instance.
void AttributeQuantizationTransform::CopyToAttributeTransformData(
    AttributeTransformData* out_data) const {
  out_data->set_transform_type(ATTRIBUTE_QUANTIZATION_TRANSFORM);
  out_data->AppendParameterValue(quantization_bits_);
  // Store the data type so that the reader of the transform data knows the
  // precision of the remaining parameters (the attribute holding the
  // transform data is usually the portable, quantized, attribute whose data
  // type does not carry this information).
  out_data->AppendParameterValue(static_cast<int32_t>(data_type_));
  if (data_type_ == DT_FLOAT64) {
    for (int i = 0; i < min_values_d_.size(); ++i) {
      out_data->AppendParameterValue(min_values_d_[i]);
    }
    out_data->AppendParameterValue(range_d_);
  } else {
    for (int i = 0; i < min_values_.size(); ++i) {
      out_data->AppendParameterValue(min_values_[i]);
    }
    out_data->AppendParameterValue(range_);
  }
}

bool AttributeQuantizationTransform::TransformAttribute(
    const PointAttribute& attribute, const std::vector<PointIndex>& point_ids,
    PointAttribute* target_attribute) {
  if (point_ids.empty()) {
    GeneratePortableAttribute(attribute, target_attribute->size(),
                              target_attribute);
  } else {
    GeneratePortableAttribute(attribute, point_ids, target_attribute->size(),
                              target_attribute);
  }
  return true;
}

bool AttributeQuantizationTransform::InverseTransformAttribute(
    const PointAttribute& attribute, PointAttribute* target_attribute) {
  if (target_attribute->data_type() == DT_FLOAT64) {
    // Convert all quantized values back to doubles.
    const int32_t max_quantized_value =
        (1u << static_cast<uint32_t>(quantization_bits_)) - 1;
    if (max_quantized_value <= 0) {
      return false;
    }
    const int num_components = target_attribute->num_components();
    const int entry_size = sizeof(double) * num_components;
    const std::unique_ptr<double[]> att_val(new double[num_components]);
    const double max_q = static_cast<double>(max_quantized_value);
    int quant_val_id = 0;
    int out_byte_pos = 0;
    const int32_t* const source_attribute_data =
        reinterpret_cast<const int32_t*>(
            attribute.GetAddress(AttributeValueIndex(0)));

    const int num_values = target_attribute->size();

    for (uint32_t i = 0; i < num_values; ++i) {
      for (int c = 0; c < num_components; ++c) {
        const double quantized_value =
            static_cast<double>(source_attribute_data[quant_val_id++]);
        // Compute the dequantized value as min + (q / max_q) * range. This
        // formulation avoids overflow for very large ranges and underflow to
        // zero for very small ranges (unlike a precomputed delta =
        // range / max_q multiplier).
        att_val[c] = min_values_d_[c] +
                     (quantized_value / max_q) * range_d_;
      }
      // Store the floating point value into the attribute buffer.
      target_attribute->buffer()->Write(out_byte_pos, att_val.get(),
                                        entry_size);
      out_byte_pos += entry_size;
    }
    return true;
  }
  if (target_attribute->data_type() != DT_FLOAT32) {
    return false;
  }

  // Convert all quantized values back to floats.
  const int32_t max_quantized_value =
      (1u << static_cast<uint32_t>(quantization_bits_)) - 1;
  const int num_components = target_attribute->num_components();
  const int entry_size = sizeof(float) * num_components;
  const std::unique_ptr<float[]> att_val(new float[num_components]);
  int quant_val_id = 0;
  int out_byte_pos = 0;
  Dequantizer dequantizer;
  if (!dequantizer.Init(range_, max_quantized_value)) {
    return false;
  }
  const int32_t* const source_attribute_data = reinterpret_cast<const int32_t*>(
      attribute.GetAddress(AttributeValueIndex(0)));

  const int num_values = target_attribute->size();

  for (uint32_t i = 0; i < num_values; ++i) {
    for (int c = 0; c < num_components; ++c) {
      float value =
          dequantizer.DequantizeFloat(source_attribute_data[quant_val_id++]);
      value = value + min_values_[c];
      att_val[c] = value;
    }
    // Store the floating point value into the attribute buffer.
    target_attribute->buffer()->Write(out_byte_pos, att_val.get(), entry_size);
    out_byte_pos += entry_size;
  }
  return true;
}

bool AttributeQuantizationTransform::IsQuantizationValid(
    int quantization_bits) {
  // Currently we allow only up to 30 bit quantization.
  return quantization_bits >= 1 && quantization_bits <= 30;
}

bool AttributeQuantizationTransform::SetParameters(int quantization_bits,
                                                   const float* min_values,
                                                   int num_components,
                                                   float range) {
  if (!IsQuantizationValid(quantization_bits)) {
    return false;
  }
  quantization_bits_ = quantization_bits;
  data_type_ = DT_FLOAT32;
  min_values_.assign(min_values, min_values + num_components);
  range_ = range;
  SyncDoubleParametersFromFloat();
  return true;
}

bool AttributeQuantizationTransform::SetParameters(int quantization_bits,
                                                   const double* min_values,
                                                   int num_components,
                                                   double range) {
  if (!IsQuantizationValid(quantization_bits)) {
    return false;
  }
  for (int i = 0; i < num_components; ++i) {
    if (std::isnan(min_values[i]) || std::isinf(min_values[i])) {
      return false;
    }
  }
  if (std::isnan(range) || std::isinf(range) || range < 0.0) {
    return false;
  }
  quantization_bits_ = quantization_bits;
  data_type_ = DT_FLOAT64;
  min_values_d_.assign(min_values, min_values + num_components);
  range_d_ = range;
  // In case the range is degenerate, initialize it to unit length. This will
  // ensure that all values are quantized properly to the same value.
  if (range_d_ == 0.0) {
    range_d_ = 1.0;
  }
  SyncFloatParametersFromDouble();
  return true;
}

bool AttributeQuantizationTransform::ComputeParameters(
    const PointAttribute& attribute, const int quantization_bits) {
  if (quantization_bits_ != -1) {
    return false;  // already initialized.
  }
  if (!IsQuantizationValid(quantization_bits)) {
    return false;
  }
  if (attribute.size() == 0) {
    return false;  // No values to compute the parameters from.
  }
  if (attribute.data_type() == DT_FLOAT64) {
    return ComputeParametersFromDoubleAttribute(attribute, quantization_bits);
  }
  if (attribute.data_type() != DT_FLOAT32) {
    return false;
  }
  quantization_bits_ = quantization_bits;
  data_type_ = DT_FLOAT32;

  const int num_components = attribute.num_components();
  range_ = 0.f;
  min_values_ = std::vector<float>(num_components, 0.f);
  const std::unique_ptr<float[]> max_values(new float[num_components]);
  const std::unique_ptr<float[]> att_val(new float[num_components]);
  // Compute minimum values and max value difference.
  attribute.GetValue(AttributeValueIndex(0), att_val.get());
  attribute.GetValue(AttributeValueIndex(0), min_values_.data());
  attribute.GetValue(AttributeValueIndex(0), max_values.get());

  for (AttributeValueIndex i(1); i < static_cast<uint32_t>(attribute.size());
       ++i) {
    attribute.GetValue(i, att_val.get());
    for (int c = 0; c < num_components; ++c) {
      if (std::isnan(att_val[c])) {
        return false;
      }
      if (min_values_[c] > att_val[c]) {
        min_values_[c] = att_val[c];
      }
      if (max_values[c] < att_val[c]) {
        max_values[c] = att_val[c];
      }
    }
  }
  for (int c = 0; c < num_components; ++c) {
    if (std::isnan(min_values_[c]) || std::isinf(min_values_[c]) ||
        std::isnan(max_values[c]) || std::isinf(max_values[c])) {
      return false;
    }
    const float dif = max_values[c] - min_values_[c];
    if (dif > range_) {
      range_ = dif;
    }
  }

  // Both ends are finite, but their difference can overflow, for example for
  // values from -3.4e38 to 3.4e38. The range is stored as a float, so such
  // values cannot be quantized.
  if (std::isinf(range_)) {
    return false;
  }

  // In case all values are the same, initialize the range to unit length. This
  // will ensure that all values are quantized properly to the same value.
  if (range_ == 0.f) {
    range_ = 1.f;
  }

  SyncDoubleParametersFromFloat();
  return true;
}

bool AttributeQuantizationTransform::ComputeParametersFromDoubleAttribute(
    const PointAttribute& attribute, const int quantization_bits) {
  quantization_bits_ = quantization_bits;
  data_type_ = DT_FLOAT64;

  const int num_components = attribute.num_components();
  range_d_ = 0.0;
  min_values_d_ = std::vector<double>(num_components, 0.0);
  std::vector<double> max_values(num_components, 0.0);
  std::vector<double> att_val(num_components, 0.0);
  // Compute minimum values and max value difference in double precision. Note
  // that computing the parameters in single precision could overflow or lose
  // all significant bits for double precision input (e.g. for coordinates
  // with a large offset and a small spread).
  attribute.GetValue(AttributeValueIndex(0), att_val.data());
  for (int c = 0; c < num_components; ++c) {
    min_values_d_[c] = att_val[c];
    max_values[c] = att_val[c];
  }

  for (AttributeValueIndex i(1); i < static_cast<uint32_t>(attribute.size());
       ++i) {
    attribute.GetValue(i, att_val.data());
    for (int c = 0; c < num_components; ++c) {
      if (std::isnan(att_val[c])) {
        return false;
      }
      if (min_values_d_[c] > att_val[c]) {
        min_values_d_[c] = att_val[c];
      }
      if (max_values[c] < att_val[c]) {
        max_values[c] = att_val[c];
      }
    }
  }
  for (int c = 0; c < num_components; ++c) {
    if (std::isnan(min_values_d_[c]) || std::isinf(min_values_d_[c]) ||
        std::isnan(max_values[c]) || std::isinf(max_values[c])) {
      return false;
    }
    const double dif = max_values[c] - min_values_d_[c];
    if (dif > range_d_) {
      range_d_ = dif;
    }
  }

  // Both ends are finite, but their difference can overflow, for example for
  // values from -1.7e308 to 1.7e308. The range is stored as a double, so such
  // values cannot be quantized.
  if (std::isinf(range_d_)) {
    return false;
  }

  // In case all values are the same, initialize the range to unit length. This
  // will ensure that all values are quantized properly to the same value.
  if (range_d_ == 0.0) {
    range_d_ = 1.0;
  }

  SyncFloatParametersFromDouble();
  return true;
}

bool AttributeQuantizationTransform::EncodeParameters(
    EncoderBuffer* encoder_buffer) const {
  if (is_initialized()) {
    if (data_type_ == DT_FLOAT64) {
      // Double precision attributes store their quantization parameters in
      // double precision. The decoder derives the parameter precision from
      // the data type of the decoded attribute, so both sides of the
      // bitstream always agree on the format.
      encoder_buffer->Encode(min_values_d_.data(),
                             sizeof(double) * min_values_d_.size());
      encoder_buffer->Encode(range_d_);
    } else {
      encoder_buffer->Encode(min_values_.data(),
                             sizeof(float) * min_values_.size());
      encoder_buffer->Encode(range_);
    }
    encoder_buffer->Encode(static_cast<uint8_t>(quantization_bits_));
    return true;
  }
  return false;
}

bool AttributeQuantizationTransform::DecodeParameters(
    const PointAttribute& attribute, DecoderBuffer* decoder_buffer) {
  const int num_components = attribute.num_components();
  if (data_type_ == DT_FLOAT64) {
    min_values_d_.resize(num_components);
    if (!decoder_buffer->Decode(&min_values_d_[0],
                                sizeof(double) * min_values_d_.size())) {
      return false;
    }
    if (!decoder_buffer->Decode(&range_d_)) {
      return false;
    }
    for (int c = 0; c < num_components; ++c) {
      if (std::isnan(min_values_d_[c]) || std::isinf(min_values_d_[c])) {
        return false;
      }
    }
    if (std::isnan(range_d_) || std::isinf(range_d_) || range_d_ < 0.0) {
      return false;
    }
    // Guard against degenerate ranges (the encoder never stores them, but a
    // corrupted stream could).
    if (range_d_ == 0.0) {
      range_d_ = 1.0;
    }
    SyncFloatParametersFromDouble();
  } else {
    data_type_ = DT_FLOAT32;
    min_values_.resize(num_components);
    if (!decoder_buffer->Decode(&min_values_[0],
                                sizeof(float) * min_values_.size())) {
      return false;
    }
    if (!decoder_buffer->Decode(&range_)) {
      return false;
    }
    SyncDoubleParametersFromFloat();
  }
  uint8_t quantization_bits;
  if (!decoder_buffer->Decode(&quantization_bits)) {
    return false;
  }
  if (!IsQuantizationValid(quantization_bits)) {
    return false;
  }
  quantization_bits_ = quantization_bits;
  return true;
}

void AttributeQuantizationTransform::GeneratePortableAttribute(
    const PointAttribute& attribute, int num_points,
    PointAttribute* target_attribute) const {
  DRACO_DCHECK(is_initialized());

  const int num_components = attribute.num_components();

  // Quantize all values using the order given by point_ids.
  int32_t* const portable_attribute_data = reinterpret_cast<int32_t*>(
      target_attribute->GetAddress(AttributeValueIndex(0)));
  if (data_type_ == DT_FLOAT64) {
    const double max_quantized_value =
        static_cast<double>((1u << static_cast<uint32_t>(quantization_bits_)) -
                            1);
    int32_t dst_index = 0;
    const std::unique_ptr<double[]> att_val(new double[num_components]);
    for (PointIndex i(0); i < num_points; ++i) {
      const AttributeValueIndex att_val_id = attribute.mapped_index(i);
      attribute.GetValue(att_val_id, att_val.get());
      for (int c = 0; c < num_components; ++c) {
        portable_attribute_data[dst_index++] = QuantizeDoubleValue(
            att_val[c], min_values_d_[c], max_quantized_value);
      }
    }
    return;
  }
  const uint32_t max_quantized_value = (1 << (quantization_bits_)) - 1;
  Quantizer quantizer;
  quantizer.Init(range(), max_quantized_value);
  int32_t dst_index = 0;
  const std::unique_ptr<float[]> att_val(new float[num_components]);
  for (PointIndex i(0); i < num_points; ++i) {
    const AttributeValueIndex att_val_id = attribute.mapped_index(i);
    attribute.GetValue(att_val_id, att_val.get());
    for (int c = 0; c < num_components; ++c) {
      const float value = (att_val[c] - min_values()[c]);
      const int32_t q_val = quantizer.QuantizeFloat(value);
      portable_attribute_data[dst_index++] = q_val;
    }
  }
}

void AttributeQuantizationTransform::GeneratePortableAttribute(
    const PointAttribute& attribute, const std::vector<PointIndex>& point_ids,
    int num_points, PointAttribute* target_attribute) const {
  DRACO_DCHECK(is_initialized());

  const int num_components = attribute.num_components();

  // Quantize all values using the order given by point_ids.
  int32_t* const portable_attribute_data = reinterpret_cast<int32_t*>(
      target_attribute->GetAddress(AttributeValueIndex(0)));
  if (data_type_ == DT_FLOAT64) {
    const double max_quantized_value =
        static_cast<double>((1u << static_cast<uint32_t>(quantization_bits_)) -
                            1);
    int32_t dst_index = 0;
    const std::unique_ptr<double[]> att_val(new double[num_components]);
    for (uint32_t i = 0; i < point_ids.size(); ++i) {
      const AttributeValueIndex att_val_id =
          attribute.mapped_index(point_ids[i]);
      attribute.GetValue(att_val_id, att_val.get());
      for (int c = 0; c < num_components; ++c) {
        portable_attribute_data[dst_index++] = QuantizeDoubleValue(
            att_val[c], min_values_d_[c], max_quantized_value);
      }
    }
    return;
  }
  const uint32_t max_quantized_value = (1 << (quantization_bits_)) - 1;
  Quantizer quantizer;
  quantizer.Init(range(), max_quantized_value);
  int32_t dst_index = 0;
  const std::unique_ptr<float[]> att_val(new float[num_components]);
  for (uint32_t i = 0; i < point_ids.size(); ++i) {
    const AttributeValueIndex att_val_id = attribute.mapped_index(point_ids[i]);
    attribute.GetValue(att_val_id, att_val.get());
    for (int c = 0; c < num_components; ++c) {
      const float value = (att_val[c] - min_values()[c]);
      const int32_t q_val = quantizer.QuantizeFloat(value);
      portable_attribute_data[dst_index++] = q_val;
    }
  }
}

int32_t AttributeQuantizationTransform::QuantizeDoubleValue(
    double value, double min_value, double max_quantized_value) const {
  // Normalize the value to the [0, 1] interval first. This avoids overflow
  // when the range is very small (where a precomputed max_quantized_value /
  // range multiplier could overflow to infinity) and underflow when the range
  // is very large.
  const double normalized = (value - min_value) / range_d_;
  double quantized_value =
      std::floor(normalized * max_quantized_value + 0.5);
  // Clamp the result to the valid range to guard against rounding errors for
  // values lying on (or marginally outside of) the quantization bounds.
  if (quantized_value < 0.0) {
    quantized_value = 0.0;
  } else if (quantized_value > max_quantized_value) {
    quantized_value = max_quantized_value;
  }
  return static_cast<int32_t>(quantized_value);
}

void AttributeQuantizationTransform::SyncDoubleParametersFromFloat() {
  min_values_d_.assign(min_values_.begin(), min_values_.end());
  range_d_ = static_cast<double>(range_);
}

void AttributeQuantizationTransform::SyncFloatParametersFromDouble() {
  min_values_.assign(min_values_d_.begin(), min_values_d_.end());
  range_ = static_cast<float>(range_d_);
}

}  // namespace draco
