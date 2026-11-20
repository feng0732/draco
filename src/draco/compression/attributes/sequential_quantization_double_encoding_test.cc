// Copyright 2026 The Draco Authors.
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
// Tests for quantization of double precision (DT_FLOAT64) attributes.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "draco/attributes/attribute_quantization_transform.h"
#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/draco_test_base.h"
#include "draco/mesh/mesh.h"
#include "draco/point_cloud/point_cloud.h"
#include "draco/point_cloud/point_cloud_builder.h"

namespace draco {

namespace {

// Creates a point cloud with a double precision position attribute and a
// single precision generic attribute. |positions| must contain 3 * num_points
// values.
std::unique_ptr<PointCloud> CreateDoublePointCloud(
    const std::vector<double>& positions,
    const std::vector<float>& generic_att) {
  PointCloudBuilder builder;
  const int num_points = static_cast<int>(positions.size() / 3);
  builder.Start(num_points);
  const int pos_att_id =
      builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT64);
  const int gen_att_id =
      builder.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  for (PointIndex i(0); i < num_points; ++i) {
    builder.SetAttributeValueForPoint(pos_att_id, i, &positions[3 * i.value()]);
    builder.SetAttributeValueForPoint(gen_att_id, i,
                                      &generic_att[i.value()]);
  }
  return builder.Finalize(false);
}

// Creates a mesh (grid of triangles) with double precision positions and a
// single precision texture coordinate attribute.
std::unique_ptr<Mesh> CreateDoubleMesh(const std::vector<double>& positions,
                                       const std::vector<float>& tex_coords,
                                       const std::vector<Mesh::Face>& faces) {
  std::unique_ptr<Mesh> mesh(new Mesh());
  mesh->SetNumFaces(faces.size());
  const int num_points = static_cast<int>(positions.size() / 3);
  mesh->set_num_points(num_points);
  for (size_t i = 0; i < faces.size(); ++i) {
    mesh->SetFace(FaceIndex(static_cast<uint32_t>(i)), faces[i]);
  }

  {
    GeometryAttribute va;
    va.Init(GeometryAttribute::POSITION, nullptr, 3, DT_FLOAT64, false,
            3 * sizeof(double), 0);
    const int att_id = mesh->AddAttribute(va, true, num_points);
    PointAttribute* const att = mesh->attribute(att_id);
    for (AttributeValueIndex i(0); i < num_points; ++i) {
      att->SetAttributeValue(i, &positions[3 * i.value()]);
    }
  }
  {
    GeometryAttribute va;
    va.Init(GeometryAttribute::TEX_COORD, nullptr, 2, DT_FLOAT32, false,
            2 * sizeof(float), 0);
    const int att_id = mesh->AddAttribute(va, true, num_points);
    PointAttribute* const att = mesh->attribute(att_id);
    for (AttributeValueIndex i(0); i < num_points; ++i) {
      att->SetAttributeValue(i, &tex_coords[2 * i.value()]);
    }
  }
  return mesh;
}

// Computes the maximum per-component difference between the original and the
// decoded double precision position attribute. Returns -1 on mismatch.
double ComputeMaxPositionError(const PointAttribute* original,
                               const PointAttribute* decoded) {
  if (original == nullptr || decoded == nullptr) {
    return -1.0;
  }
  if (decoded->data_type() != DT_FLOAT64) {
    return -1.0;
  }
  if (original->size() != decoded->size() ||
      original->num_components() != decoded->num_components()) {
    return -1.0;
  }
  const int num_components = original->num_components();
  std::vector<double> val_0(num_components);
  std::vector<double> val_1(num_components);
  double max_error = 0.0;
  for (AttributeValueIndex i(0); i < original->size(); ++i) {
    original->GetValue(i, val_0.data());
    decoded->GetValue(i, val_1.data());
    for (int c = 0; c < num_components; ++c) {
      if (std::isnan(val_1[c])) {
        return std::numeric_limits<double>::infinity();
      }
      const double error = std::fabs(val_0[c] - val_1[c]);
      if (error > max_error) {
        max_error = error;
      }
    }
  }
  return max_error;
}

// Computes the quantization error bound (half of the quantization step) plus
// a tolerance that accounts for the floating point representation of the
// input values.
double ComputeErrorBound(const std::vector<double>& positions,
                         int quantization_bits) {
  double min_val = positions[0];
  double max_val = positions[0];
  double max_abs = 0.0;
  for (double v : positions) {
    min_val = std::min(min_val, v);
    max_val = std::max(max_val, v);
    max_abs = std::max(max_abs, std::fabs(v));
  }
  double range = max_val - min_val;
  if (range == 0.0) {
    range = 1.0;
  }
  const double max_quantized_value =
      static_cast<double>((1 << quantization_bits) - 1);
  const double delta = range / max_quantized_value;
  // The quantization error is bounded by delta / 2. On top of that, the
  // decoded values are subject to the rounding error of the double precision
  // arithmetic used for values of the magnitude of the input.
  return 0.5 * delta +
         8.0 * std::numeric_limits<double>::epsilon() *
             (max_abs + range);
}

}  // namespace

class DoubleQuantizationTest : public ::testing::Test {
 protected:
  // Encodes and decodes a point cloud using the sequential encoder with the
  // given quantization bits for the position attribute.
  std::unique_ptr<PointCloud> EncodeAndDecodePointCloud(const PointCloud* pc,
                                                        int quantization_bits) {
    Encoder encoder;
    encoder.SetAttributeQuantization(GeometryAttribute::POSITION,
                                     quantization_bits);
    EncoderBuffer buffer;
    if (!encoder.EncodePointCloudToBuffer(*pc, &buffer).ok()) {
      return nullptr;
    }
    Decoder decoder;
    DecoderBuffer dec_buffer;
    dec_buffer.Init(buffer.data(), buffer.size());
    auto result = decoder.DecodePointCloudFromBuffer(&dec_buffer);
    if (!result.ok()) {
      return nullptr;
    }
    return std::move(result).value();
  }

  std::unique_ptr<Mesh> EncodeAndDecodeMesh(const Mesh* mesh,
                                            int quantization_bits) {
    // Use sequential mesh encoding to preserve the order of points and faces
    // so that connectivity and attribute mapping can be compared directly.
    Encoder encoder;
    encoder.SetEncodingMethod(MESH_SEQUENTIAL_ENCODING);
    encoder.SetAttributeQuantization(GeometryAttribute::POSITION,
                                     quantization_bits);
    EncoderBuffer buffer;
    if (!encoder.EncodeMeshToBuffer(*mesh, &buffer).ok()) {
      return nullptr;
    }
    Decoder decoder;
    DecoderBuffer dec_buffer;
    dec_buffer.Init(buffer.data(), buffer.size());
    auto result = decoder.DecodeMeshFromBuffer(&dec_buffer);
    if (!result.ok()) {
      return nullptr;
    }
    return std::move(result).value();
  }
};

// Tests basic encoding and decoding of a point cloud with double precision
// positions.
TEST_F(DoubleQuantizationTest, PointCloudDoublePositions) {
  const int kNumPoints = 100;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = -10.0 + 0.1 * i;
    positions[3 * i + 1] = 5.0 - 0.05 * i;
    positions[3 * i + 2] = 0.001 * i * i;
    generic_att[i] = 0.25f * i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  constexpr int kQuantizationBits = 16;
  std::unique_ptr<PointCloud> decoded_pc =
      EncodeAndDecodePointCloud(pc.get(), kQuantizationBits);
  ASSERT_NE(decoded_pc, nullptr);
  ASSERT_EQ(decoded_pc->num_points(), pc->num_points());

  const PointAttribute* const decoded_pos_att =
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(decoded_pos_att, nullptr);
  // The decoded attribute must preserve the double precision data type.
  ASSERT_EQ(decoded_pos_att->data_type(), DT_FLOAT64);

  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION), decoded_pos_att);
  EXPECT_LE(max_error,
            ComputeErrorBound(positions, kQuantizationBits));

  // The non-quantized generic attribute must be preserved exactly.
  const PointAttribute* const decoded_gen_att =
      decoded_pc->GetNamedAttribute(GeometryAttribute::GENERIC);
  ASSERT_NE(decoded_gen_att, nullptr);
  ASSERT_EQ(decoded_gen_att->size(), kNumPoints);
  for (AttributeValueIndex i(0); i < kNumPoints; ++i) {
    float val;
    decoded_gen_att->GetValue(i, &val);
    EXPECT_EQ(val, generic_att[i.value()]);
  }
}

// Tests that degenerate ranges (all values identical) do not produce NaNs or
// a wrong zero point.
TEST_F(DoubleQuantizationTest, DegenerateRange) {
  const int kNumPoints = 10;
  std::vector<double> positions(3 * kNumPoints, -123.456);
  std::vector<float> generic_att(kNumPoints, 1.f);
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  std::unique_ptr<PointCloud> decoded_pc =
      EncodeAndDecodePointCloud(pc.get(), 14);
  ASSERT_NE(decoded_pc, nullptr);
  const PointAttribute* const decoded_pos_att =
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(decoded_pos_att, nullptr);
  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION), decoded_pos_att);
  EXPECT_EQ(max_error, 0.0);
}

// Tests quantization of coordinates with a very large offset and a small
// spread (a scenario where single precision parameter computation would lose
// all significant bits).
TEST_F(DoubleQuantizationTest, LargeOffsetCoordinates) {
  const int kNumPoints = 50;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  const double offset = 1.0e12;
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = offset + 0.5 * i;
    positions[3 * i + 1] = -offset - 0.25 * i;
    positions[3 * i + 2] = offset + 1.0 - 0.1 * i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  constexpr int kQuantizationBits = 20;
  std::unique_ptr<PointCloud> decoded_pc =
      EncodeAndDecodePointCloud(pc.get(), kQuantizationBits);
  ASSERT_NE(decoded_pc, nullptr);
  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION),
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION));
  EXPECT_LE(max_error, ComputeErrorBound(positions, kQuantizationBits));
}

// Tests quantization of very small coordinates (the quantization range is
// tiny and a naive max_quantized_value / range multiplier would overflow).
TEST_F(DoubleQuantizationTest, TinyCoordinates) {
  const int kNumPoints = 50;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = 1.0e-200 + 1.0e-201 * i;
    positions[3 * i + 1] = -1.0e-200 - 1.0e-202 * i;
    positions[3 * i + 2] = 1.0e-205 * i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  constexpr int kQuantizationBits = 16;
  std::unique_ptr<PointCloud> decoded_pc =
      EncodeAndDecodePointCloud(pc.get(), kQuantizationBits);
  ASSERT_NE(decoded_pc, nullptr);
  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION),
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION));
  EXPECT_LE(max_error, ComputeErrorBound(positions, kQuantizationBits));
}

// Tests quantization of coordinates with a very large magnitude.
TEST_F(DoubleQuantizationTest, HugeCoordinates) {
  const int kNumPoints = 50;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = -1.0e150 + 1.0e148 * i;
    positions[3 * i + 1] = 1.0e150 - 1.0e148 * i;
    positions[3 * i + 2] = 0.5e150;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  constexpr int kQuantizationBits = 16;
  std::unique_ptr<PointCloud> decoded_pc =
      EncodeAndDecodePointCloud(pc.get(), kQuantizationBits);
  ASSERT_NE(decoded_pc, nullptr);
  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION),
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION));
  EXPECT_LE(max_error, ComputeErrorBound(positions, kQuantizationBits));
}

// Tests explicit quantization with a negative origin (translation) for a
// double precision attribute.
TEST_F(DoubleQuantizationTest, ExplicitQuantizationNegativeOrigin) {
  const int kNumPoints = 50;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  const double origin[3] = {-1.0e9 - 0.123456789, -2.5, -1.0e-7};
  const double range = 100.0;
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = origin[0] + 1.5 * i;
    positions[3 * i + 1] = origin[1] + 0.5 * i;
    positions[3 * i + 2] = origin[2] + 0.25 * i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  constexpr int kQuantizationBits = 18;
  Encoder encoder;
  encoder.SetAttributeExplicitQuantization(GeometryAttribute::POSITION,
                                           kQuantizationBits, 3, origin,
                                           range);
  EncoderBuffer buffer;
  ASSERT_TRUE(encoder.EncodePointCloudToBuffer(*pc, &buffer).ok());
  Decoder decoder;
  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());
  auto result = decoder.DecodePointCloudFromBuffer(&dec_buffer);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PointCloud> decoded_pc = std::move(result).value();
  ASSERT_NE(decoded_pc, nullptr);

  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION),
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION));
  const double max_quantized_value =
      static_cast<double>((1 << kQuantizationBits) - 1);
  const double delta = range / max_quantized_value;
  EXPECT_LE(max_error,
            0.5 * delta +
                8.0 * std::numeric_limits<double>::epsilon() *
                    (std::fabs(origin[0]) + range));
}

// Tests encoding of a mesh with double precision positions. Verifies that
// connectivity, attribute mapping and other attributes are preserved.
TEST_F(DoubleQuantizationTest, MeshDoublePositions) {
  // Create a simple grid mesh.
  constexpr int kGridSize = 8;
  constexpr int kNumPoints = kGridSize * kGridSize;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> tex_coords(2 * kNumPoints);
  for (int y = 0; y < kGridSize; ++y) {
    for (int x = 0; x < kGridSize; ++x) {
      const int i = y * kGridSize + x;
      positions[3 * i + 0] = -1000.5 + 2.5 * x;
      positions[3 * i + 1] = 500.25 - 1.25 * y;
      positions[3 * i + 2] = 0.125 * x * y;
      tex_coords[2 * i + 0] = 0.1f * x;
      tex_coords[2 * i + 1] = 0.2f * y;
    }
  }
  std::vector<Mesh::Face> faces;
  for (int y = 0; y < kGridSize - 1; ++y) {
    for (int x = 0; x < kGridSize - 1; ++x) {
      const int i = y * kGridSize + x;
      faces.push_back(
          Mesh::Face({PointIndex(i), PointIndex(i + 1),
                      PointIndex(i + kGridSize)}));
      faces.push_back(
          Mesh::Face({PointIndex(i + 1), PointIndex(i + kGridSize + 1),
                      PointIndex(i + kGridSize)}));
    }
  }
  std::unique_ptr<Mesh> mesh = CreateDoubleMesh(positions, tex_coords, faces);
  ASSERT_NE(mesh, nullptr);
  mesh->DeduplicateAttributeValues();
  mesh->DeduplicatePointIds();

  constexpr int kQuantizationBits = 14;
  std::unique_ptr<Mesh> decoded_mesh =
      EncodeAndDecodeMesh(mesh.get(), kQuantizationBits);
  ASSERT_NE(decoded_mesh, nullptr);
  ASSERT_EQ(decoded_mesh->num_points(), mesh->num_points());
  ASSERT_EQ(decoded_mesh->num_faces(), mesh->num_faces());

  // Connectivity must be preserved.
  for (FaceIndex fi(0); fi < mesh->num_faces(); ++fi) {
    EXPECT_EQ(decoded_mesh->face(fi), mesh->face(fi));
  }

  const PointAttribute* const decoded_pos_att =
      decoded_mesh->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(decoded_pos_att, nullptr);
  ASSERT_EQ(decoded_pos_att->data_type(), DT_FLOAT64);
  const double max_error = ComputeMaxPositionError(
      mesh->GetNamedAttribute(GeometryAttribute::POSITION), decoded_pos_att);
  EXPECT_LE(max_error, ComputeErrorBound(positions, kQuantizationBits));

  // Texture coordinates are not quantized and must be preserved exactly.
  const PointAttribute* const decoded_tex_att =
      decoded_mesh->GetNamedAttribute(GeometryAttribute::TEX_COORD);
  ASSERT_NE(decoded_tex_att, nullptr);
  ASSERT_EQ(decoded_tex_att->size(), kNumPoints);
  for (AttributeValueIndex i(0); i < kNumPoints; ++i) {
    float val[2];
    decoded_tex_att->GetValue(i, val);
    EXPECT_EQ(val[0], tex_coords[2 * i.value()]);
    EXPECT_EQ(val[1], tex_coords[2 * i.value() + 1]);
  }
}

// Tests encoding of a mesh with double precision positions using the default
// (edgebreaker) encoding method. The edgebreaker encoder does not preserve
// the order of points, so the test verifies geometric equivalence of the
// decoded mesh instead of per-index equality.
TEST_F(DoubleQuantizationTest, MeshDoublePositionsEdgebreaker) {
  constexpr int kGridSize = 8;
  constexpr int kNumPoints = kGridSize * kGridSize;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> tex_coords(2 * kNumPoints, 0.f);
  for (int y = 0; y < kGridSize; ++y) {
    for (int x = 0; x < kGridSize; ++x) {
      const int i = y * kGridSize + x;
      positions[3 * i + 0] = -1000.5 + 2.5 * x;
      positions[3 * i + 1] = 500.25 - 1.25 * y;
      positions[3 * i + 2] = 0.125 * x * y;
    }
  }
  std::vector<Mesh::Face> faces;
  for (int y = 0; y < kGridSize - 1; ++y) {
    for (int x = 0; x < kGridSize - 1; ++x) {
      const int i = y * kGridSize + x;
      faces.push_back(
          Mesh::Face({PointIndex(i), PointIndex(i + 1),
                      PointIndex(i + kGridSize)}));
      faces.push_back(
          Mesh::Face({PointIndex(i + 1), PointIndex(i + kGridSize + 1),
                      PointIndex(i + kGridSize)}));
    }
  }
  std::unique_ptr<Mesh> mesh = CreateDoubleMesh(positions, tex_coords, faces);
  ASSERT_NE(mesh, nullptr);
  mesh->DeduplicateAttributeValues();
  mesh->DeduplicatePointIds();

  constexpr int kQuantizationBits = 14;
  Encoder encoder;
  encoder.SetAttributeQuantization(GeometryAttribute::POSITION,
                                   kQuantizationBits);
  EncoderBuffer buffer;
  ASSERT_TRUE(encoder.EncodeMeshToBuffer(*mesh, &buffer).ok());
  Decoder decoder;
  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());
  auto result = decoder.DecodeMeshFromBuffer(&dec_buffer);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<Mesh> decoded_mesh = std::move(result).value();
  ASSERT_NE(decoded_mesh, nullptr);
  ASSERT_EQ(decoded_mesh->num_points(), mesh->num_points());
  ASSERT_EQ(decoded_mesh->num_faces(), mesh->num_faces());

  const PointAttribute* const pos_att =
      decoded_mesh->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(pos_att, nullptr);
  ASSERT_EQ(pos_att->data_type(), DT_FLOAT64);
  const double bound = ComputeErrorBound(positions, kQuantizationBits);

  // Map every decoded point to the nearest original point. Note that the
  // decoded attribute may not use identity mapping between point indices and
  // attribute value indices.
  std::vector<int> dec_to_orig(kNumPoints, -1);
  std::vector<double> val(3);
  for (PointIndex i(0); i < kNumPoints; ++i) {
    pos_att->GetValue(pos_att->mapped_index(i), val.data());
    double best = std::numeric_limits<double>::infinity();
    int best_j = -1;
    for (int j = 0; j < kNumPoints; ++j) {
      const double dist = std::fabs(val[0] - positions[3 * j]) +
                          std::fabs(val[1] - positions[3 * j + 1]) +
                          std::fabs(val[2] - positions[3 * j + 2]);
      if (dist < best) {
        best = dist;
        best_j = j;
      }
    }
    ASSERT_LE(best, 3.0 * bound);
    dec_to_orig[i.value()] = best_j;
  }
  // Every original point must be covered by some decoded point.
  std::vector<bool> covered(kNumPoints, false);
  for (int i = 0; i < kNumPoints; ++i) {
    if (dec_to_orig[i] >= 0) {
      covered[dec_to_orig[i]] = true;
    }
  }
  for (int j = 0; j < kNumPoints; ++j) {
    EXPECT_TRUE(covered[j]);
  }
  // Every decoded face must correspond to an original face (compared as
  // unordered vertex sets).
  std::vector<std::array<int, 3>> orig_triangles;
  for (const auto& face : faces) {
    std::array<int, 3> tri = {{static_cast<int>(face[0].value()),
                               static_cast<int>(face[1].value()),
                               static_cast<int>(face[2].value())}};
    std::sort(tri.begin(), tri.end());
    orig_triangles.push_back(tri);
  }
  std::sort(orig_triangles.begin(), orig_triangles.end());
  for (FaceIndex fi(0); fi < decoded_mesh->num_faces(); ++fi) {
    const auto& face = decoded_mesh->face(fi);
    std::array<int, 3> tri = {{dec_to_orig[face[0].value()],
                               dec_to_orig[face[1].value()],
                               dec_to_orig[face[2].value()]}};
    std::sort(tri.begin(), tri.end());
    EXPECT_TRUE(std::binary_search(orig_triangles.begin(),
                                   orig_triangles.end(), tri));
  }
}

// Tests that a double precision attribute that is not quantized is encoded
// losslessly (backwards compatibility of non-quantized calls).
TEST_F(DoubleQuantizationTest, NoQuantizationIsLossless) {
  const int kNumPoints = 20;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = 1.0e10 + 0.1 * i;
    positions[3 * i + 1] = -3.14 * i;
    positions[3 * i + 2] = 1.0e-5 * i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);

  Encoder encoder;
  EncoderBuffer buffer;
  ASSERT_TRUE(encoder.EncodePointCloudToBuffer(*pc, &buffer).ok());
  Decoder decoder;
  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());
  auto result = decoder.DecodePointCloudFromBuffer(&dec_buffer);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PointCloud> decoded_pc = std::move(result).value();
  ASSERT_NE(decoded_pc, nullptr);
  const double max_error = ComputeMaxPositionError(
      pc->GetNamedAttribute(GeometryAttribute::POSITION),
      decoded_pc->GetNamedAttribute(GeometryAttribute::POSITION));
  EXPECT_EQ(max_error, 0.0);
}

// Tests the AttributeQuantizationTransform directly on a double precision
// attribute, including parameter encoding/decoding round trip.
TEST_F(DoubleQuantizationTest, TransformParameterRoundTrip) {
  const int kNumPoints = 30;
  std::vector<double> positions(3 * kNumPoints);
  std::vector<float> generic_att(kNumPoints, 0.f);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = -7.5 + 0.5 * i;
    positions[3 * i + 1] = 2.0 - 0.1 * i;
    positions[3 * i + 2] = 1.0e6 + i;
  }
  std::unique_ptr<PointCloud> pc =
      CreateDoublePointCloud(positions, generic_att);
  ASSERT_NE(pc, nullptr);
  const PointAttribute* const pos_att =
      pc->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(pos_att, nullptr);

  constexpr int kQuantizationBits = 15;
  AttributeQuantizationTransform transform;
  ASSERT_TRUE(transform.ComputeParameters(*pos_att, kQuantizationBits));
  ASSERT_TRUE(transform.is_initialized());
  ASSERT_EQ(transform.data_type(), DT_FLOAT64);
  ASSERT_EQ(transform.quantization_bits(), kQuantizationBits);

  // Encode and decode the parameters.
  EncoderBuffer buffer;
  ASSERT_TRUE(transform.EncodeParameters(&buffer));
  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());
  AttributeQuantizationTransform decoded_transform;
  decoded_transform.set_data_type(DT_FLOAT64);
  ASSERT_TRUE(decoded_transform.DecodeParameters(*pos_att, &dec_buffer));
  ASSERT_EQ(decoded_transform.quantization_bits(), kQuantizationBits);
  ASSERT_EQ(decoded_transform.range_d(), transform.range_d());
  ASSERT_EQ(decoded_transform.min_values_d(), transform.min_values_d());

  // The range must cover the maximum component span.
  double max_span = 0.0;
  for (int c = 0; c < 3; ++c) {
    double min_c = positions[c];
    double max_c = positions[c];
    for (int i = 0; i < kNumPoints; ++i) {
      min_c = std::min(min_c, positions[3 * i + c]);
      max_c = std::max(max_c, positions[3 * i + c]);
    }
    max_span = std::max(max_span, max_c - min_c);
    // The minimum values must match the component minima exactly.
    EXPECT_EQ(transform.min_value_d(c), min_c);
  }
  EXPECT_EQ(transform.range_d(), max_span);

  // Transform and inverse transform must satisfy the error contract.
  auto portable_att =
      transform.InitTransformedAttribute(*pos_att, pos_att->size());
  ASSERT_NE(portable_att, nullptr);
  ASSERT_TRUE(
      transform.TransformAttribute(*pos_att, {}, portable_att.get()));

  PointAttribute dequantized_att;
  dequantized_att.Init(GeometryAttribute::POSITION, 3, DT_FLOAT64, false,
                       pos_att->size());
  dequantized_att.SetIdentityMapping();
  ASSERT_TRUE(transform.InverseTransformAttribute(*portable_att,
                                                  &dequantized_att));
  const double max_error =
      ComputeMaxPositionError(pos_att, &dequantized_att);
  EXPECT_LE(max_error, ComputeErrorBound(positions, kQuantizationBits));
}

// Tests that the single precision path of the transform is unchanged and
// still satisfies its error contract.
TEST_F(DoubleQuantizationTest, FloatTransformUnchanged) {
  PointCloudBuilder builder;
  const int kNumPoints = 30;
  builder.Start(kNumPoints);
  const int pos_att_id =
      builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  std::vector<float> positions(3 * kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    positions[3 * i + 0] = -3.f + 0.25f * i;
    positions[3 * i + 1] = 1.5f - 0.5f * i;
    positions[3 * i + 2] = 0.01f * i;
    builder.SetAttributeValueForPoint(pos_att_id, PointIndex(i),
                                      &positions[3 * i]);
  }
  std::unique_ptr<PointCloud> pc = builder.Finalize(false);
  ASSERT_NE(pc, nullptr);
  const PointAttribute* const pos_att =
      pc->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(pos_att, nullptr);

  constexpr int kQuantizationBits = 12;
  AttributeQuantizationTransform transform;
  ASSERT_TRUE(transform.ComputeParameters(*pos_att, kQuantizationBits));
  ASSERT_EQ(transform.data_type(), DT_FLOAT32);

  auto portable_att =
      transform.InitTransformedAttribute(*pos_att, pos_att->size());
  ASSERT_TRUE(
      transform.TransformAttribute(*pos_att, {}, portable_att.get()));

  PointAttribute dequantized_att;
  dequantized_att.Init(GeometryAttribute::POSITION, 3, DT_FLOAT32, false,
                       pos_att->size());
  dequantized_att.SetIdentityMapping();
  ASSERT_TRUE(transform.InverseTransformAttribute(*portable_att,
                                                  &dequantized_att));

  const double max_quantized_value =
      static_cast<double>((1 << kQuantizationBits) - 1);
  const double delta = transform.range() / max_quantized_value;
  std::vector<float> val(3);
  for (AttributeValueIndex i(0); i < pos_att->size(); ++i) {
    dequantized_att.GetValue(i, val.data());
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(val[c], positions[3 * i.value() + c], 0.5 * delta * 1.001f);
    }
  }
}

}  // namespace draco
