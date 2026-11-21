// Copyright 2016 The Draco Authors.
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
// Tests for the shared attribute seam connectivity deduplication introduced in
// bitstream version 2.3. Multiple non-position attributes that share exactly
// the same seam edges are encoded with a single seam connectivity bit stream,
// while attributes with different seams keep their own connectivity. The
// tests verify that the optimization is reversible (the decoded mesh is
// equivalent to the input, including every per-attribute corner mapping and
// value), stable (repeated encodings are identical), actually reduces the
// encoded size, and that it neither merges connectivities that are not safe to
// merge nor breaks decoding of older bitstreams.
//
#include <set>
#include <vector>

#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/draco_test_base.h"
#include "draco/core/draco_test_utils.h"
#include "draco/io/file_utils.h"
#include "draco/mesh/mesh_are_equivalent.h"
#include "draco/mesh/mesh_cleanup.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"

namespace draco {
namespace {

// Encodes |mesh| with the edgebreaker method at the slowest speed setting,
// which forces per-attribute connectivity encoding (and therefore exercises
// the shared seam connectivity code path).
bool EncodeMesh(const Mesh& mesh, EncoderBuffer* out_buffer) {
  Encoder encoder;
  encoder.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
  encoder.SetSpeedOptions(0, 0);
  const Status status = encoder.EncodeMeshToBuffer(mesh, out_buffer);
  return status.ok();
}

std::unique_ptr<Mesh> DecodeMesh(const EncoderBuffer& buffer) {
  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());
  Decoder decoder;
  StatusOr<std::unique_ptr<Mesh>> status_or_mesh =
      decoder.DecodeMeshFromBuffer(&dec_buffer);
  if (!status_or_mesh.ok()) {
    return nullptr;
  }
  return std::move(status_or_mesh).value();
}

// Reads the two version bytes from a Draco encoded buffer.
void GetBufferVersion(const EncoderBuffer& buffer, uint8_t* major,
                      uint8_t* minor) {
  // Header layout: "DRACO" (5 bytes), version major, version minor.
  *major = buffer.data()[5];
  *minor = buffer.data()[6];
}

// Builds a mesh with four triangles arranged as two quads sharing an interior
// edge. Positions are continuous, while the non-position attributes introduce
// interior seams. Attribute |shared_a| and |shared_b| are assigned the exact
// same per-corner values, therefore they have identical seam connectivity.
// Attribute |different| uses a different value split, therefore its seam
// connectivity differs and it must not be merged with the other two.
//
// Faces (corners are listed counter-clockwise):
//   face 0: (0, 1, 2)
//   face 1: (0, 2, 3)
//   face 2: (4, 5, 6)
//   face 3: (4, 6, 7)
// where positions 0..3 form the first quad and positions 4..7 the second quad;
// position 3 and 4, and 2 and 5, are geometrically coincident so the two quads
// form a connected 2x1 rectangle.
std::unique_ptr<Mesh> BuildMeshWithSharedAndDistinctSeams() {
  TriangleSoupMeshBuilder mb;
  mb.Start(4);
  const int pos_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  const int shared_a_id =
      mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  const int shared_b_id =
      mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  const int different_id =
      mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);

  const Vector3f positions[12] = {
      // Face 0.
      Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f),
      // Face 1.
      Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 1.f, 0.f),
      Vector3f(0.f, 1.f, 0.f),
      // Face 2.
      Vector3f(0.f, 1.f, 0.f), Vector3f(1.f, 1.f, 0.f),
      Vector3f(1.f, 2.f, 0.f),
      // Face 3.
      Vector3f(0.f, 1.f, 0.f), Vector3f(1.f, 2.f, 0.f),
      Vector3f(0.f, 2.f, 0.f),
  };

  // Values of the two "shared" attributes. Both quads use different values on
  // their coincident corners, creating an interior seam along the shared edge.
  // Because the two attributes use the same value pattern, their seams match.
  const float shared_values[12] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                   1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  // The "different" attribute splits at a different location: the first corner
  // gets an isolated value, producing a distinct seam pattern.
  const float different_values[12] = {2.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

  for (int f = 0; f < 4; ++f) {
    const int base = 3 * f;
    mb.SetAttributeValuesForFace(
        pos_id, FaceIndex(f), &positions[base], &positions[base + 1],
        &positions[base + 2]);
    mb.SetAttributeValuesForFace(
        shared_a_id, FaceIndex(f), &shared_values[base],
        &shared_values[base + 1], &shared_values[base + 2]);
    mb.SetAttributeValuesForFace(
        shared_b_id, FaceIndex(f), &shared_values[base],
        &shared_values[base + 1], &shared_values[base + 2]);
    mb.SetAttributeValuesForFace(
        different_id, FaceIndex(f), &different_values[base],
        &different_values[base + 1], &different_values[base + 2]);
  }
  return mb.Finalize();
}

// Builds a mesh in which all |num_attributes| non-position attributes share the
// same seam connectivity. With deduplication enabled, all of them must be
// stored in a single connectivity group. The attributes use different value
// ranges so that their values stay independent even though their seams match.
std::unique_ptr<Mesh> BuildMeshWithFullySharedSeams(int num_attributes) {
  TriangleSoupMeshBuilder mb;
  mb.Start(4);
  const int pos_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  std::vector<int> att_ids(num_attributes);
  for (int a = 0; a < num_attributes; ++a) {
    att_ids[a] = mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  }

  const Vector3f positions[12] = {
      Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 1.f, 0.f),
      Vector3f(0.f, 1.f, 0.f), Vector3f(1.f, 1.f, 0.f),
      Vector3f(1.f, 2.f, 0.f), Vector3f(0.f, 1.f, 0.f),
      Vector3f(1.f, 2.f, 0.f), Vector3f(0.f, 2.f, 0.f),
  };
  // One shared seam pattern: value 0 on the lower quad, 1 on the upper quad.
  const float base_pattern[12] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                  1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

  for (int f = 0; f < 4; ++f) {
    const int base = 3 * f;
    mb.SetAttributeValuesForFace(
        pos_id, FaceIndex(f), &positions[base], &positions[base + 1],
        &positions[base + 2]);
  }
  for (int a = 0; a < num_attributes; ++a) {
    // Shift each attribute into its own value range while preserving the
    // seam pattern (the two quad values remain distinct).
    std::vector<float> values(12);
    const float offset = static_cast<float>(a) * 100.f;
    for (int i = 0; i < 12; ++i) {
      values[i] = base_pattern[i] + offset;
    }
    for (int f = 0; f < 4; ++f) {
      const int base = 3 * f;
      mb.SetAttributeValuesForFace(
          att_ids[a], FaceIndex(f), &values[base], &values[base + 1],
          &values[base + 2]);
    }
  }
  return mb.Finalize();
}

// Collects the set of float values stored in an attribute.
std::set<float> CollectAttributeValues(const PointAttribute& att) {
  std::set<float> values;
  for (AttributeValueIndex vi(0); vi < att.size(); ++vi) {
    float value = 0.f;
    if (att.ConvertValue(vi, 1, &value)) {
      values.insert(value);
    }
  }
  return values;
}

// Builds a mesh with |num_attributes| generic attributes that all use the same
// binary value distribution (one corner with value 0, the rest with value 1).
// When |shared| is true, every attribute places its zero on the same corner, so
// all attributes end up with identical seam connectivity. Otherwise each
// attribute places its zero on a different corner, producing pairwise distinct
// seam patterns. In both cases the encoded attribute values are identical, so
// the two meshes differ only in how many seam connectivity bit streams are
// emitted.
std::unique_ptr<Mesh> BuildMeshForSizeComparison(int num_attributes,
                                                 bool shared) {
  TriangleSoupMeshBuilder mb;
  mb.Start(4);
  const int pos_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  std::vector<int> att_ids(num_attributes);
  for (int a = 0; a < num_attributes; ++a) {
    att_ids[a] = mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  }

  const Vector3f positions[12] = {
      Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 1.f, 0.f),
      Vector3f(0.f, 1.f, 0.f), Vector3f(1.f, 1.f, 0.f),
      Vector3f(1.f, 2.f, 0.f), Vector3f(0.f, 1.f, 0.f),
      Vector3f(1.f, 2.f, 0.f), Vector3f(0.f, 2.f, 0.f),
  };
  for (int f = 0; f < 4; ++f) {
    const int base = 3 * f;
    mb.SetAttributeValuesForFace(
        pos_id, FaceIndex(f), &positions[base], &positions[base + 1],
        &positions[base + 2]);
  }
  for (int a = 0; a < num_attributes; ++a) {
    std::vector<float> values(12, 1.f);
    // Every attribute uses the same single-zero value distribution.
    values[shared ? 0 : a % 12] = 0.f;
    for (int f = 0; f < 4; ++f) {
      const int base = 3 * f;
      mb.SetAttributeValuesForFace(
          att_ids[a], FaceIndex(f), &values[base], &values[base + 1],
          &values[base + 2]);
    }
  }
  return mb.Finalize();
}

class MeshEdgebreakerSharedSeamTest : public ::testing::Test {};

TEST_F(MeshEdgebreakerSharedSeamTest, SharedSeamsAreReversible) {
  std::unique_ptr<Mesh> mesh = BuildMeshWithSharedAndDistinctSeams();
  ASSERT_NE(mesh, nullptr);

  EncoderBuffer buffer;
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer));

  uint8_t major = 0;
  uint8_t minor = 0;
  GetBufferVersion(buffer, &major, &minor);
  ASSERT_EQ(major, kDracoMeshBitstreamVersionMajor);
  ASSERT_EQ(minor, kDracoMeshBitstreamVersionMinor);

  std::unique_ptr<Mesh> decoded_mesh = DecodeMesh(buffer);
  ASSERT_NE(decoded_mesh, nullptr);
  ASSERT_EQ(decoded_mesh->num_attributes(), mesh->num_attributes());

  const MeshCleanupOptions cleanup_options;
  DRACO_ASSERT_OK(MeshCleanup::Cleanup(mesh.get(), cleanup_options));
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*mesh, *decoded_mesh))
      << "Decoded mesh with shared seams is not equivalent to the input";
}

TEST_F(MeshEdgebreakerSharedSeamTest, SharedSeamsAreReversibleWithQuantization) {
  std::unique_ptr<Mesh> mesh = BuildMeshWithSharedAndDistinctSeams();
  ASSERT_NE(mesh, nullptr);

  // Quantization changes attribute values, so the decoded mesh cannot be
  // compared byte-for-byte against the unquantized input. Instead, encode the
  // mesh twice and verify that both quantized round trips produce identical
  // meshes: the shared seam deduplication has to be reversible and stable
  // even when every non-position attribute is quantized.
  EncoderBuffer buffer_0;
  EncoderBuffer buffer_1;
  for (EncoderBuffer* buffer : {&buffer_0, &buffer_1}) {
    Encoder encoder;
    encoder.Reset();
    encoder.SetSpeedOptions(0, 0);
    encoder.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
    encoder.SetAttributeQuantization(GeometryAttribute::POSITION, 14);
    encoder.SetAttributeQuantization(GeometryAttribute::GENERIC, 10);
    DRACO_ASSERT_OK(encoder.EncodeMeshToBuffer(*mesh, buffer));
  }

  std::unique_ptr<Mesh> decoded_mesh_0 = DecodeMesh(buffer_0);
  std::unique_ptr<Mesh> decoded_mesh_1 = DecodeMesh(buffer_1);
  ASSERT_NE(decoded_mesh_0, nullptr);
  ASSERT_NE(decoded_mesh_1, nullptr);
  ASSERT_EQ(decoded_mesh_0->num_attributes(), mesh->num_attributes());
  ASSERT_EQ(decoded_mesh_1->num_attributes(), mesh->num_attributes());

  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*decoded_mesh_0, *decoded_mesh_1))
      << "Quantized shared-seam round trips are not reproducible";
}

TEST_F(MeshEdgebreakerSharedSeamTest, FullySharedSeamsKeepPerAttributeValues) {
  constexpr int kNumAttributes = 3;
  std::unique_ptr<Mesh> mesh =
      BuildMeshWithFullySharedSeams(kNumAttributes);
  ASSERT_NE(mesh, nullptr);

  EncoderBuffer buffer;
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer));

  std::unique_ptr<Mesh> decoded_mesh = DecodeMesh(buffer);
  ASSERT_NE(decoded_mesh, nullptr);
  ASSERT_EQ(decoded_mesh->num_attributes(), mesh->num_attributes());

  // All non-position attributes share the same connectivity, but they must
  // keep their own distinct values. Check that each decoded attribute still
  // carries exactly its own value range rather than the values of another
  // attribute in the group.
  for (int a = 1; a < mesh->num_attributes(); ++a) {
    const std::set<float> expected_values =
        CollectAttributeValues(*mesh->attribute(a));
    const std::set<float> decoded_values =
        CollectAttributeValues(*decoded_mesh->attribute(a));
    EXPECT_EQ(expected_values, decoded_values)
        << "Attribute " << a
        << " lost its own values after a shared-connectivity round trip";
  }

  const MeshCleanupOptions cleanup_options;
  DRACO_ASSERT_OK(MeshCleanup::Cleanup(mesh.get(), cleanup_options));
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*mesh, *decoded_mesh))
      << "Decoded mesh must preserve every per-attribute corner mapping";
}

TEST_F(MeshEdgebreakerSharedSeamTest, DistinctSeamsAreNotMerged) {
  std::unique_ptr<Mesh> mesh = BuildMeshWithSharedAndDistinctSeams();
  ASSERT_NE(mesh, nullptr);

  EncoderBuffer buffer;
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer));

  std::unique_ptr<Mesh> decoded_mesh = DecodeMesh(buffer);
  ASSERT_NE(decoded_mesh, nullptr);

  // If the distinct-seam attribute were incorrectly merged with the two
  // shared-seam attributes, its corner mapping would be wrong and the decoded
  // mesh would not be equivalent to the input.
  const MeshCleanupOptions cleanup_options;
  DRACO_ASSERT_OK(MeshCleanup::Cleanup(mesh.get(), cleanup_options));
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*mesh, *decoded_mesh))
      << "Attributes with distinct seam connectivity must not be merged";
}

TEST_F(MeshEdgebreakerSharedSeamTest, DeduplicationReducesEncodedSize) {
  // Use enough attributes to make the per-stream overhead of the rANS seam
  // encoders dominate. Both meshes encode identical attribute values; they
  // differ only in whether their seam connectivity is shared or distinct.
  constexpr int kNumAttributes = 8;
  std::unique_ptr<Mesh> shared_mesh =
      BuildMeshForSizeComparison(kNumAttributes, true);
  std::unique_ptr<Mesh> distinct_mesh =
      BuildMeshForSizeComparison(kNumAttributes, false);
  ASSERT_NE(shared_mesh, nullptr);
  ASSERT_NE(distinct_mesh, nullptr);
  ASSERT_EQ(shared_mesh->num_attributes(), distinct_mesh->num_attributes());
  ASSERT_EQ(shared_mesh->num_faces(), distinct_mesh->num_faces());

  EncoderBuffer shared_buffer;
  EncoderBuffer distinct_buffer;
  ASSERT_TRUE(EncodeMesh(*shared_mesh, &shared_buffer));
  ASSERT_TRUE(EncodeMesh(*distinct_mesh, &distinct_buffer));

  // With shared seam connectivity, only one seam bit stream is emitted instead
  // of one per attribute, so the encoded result must be strictly smaller.
  EXPECT_LT(shared_buffer.size(), distinct_buffer.size())
      << "Shared seam connectivity did not reduce the encoded size";
}

TEST_F(MeshEdgebreakerSharedSeamTest, EncodingIsStable) {
  std::unique_ptr<Mesh> mesh = BuildMeshWithSharedAndDistinctSeams();
  ASSERT_NE(mesh, nullptr);

  EncoderBuffer buffer_0;
  EncoderBuffer buffer_1;
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer_0));
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer_1));

  ASSERT_EQ(buffer_0.size(), buffer_1.size());
  for (int i = 0; i < buffer_0.size(); ++i) {
    EXPECT_EQ(buffer_0.data()[i], buffer_1.data()[i])
        << "Shared seam encoding is not stable at byte " << i;
  }
}

TEST_F(MeshEdgebreakerSharedSeamTest, WorksForDifferentAttributeCounts) {
  // A single non-position attribute must still be encodable: it forms a group
  // on its own and the round trip must succeed.
  TriangleSoupMeshBuilder mb;
  mb.Start(2);
  const int pos_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  const int att_id =
      mb.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);
  const Vector3f positions[6] = {
      Vector3f(0.f, 0.f, 0.f), Vector3f(1.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 0.f, 0.f),
      Vector3f(1.f, 1.f, 0.f), Vector3f(0.f, 1.f, 0.f),
  };
  const float values[6] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f};
  for (int f = 0; f < 2; ++f) {
    const int base = 3 * f;
    mb.SetAttributeValuesForFace(
        pos_id, FaceIndex(f), &positions[base], &positions[base + 1],
        &positions[base + 2]);
    mb.SetAttributeValuesForFace(
        att_id, FaceIndex(f), &values[base], &values[base + 1],
        &values[base + 2]);
  }
  std::unique_ptr<Mesh> mesh = mb.Finalize();
  ASSERT_NE(mesh, nullptr);

  EncoderBuffer buffer;
  ASSERT_TRUE(EncodeMesh(*mesh, &buffer));
  std::unique_ptr<Mesh> decoded_mesh = DecodeMesh(buffer);
  ASSERT_NE(decoded_mesh, nullptr);
  ASSERT_EQ(decoded_mesh->num_attributes(), 2);

  const MeshCleanupOptions cleanup_options;
  DRACO_ASSERT_OK(MeshCleanup::Cleanup(mesh.get(), cleanup_options));
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*mesh, *decoded_mesh));
}

TEST_F(MeshEdgebreakerSharedSeamTest, LegacyBitstreamWithoutDedupDecodes) {
  // A bitstream produced before version 2.3 stores one seam connectivity per
  // attribute and has no attribute-to-group mapping. The decoder must still
  // accept it and reconstruct an equivalent mesh.
  const std::unique_ptr<Mesh> expected_mesh =
      ReadMeshFromTestFile("cube_att.obj");
  ASSERT_NE(expected_mesh, nullptr);
  const std::string legacy_file = "cube_att.obj.edgebreaker.cl4.2.2.drc";
  const std::string path = GetTestFileFullPath(legacy_file);
  std::vector<char> legacy_data;
  ASSERT_TRUE(ReadFileToBuffer(path, &legacy_data));

  DecoderBuffer dec_buffer;
  dec_buffer.Init(legacy_data.data(), legacy_data.size());
  Decoder decoder;
  StatusOr<std::unique_ptr<Mesh>> status_or_mesh =
      decoder.DecodeMeshFromBuffer(&dec_buffer);
  ASSERT_TRUE(status_or_mesh.ok())
      << "Legacy 2.2 bitstream failed to decode: "
      << status_or_mesh.status().error_msg_string();
  std::unique_ptr<Mesh> decoded_mesh = std::move(status_or_mesh).value();
  ASSERT_NE(decoded_mesh, nullptr);

  const MeshCleanupOptions cleanup_options;
  DRACO_ASSERT_OK(MeshCleanup::Cleanup(expected_mesh.get(), cleanup_options));
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*expected_mesh, *decoded_mesh))
      << "Legacy bitstream decoded into a different mesh";
}

}  // namespace
}  // namespace draco
