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
#include <sstream>

#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/compression/mesh/mesh_edgebreaker_decoder.h"
#include "draco/compression/mesh/mesh_edgebreaker_encoder.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/draco_test_base.h"
#include "draco/core/draco_test_utils.h"
#include "draco/io/mesh_io.h"
#include "draco/io/obj_decoder.h"
#include "draco/mesh/mesh_are_equivalent.h"
#include "draco/mesh/mesh_cleanup.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"

namespace draco {

class MeshEdgebreakerEncodingTest : public ::testing::Test {
 protected:
  void TestFile(const std::string& file_name) { TestFile(file_name, -1); }

  void TestFile(const std::string& file_name, int compression_level) {
    const std::unique_ptr<Mesh> mesh(ReadMeshFromTestFile(file_name));
    ASSERT_NE(mesh, nullptr) << "Failed to load test model " << file_name;

    TestMesh(mesh.get(), compression_level);
  }

  void TestMesh(Mesh* mesh, int compression_level) {
    EncoderBuffer buffer;
    MeshEdgebreakerEncoder encoder;
    EncoderOptions encoder_options = EncoderOptions::CreateDefaultOptions();
    encoder_options.SetSpeed(10 - compression_level, 10 - compression_level);
    encoder.SetMesh(*mesh);
    DRACO_ASSERT_OK(encoder.Encode(encoder_options, &buffer));

    DecoderBuffer dec_buffer;
    dec_buffer.Init(buffer.data(), buffer.size());
    MeshEdgebreakerDecoder decoder;

    std::unique_ptr<Mesh> decoded_mesh(new Mesh());
    DecoderOptions dec_options;
    DRACO_ASSERT_OK(
        decoder.Decode(dec_options, &dec_buffer, decoded_mesh.get()));

    // Cleanup the input mesh to make sure that input and output can be
    // compared (edgebreaker method discards degenerated triangles and isolated
    // vertices).
    const MeshCleanupOptions options;
    DRACO_ASSERT_OK(MeshCleanup::Cleanup(mesh, options));

    MeshAreEquivalent eq;
    ASSERT_TRUE(eq(*mesh, *decoded_mesh.get()))
        << "Decoded mesh is not the same as the input";
  }
};

TEST_F(MeshEdgebreakerEncodingTest, TestNmOBJ) {
  const std::string file_name = "test_nm.obj";
  TestFile(file_name);
}

TEST_F(MeshEdgebreakerEncodingTest, ThreeFacesOBJ) {
  const std::string file_name = "extra_vertex.obj";
  TestFile(file_name);
}

TEST_F(MeshEdgebreakerEncodingTest, TestPly) {
  // Tests whether the edgebreaker successfully encodes and decodes the test
  // file (ply with color).
  const std::string file_name = "test_pos_color.ply";
  TestFile(file_name);
}

TEST_F(MeshEdgebreakerEncodingTest, TestMultiAttributes) {
  // Tests encoding of model with many attributes.
  const std::string file_name = "cube_att.obj";
  TestFile(file_name, 10);
}

TEST_F(MeshEdgebreakerEncodingTest, TestEncoderReuse) {
  // Tests whether the edgebreaker encoder can be reused multiple times to
  // encode a given mesh.
  const std::string file_name = "test_pos_color.ply";
  const std::unique_ptr<Mesh> mesh(ReadMeshFromTestFile(file_name));
  ASSERT_NE(mesh, nullptr) << "Failed to load test model " << file_name;

  MeshEdgebreakerEncoder encoder;
  EncoderOptions encoder_options = EncoderOptions::CreateDefaultOptions();
  encoder.SetMesh(*mesh);
  EncoderBuffer buffer_0, buffer_1;
  DRACO_ASSERT_OK(encoder.Encode(encoder_options, &buffer_0));
  DRACO_ASSERT_OK(encoder.Encode(encoder_options, &buffer_1));

  // Make sure both buffer are identical.
  ASSERT_EQ(buffer_0.size(), buffer_1.size());
  for (int i = 0; i < buffer_0.size(); ++i) {
    ASSERT_EQ(buffer_0.data()[i], buffer_1.data()[i]);
  }
}

TEST_F(MeshEdgebreakerEncodingTest, TestDecoderReuse) {
  // Tests whether the edgebreaker decoder can be reused multiple times to
  // decode a given mesh.
  const std::string file_name = "test_pos_color.ply";
  const std::unique_ptr<Mesh> mesh(ReadMeshFromTestFile(file_name));
  ASSERT_NE(mesh, nullptr) << "Failed to load test model " << file_name;

  MeshEdgebreakerEncoder encoder;
  EncoderOptions encoder_options = EncoderOptions::CreateDefaultOptions();
  encoder.SetMesh(*mesh);
  EncoderBuffer buffer;
  DRACO_ASSERT_OK(encoder.Encode(encoder_options, &buffer));

  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());

  MeshEdgebreakerDecoder decoder;

  // Decode the mesh two times.
  std::unique_ptr<Mesh> decoded_mesh_0(new Mesh());
  DecoderOptions dec_options;
  DRACO_ASSERT_OK(
      decoder.Decode(dec_options, &dec_buffer, decoded_mesh_0.get()));

  dec_buffer.Init(buffer.data(), buffer.size());
  std::unique_ptr<Mesh> decoded_mesh_1(new Mesh());
  DRACO_ASSERT_OK(
      decoder.Decode(dec_options, &dec_buffer, decoded_mesh_1.get()));

  // Make sure both of the meshes are identical.
  MeshAreEquivalent eq;
  ASSERT_TRUE(eq(*decoded_mesh_0.get(), *decoded_mesh_1.get()))
      << "Decoded meshes are not the same";
}

TEST_F(MeshEdgebreakerEncodingTest, TestSingleConnectivityEncoding) {
  // Tests whether the edgebreaker method successfully encodes a mesh with
  // multiple attributes using single connectivity by breaking the mesh along
  // attribute seams.
  const std::string file_name = "cube_att.obj";
  const std::unique_ptr<Mesh> mesh(ReadMeshFromTestFile(file_name));
  ASSERT_NE(mesh, nullptr) << "Failed to load test model " << file_name;

  for (int i = 0; i < 2; ++i) {
    // Set the option to enable/disable single connectivity encoding.
    EncoderOptionsBase<GeometryAttribute::Type> options =
        EncoderOptionsBase<GeometryAttribute::Type>::CreateDefaultOptions();
    options.SetGlobalBool("split_mesh_on_seams", i == 0 ? true : false);

    EncoderBuffer buffer;
    draco::Encoder encoder;
    encoder.Reset(options);
    encoder.SetSpeedOptions(0, 0);
    encoder.SetAttributeQuantization(GeometryAttribute::POSITION, 8);
    encoder.SetAttributeQuantization(GeometryAttribute::TEX_COORD, 8);
    encoder.SetAttributeQuantization(GeometryAttribute::NORMAL, 8);
    encoder.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
    DRACO_ASSERT_OK(encoder.EncodeMeshToBuffer(*mesh, &buffer));

    DecoderBuffer dec_buffer;
    dec_buffer.Init(buffer.data(), buffer.size());

    Decoder decoder;
    auto dec_mesh = decoder.DecodeMeshFromBuffer(&dec_buffer).value();
    ASSERT_NE(dec_mesh, nullptr);
    ASSERT_EQ(dec_mesh->num_points(), 24);
    ASSERT_EQ(dec_mesh->num_attributes(), 3);
    ASSERT_EQ(dec_mesh->attribute(0)->size(), i == 0 ? 24 : 8);
    ASSERT_EQ(dec_mesh->attribute(1)->size(), 24);
    ASSERT_EQ(dec_mesh->attribute(2)->size(), 24);
  }
}

TEST_F(MeshEdgebreakerEncodingTest, TestWrongAttributeOrder) {
  // Tests whether the edgebreaker method successfully encodes a mesh where the
  // input attributes are in wrong order (because of their internal
  // dependencies). In such case the attributes should be rearranged to the
  // correct order.
  TriangleSoupMeshBuilder mb;
  mb.Start(1);
  const int32_t norm_att_id =
      mb.AddAttribute(GeometryAttribute::NORMAL, 3, DT_FLOAT32);
  const int32_t pos_att_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);

  mb.SetAttributeValuesForFace(
      pos_att_id, FaceIndex(0), Vector3f(0.f, 0.f, 0.f).data(),
      Vector3f(1.f, 0.f, 0.f).data(), Vector3f(0.f, 1.f, 0.f).data());

  mb.SetAttributeValuesForFace(
      norm_att_id, FaceIndex(0), Vector3f(0.f, 0.f, 1.f).data(),
      Vector3f(0.f, 0.f, 0.f).data(), Vector3f(0.f, 0.f, 1.f).data());
  std::unique_ptr<Mesh> mesh = mb.Finalize();
  ASSERT_NE(mesh, nullptr);
  ASSERT_EQ(mesh->num_attributes(), 2);
  ASSERT_EQ(mesh->attribute(0)->attribute_type(), GeometryAttribute::NORMAL);
  ASSERT_EQ(mesh->attribute(1)->attribute_type(), GeometryAttribute::POSITION);

  EncoderBuffer buffer;
  draco::Encoder encoder;
  encoder.SetSpeedOptions(3, 3);
  encoder.SetAttributeQuantization(GeometryAttribute::POSITION, 8);
  encoder.SetAttributeQuantization(GeometryAttribute::NORMAL, 8);
  encoder.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
  DRACO_ASSERT_OK(encoder.EncodeMeshToBuffer(*mesh, &buffer));

  DecoderBuffer dec_buffer;
  dec_buffer.Init(buffer.data(), buffer.size());

  Decoder decoder;
  auto dec_mesh = decoder.DecodeMeshFromBuffer(&dec_buffer).value();
  ASSERT_NE(dec_mesh, nullptr);
  ASSERT_EQ(dec_mesh->num_attributes(), 2);
  ASSERT_EQ(dec_mesh->attribute(0)->attribute_type(),
            GeometryAttribute::POSITION);
  ASSERT_EQ(dec_mesh->attribute(1)->attribute_type(),
            GeometryAttribute::NORMAL);
}

TEST_F(MeshEdgebreakerEncodingTest, TestDegenerateMesh) {
  // Tests whether we can process a mesh that contains degenerate faces only.
  const std::string file_name = "degenerate_mesh.obj";
  const std::unique_ptr<Mesh> mesh(ReadMeshFromTestFile(file_name));
  ASSERT_NE(mesh, nullptr) << "Failed to load test model " << file_name;
  EncoderBuffer buffer;
  MeshEdgebreakerEncoder encoder;
  EncoderOptions encoder_options = EncoderOptions::CreateDefaultOptions();
  encoder.SetMesh(*mesh);
  // We expect the encoding to fail as edgebreaker can only process valid faces.
  ASSERT_FALSE(encoder.Encode(encoder_options, &buffer).ok());
}

// Builds a mesh in which a single geometric position is shared by many
// corners, but those corners carry distinct non-position attribute locations
// (one position -> multiple attribute locations), together with a face that is
// degenerate with respect to the position attribute. Such a degenerate face
// has distinct point ids on its duplicated corners, so it must still be
// excluded from the edgebreaker connectivity on both the per-attribute and the
// single-connectivity ("split on seams") paths.
TEST_F(MeshEdgebreakerEncodingTest,
       TestMultiPositionAttributesWithDegenerateFace) {
  TriangleSoupMeshBuilder mb;
  mb.Start(3);
  const int32_t pos_att_id =
      mb.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  const int32_t tex_att_id =
      mb.AddAttribute(GeometryAttribute::TEX_COORD, 2, DT_FLOAT32);

  const Vector3f a(0.f, 0.f, 0.f), b(1.f, 0.f, 0.f), c(0.f, 1.f, 0.f),
      d(1.f, 1.f, 0.f);

  // Two distinct valid faces sharing the edge (b, c).
  mb.SetAttributeValuesForFace(pos_att_id, FaceIndex(0), a.data(), b.data(),
                               c.data());
  mb.SetAttributeValuesForFace(tex_att_id, FaceIndex(0),
                               Vector3f(0.f, 0.f, 0.f).data(),
                               Vector3f(1.f, 0.f, 0.f).data(),
                               Vector3f(0.f, 1.f, 0.f).data());
  mb.SetAttributeValuesForFace(pos_att_id, FaceIndex(1), b.data(), d.data(),
                               c.data());
  mb.SetAttributeValuesForFace(tex_att_id, FaceIndex(1),
                               Vector3f(1.f, 0.f, 0.f).data(),
                               Vector3f(1.f, 1.f, 0.f).data(),
                               Vector3f(0.f, 1.f, 0.f).data());

  // A position-degenerate face (a repeated). Its three corners get distinct
  // texture coordinates and therefore distinct point ids, which previously hid
  // the degeneracy from the point-id based corner table.
  mb.SetAttributeValuesForFace(pos_att_id, FaceIndex(2), a.data(), a.data(),
                               b.data());
  mb.SetAttributeValuesForFace(tex_att_id, FaceIndex(2),
                               Vector3f(0.f, 0.f, 0.f).data(),
                               Vector3f(1.f, 1.f, 0.f).data(),
                               Vector3f(1.f, 0.f, 0.f).data());

  std::unique_ptr<Mesh> mesh = mb.Finalize();
  ASSERT_NE(mesh, nullptr);

  // Run under both connectivity paths. Speed >= 6 selects the single
  // connectivity fast path while lower speeds encode each attribute's
  // connectivity separately.
  for (const bool split_mesh_on_seams : {false, true}) {
    EncoderOptionsBase<GeometryAttribute::Type> options =
        EncoderOptionsBase<GeometryAttribute::Type>::CreateDefaultOptions();
    options.SetGlobalBool("split_mesh_on_seams", split_mesh_on_seams);

    EncoderBuffer buffer;
    draco::Encoder encoder;
    encoder.Reset(options);
    encoder.SetSpeedOptions(split_mesh_on_seams ? 10 : 0,
                            split_mesh_on_seams ? 10 : 0);
    encoder.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
    DRACO_ASSERT_OK(encoder.EncodeMeshToBuffer(*mesh, &buffer));

    DecoderBuffer dec_buffer;
    dec_buffer.Init(buffer.data(), buffer.size());
    Decoder decoder;
    auto dec_result = decoder.DecodeMeshFromBuffer(&dec_buffer);
    ASSERT_TRUE(dec_result.ok()) << dec_result.status().error_msg();
    std::unique_ptr<Mesh> decoded_mesh = std::move(dec_result).value();

    // The degenerate face must have been excluded on both paths.
    ASSERT_EQ(decoded_mesh->num_faces(), 2);

    // The round trip must preserve the topology contract: re-encoding the
    // decoded mesh and decoding again yields an equivalent mesh.
    EncoderBuffer buffer2;
    draco::Encoder encoder2;
    encoder2.Reset(options);
    encoder2.SetSpeedOptions(split_mesh_on_seams ? 10 : 0,
                             split_mesh_on_seams ? 10 : 0);
    encoder2.SetEncodingMethod(MESH_EDGEBREAKER_ENCODING);
    DRACO_ASSERT_OK(encoder2.EncodeMeshToBuffer(*decoded_mesh, &buffer2));

    DecoderBuffer dec_buffer2;
    dec_buffer2.Init(buffer2.data(), buffer2.size());
    Decoder decoder2;
    auto dec_result2 = decoder2.DecodeMeshFromBuffer(&dec_buffer2);
    ASSERT_TRUE(dec_result2.ok()) << dec_result2.status().error_msg();
    std::unique_ptr<Mesh> decoded_mesh2 = std::move(dec_result2).value();

    ASSERT_EQ(decoded_mesh2->num_faces(), decoded_mesh->num_faces());
    MeshAreEquivalent eq;
    ASSERT_TRUE(eq(*decoded_mesh, *decoded_mesh2));
  }
}

}  // namespace draco
