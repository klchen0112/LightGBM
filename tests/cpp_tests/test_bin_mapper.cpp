/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/bin.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

using LightGBM::BinMapper;
using LightGBM::BinType;
using LightGBM::MissingType;

const double kNaN = std::numeric_limits<double>::quiet_NaN();

void FindBins(BinMapper* mapper, std::vector<double> values, BinType type, size_t num_zeros = 0) {
  mapper->FindBin(values.data(), static_cast<int>(values.size()), values.size() + num_zeros,
                  255, 1, 1, true, type, true, false, {});
}

void ExpectRoundTrip(const BinMapper& mapper) {
  std::vector<char> buffer(mapper.SizesInByte());
  mapper.CopyTo(buffer.data());
  BinMapper restored(buffer.data());

  ASSERT_EQ(mapper.num_bin(), restored.num_bin());
  EXPECT_EQ(mapper.missing_type(), restored.missing_type());
  EXPECT_EQ(mapper.bin_type(), restored.bin_type());
  EXPECT_EQ(mapper.bin_info_string(), restored.bin_info_string());
  EXPECT_EQ(mapper.is_trivial(), restored.is_trivial());
  EXPECT_EQ(mapper.GetDefaultBin(), restored.GetDefaultBin());
  EXPECT_EQ(mapper.GetMostFreqBin(), restored.GetMostFreqBin());
  EXPECT_DOUBLE_EQ(mapper.sparse_rate(), restored.sparse_rate());
  EXPECT_EQ(mapper.SizesInByte(), restored.SizesInByte());
  for (int bin = 0; bin < mapper.num_bin(); ++bin) {
    if (std::isnan(mapper.BinToValue(bin))) {
      EXPECT_TRUE(std::isnan(restored.BinToValue(bin)));
    } else {
      EXPECT_DOUBLE_EQ(mapper.BinToValue(bin), restored.BinToValue(bin));
    }
  }
  for (double value : {kNaN, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 100.0}) {
    EXPECT_EQ(mapper.ValueToBin(value), restored.ValueToBin(value));
  }
}

struct MissingCategories {
  std::string name;
  std::vector<double> values;
};

class MissingCategoricalBinMapperTest : public ::testing::TestWithParam<MissingCategories> {};

TEST_P(MissingCategoricalBinMapperTest, InitializesMissingBinAndSerializes) {
  BinMapper mapper;
  FindBins(&mapper, GetParam().values, BinType::CategoricalBin);

  // Check the backing category array before indexing it or serializing it.
  ASSERT_EQ("-1", mapper.bin_info_string());
  EXPECT_EQ(1, mapper.num_bin());
  EXPECT_EQ(MissingType::NaN, mapper.missing_type());
  EXPECT_TRUE(mapper.is_trivial());
  EXPECT_EQ(0u, mapper.GetDefaultBin());
  EXPECT_EQ(0u, mapper.GetMostFreqBin());
  EXPECT_DOUBLE_EQ(1.0, mapper.sparse_rate());
  EXPECT_DOUBLE_EQ(-1.0, mapper.BinToValue(0));
  for (double value : {kNaN, -2.0, -1.0, 0.0, 1.0}) {
    EXPECT_EQ(0u, mapper.ValueToBin(value));
  }
  ExpectRoundTrip(mapper);
}

TEST_P(MissingCategoricalBinMapperTest, ReplacesExistingCategories) {
  BinMapper mapper;
  // Fail safely on versions that cannot construct an all-missing category mapper.
  FindBins(&mapper, GetParam().values, BinType::CategoricalBin);
  ASSERT_EQ("-1", mapper.bin_info_string());

  FindBins(&mapper, {1.0, 1.0, 1.0, 1.0, 2.0}, BinType::CategoricalBin);
  ASSERT_FALSE(mapper.is_trivial());
  ASSERT_NE(0u, mapper.GetMostFreqBin());

  FindBins(&mapper, GetParam().values, BinType::CategoricalBin);

  ASSERT_EQ("-1", mapper.bin_info_string());
  EXPECT_EQ(1, mapper.num_bin());
  EXPECT_TRUE(mapper.is_trivial());
  EXPECT_EQ(0u, mapper.ValueToBin(1.0));
  EXPECT_EQ(0u, mapper.ValueToBin(2.0));
  EXPECT_EQ(0u, mapper.GetMostFreqBin());
  ExpectRoundTrip(mapper);
}

INSTANTIATE_TEST_SUITE_P(
    MissingValues, MissingCategoricalBinMapperTest,
    ::testing::Values(
        MissingCategories{"AllNaN", {kNaN, kNaN, kNaN, kNaN}},
        MissingCategories{"AllNegative", {-1.0, -2.0, -3.0, -1.0}},
        MissingCategories{"NaNAndNegative", {kNaN, -1.0, kNaN, -2.0}}),
    [](const ::testing::TestParamInfo<MissingCategories>& info) { return info.param.name; });

TEST(BinMapper, PreservesValidCategoriesIncludingZero) {
  BinMapper mapper;
  // Zero samples are supplied through total_sample_cnt, as in dataset construction.
  FindBins(&mapper, {1.0, 1.0, 1.0, 2.0}, BinType::CategoricalBin, 2);

  ASSERT_EQ(4, mapper.num_bin());
  EXPECT_EQ(MissingType::None, mapper.missing_type());
  EXPECT_FALSE(mapper.is_trivial());
  EXPECT_EQ(mapper.ValueToBin(0.0), mapper.GetDefaultBin());
  for (double value : {0.0, 1.0, 2.0}) {
    EXPECT_NE(0u, mapper.ValueToBin(value));
    EXPECT_DOUBLE_EQ(value, mapper.BinToValue(mapper.ValueToBin(value)));
  }
  EXPECT_EQ(0u, mapper.ValueToBin(kNaN));
  EXPECT_EQ(0u, mapper.ValueToBin(-1.0));
  EXPECT_EQ(0u, mapper.ValueToBin(3.0));
  ExpectRoundTrip(mapper);
}

TEST(BinMapper, TrivialNumericBinsResetMostFrequentBin) {
  for (double value : {3.0, kNaN}) {
    SCOPED_TRACE(value);
    BinMapper mapper;
    FindBins(&mapper, {1.0, 1.0, 1.0, 1.0}, BinType::NumericalBin, 1);
    ASSERT_FALSE(mapper.is_trivial());
    ASSERT_EQ(1u, mapper.GetMostFreqBin());
    EXPECT_NE(mapper.ValueToBin(0.0), mapper.ValueToBin(1.0));
    ExpectRoundTrip(mapper);

    FindBins(&mapper, std::vector<double>(5, value), BinType::NumericalBin);

    EXPECT_TRUE(mapper.is_trivial());
    EXPECT_EQ(0u, mapper.GetDefaultBin());
    EXPECT_EQ(0u, mapper.GetMostFreqBin());
    EXPECT_DOUBLE_EQ(1.0, mapper.sparse_rate());
    EXPECT_EQ(std::isnan(value) ? MissingType::NaN : MissingType::None, mapper.missing_type());
    ExpectRoundTrip(mapper);
  }
}

}  // namespace
