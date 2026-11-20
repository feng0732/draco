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
#include "draco/core/options.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>

namespace draco {

namespace {

// Converts a double precision value to a string without losing precision
// (unlike std::to_string that prints only 6 decimal digits).
std::string DoubleToString(double val) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", val);
  return std::string(buf);
}

}  // namespace

void Options::MergeAndReplace(const Options& other_options) {
  for (const auto& item : other_options.options_) {
    options_[item.first] = item.second;
  }
}

void Options::SetInt(const std::string& name, int val) {
  options_[name] = std::to_string(val);
}

void Options::SetFloat(const std::string& name, float val) {
  options_[name] = std::to_string(val);
}

void Options::SetDouble(const std::string& name, double val) {
  options_[name] = DoubleToString(val);
}

void Options::SetBool(const std::string& name, bool val) {
  options_[name] = std::to_string(val ? 1 : 0);
}

void Options::SetString(const std::string& name, const std::string& val) {
  options_[name] = val;
}

int Options::GetInt(const std::string& name) const { return GetInt(name, -1); }

int Options::GetInt(const std::string& name, int default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return std::atoi(it->second.c_str());
}

float Options::GetFloat(const std::string& name) const {
  return GetFloat(name, -1);
}

float Options::GetFloat(const std::string& name, float default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return static_cast<float>(std::atof(it->second.c_str()));
}

double Options::GetDouble(const std::string& name) const {
  return GetDouble(name, -1.0);
}

double Options::GetDouble(const std::string& name, double default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return std::strtod(it->second.c_str(), nullptr);
}

template <>
void Options::SetVector<double>(const std::string& name, const double* vec,
                                int num_dims) {
  std::string out;
  for (int i = 0; i < num_dims; ++i) {
    if (i > 0) {
      out += " ";
    }
    out += DoubleToString(vec[i]);
  }
  options_[name] = out;
}

bool Options::GetBool(const std::string& name) const {
  return GetBool(name, false);
}

bool Options::GetBool(const std::string& name, bool default_val) const {
  const int ret = GetInt(name, -1);
  if (ret == -1) {
    return default_val;
  }
  return static_cast<bool>(ret);
}

std::string Options::GetString(const std::string& name) const {
  return GetString(name, "");
}

std::string Options::GetString(const std::string& name,
                               const std::string& default_val) const {
  const auto it = options_.find(name);
  if (it == options_.end()) {
    return default_val;
  }
  return it->second;
}

}  // namespace draco
