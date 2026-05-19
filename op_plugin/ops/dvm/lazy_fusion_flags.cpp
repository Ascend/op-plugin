// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/ops/dvm/lazy_fusion_flags.h"
#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <map>
#include <cstdlib>
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace lazy_fusion {
namespace {
constexpr auto kLogValidFlag =
  "Valid flag format is \"key=value\", flags are separated by spaces(e.g. \"key1=value1 key2=value2\"). bool "
  "flag's value can be implicit, the \"key\" means \"key=true\".";

std::vector<std::string> GetTokens(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start < str.size()) {
    size_t pos = str.find_first_of(delim, start);
    if (pos == std::string::npos) {
      tokens.emplace_back(str.substr(start));
      break;
    }
    if (pos > start) {
      tokens.emplace_back(str.substr(start, pos - start));
    }
    start = pos + delim.size();
  }
  return tokens;
}

std::pair<std::string, std::string> ParseFlag(const std::string &flag) {
  // Format: "key" (bare boolean flag) or "key=value".
  if (flag.empty() || flag.rfind("--", 0) == 0) {
    return std::pair<std::string, std::string>();
  }
  auto j = flag.find('=');
  if (j == std::string::npos) {
    return std::make_pair(flag, "");
  }
  if (j > 0 && flag.find('=', j + 1) == std::string::npos) {
    return std::make_pair(flag.substr(0, j), flag.substr(j + 1));
  }
  return std::pair<std::string, std::string>();
}

std::map<std::string, std::string> ParseFlags(const std::string &flags) {
  std::map<std::string, std::string> flag_map;
  auto tokens = GetTokens(flags, " ");
  for (const auto &token : tokens) {
    auto flag = ParseFlag(token);
    if (!flag.first.empty()) {
      if (!flag_map.insert(flag).second) {
        ASCEND_LOGW("Warning: The flag '%s' is repeated.", flag.first.c_str());
      }
    } else {
      ASCEND_LOGW("Warning: The flag '%s' is invalid.", token.c_str());
    }
  }
  return flag_map;
}

class FlagRegister {
 public:
  explicit FlagRegister(std::map<std::string, std::string> *flag_map) : flag_map_(*flag_map) {}
  ~FlagRegister() = default;

  template <typename T>
  void AddFlag(const std::string &flag_name, T *flag_var, T default_value) const {
    *flag_var = std::move(default_value);
    AddFlag(flag_name, flag_var);
  }

  template <typename T>
  void AddFlag(const std::string &flag_name, T *flag_var) const {
    const auto iter = flag_map_.find(flag_name);
    if (iter != flag_map_.end()) {
      T var;
      bool ret = ParseValue(iter->second, &var);
      if (ret) {
        *flag_var = std::move(var);
      } else {
        if (iter->second.empty()) {
          ASCEND_LOGW("Warning: The flag '%s' is invalid. %s", iter->first.c_str(), kLogValidFlag);
        } else {
          ASCEND_LOGW("Warning: The flag '%s=%s' is invalid. %s", iter->first.c_str(), iter->second.c_str(), kLogValidFlag);
        }
      }
      (void)flag_map_.erase(iter);
    }
  }

 private:
  bool ParseValue(const std::string &s, std::vector<std::string> *result) const {
    *result = GetTokens(s, ",");
    return !result->empty();
  }

  bool ParseValue(const std::string &s, bool *result) const {
    *result = (s.empty() || s == "true" || s == "True" || s == "on" || s == "1");
    return *result || s == "false" || s == "False" || s == "off" || s == "0";
  }

  template <typename T>
  bool ParseValue(const std::string &s, T *result) const {
    if (s.empty()) {
      return false;
    }
    std::istringstream iss(s);
    iss >> (*result);
    return iss.eof();
  }

  template <typename T>
  bool ParseValue(const std::string &s, std::vector<T> *result) const {
    result->clear();
    auto tokens = GetTokens(s, ",");
    if (tokens.empty()) {
      return false;
    }
    for (const auto &tok : tokens) {
      T temp;
      if (!ParseValue(tok, &temp)) {
        result->clear();
        return false;
      }
      result->emplace_back(temp);
    }
    return true;
  }

  std::map<std::string, std::string> &flag_map_;
};

bool ParseLevel(const std::string &s, Level *out) {
  if (s == "O1" || s == "o1") {
    *out = Level::kO1;
    return true;
  }
  if (s == "O2" || s == "o2") {
    *out = Level::kO2;
    return true;
  }
  return false;
}

void RegisterFlags(std::map<std::string, std::string> *flag_map, LazyFusionFlags *flags) {
  FlagRegister reg(flag_map);

  // `level` is internal-debug; default (kO2, enable everything) stays if absent.
  auto level_it = flag_map->find("level");
  if (level_it != flag_map->end()) {
    Level parsed;
    if (ParseLevel(level_it->second, &parsed)) {
      flags->level = parsed;
    } else {
      ASCEND_LOGW("Warning: level='%s' is invalid; valid values are O1, O2.",
                  level_it->second.c_str());
    }
    flag_map->erase(level_it);
  }

  reg.AddFlag("dump_as_text", &flags->dump_as_text);
  reg.AddFlag("dump_dir", &flags->dump_dir);
  reg.AddFlag("synchronize", &flags->synchronize);
  reg.AddFlag("online_tuning", &flags->online_tuning);
  reg.AddFlag("disable_ops", &flags->disable_ops);
  reg.AddFlag("enable_ops", &flags->enable_ops);
  reg.AddFlag("enable_ops_only", &flags->enable_ops_only);
  for (const auto &item : *flag_map) {
    ASCEND_LOGW("Unknown flag: %s", item.first.c_str());
  }
}
}  // namespace

namespace {
bool ParseBoolToken(const std::string &s, bool *out) {
  if (s == "True" || s == "true" || s == "1" || s == "on" || s == "TRUE") {
    *out = true;
    return true;
  }
  if (s == "False" || s == "false" || s == "0" || s == "off" || s == "FALSE") {
    *out = false;
    return true;
  }
  return false;
}
}  // namespace

LazyFusionFlags::LazyFusionFlags() {
  // Master switch + optional flags via TORCH_NPU_LAZY_FUSION.
  //
  // Public usage:
  //   export TORCH_NPU_LAZY_FUSION=True              # enable, default level O2 (all ops)
  //   export TORCH_NPU_LAZY_FUSION=1                 # same as above
  //   export TORCH_NPU_LAZY_FUSION=False             # disable
  //   (unset)                                        # disable
  //
  // Internal debug (must use the level=... key form, bare "O1"/"O2" is rejected):
  //   export TORCH_NPU_LAZY_FUSION="level=O1"        # enable, only the L1 op set
  //   export TORCH_NPU_LAZY_FUSION="level=O2"        # enable, explicit L2 (same as True)
  //   export TORCH_NPU_LAZY_FUSION="True dump_as_text dump_dir=/tmp/dvm"
  //   export TORCH_NPU_LAZY_FUSION="True disable_ops=where,sum"
  //
  // Flag tokens use bare "key" (boolean true) or "key=value". No "--" prefix.
  char *env = std::getenv("TORCH_NPU_LAZY_FUSION");
  if (env == nullptr) {
    enabled = false;
    return;
  }
  std::string str = env;
  size_t i = str.find_first_not_of(" \t");
  if (i == std::string::npos) {
    enabled = false;
    return;
  }
  size_t j = str.find_first_of(" \t", i);
  std::string head = (j == std::string::npos) ? str.substr(i) : str.substr(i, j - i);
  std::string rest;
  if (ParseBoolToken(head, &enabled)) {
    if (!enabled) {
      return;
    }
    rest = (j == std::string::npos) ? "" : str.substr(j + 1);
  } else {
    // No leading bool token — env value is a flag list. Presence alone enables DVM.
    enabled = true;
    rest = str.substr(i);
  }
  std::map<std::string, std::string> flag_map = ParseFlags(rest);
  RegisterFlags(&flag_map, this);
}
}  // namespace lazy_fusion
