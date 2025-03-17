// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/utils/custom_functions/opapi/FFTCommonOpApi.h"

namespace op_api {

    static FFTMixCache fftCache(10);

    FFTMixCache::FFTMixCache(int64_t c) : capacity(c) {}

    FFTCacheValue FFTMixCache::get(FFTCacheKey &cacheKey)
    {
        std::lock_guard<std::mutex> guard(fftMutex);
        auto match_key = [&cacheKey](FFTPair &pair) {return pair.first == cacheKey;};
        auto it = std::find_if(list.begin(), list.end(), match_key);
        if (it != list.end()) {
            FFTPair tmp = *it;
            list.push_back(tmp);
            list.erase(it);
            return tmp.second;
        }

        if (static_cast<int>(list.size()) >= capacity) {
            if (list.front().first.isAsdSip) {
                destoryHandle(list.front().second.handle);
            }
            list.pop_front();
        }

        FFTCacheValue value;
        if (cacheKey.isAsdSip) {
            value.handle = createHandle(cacheKey.fftParam);
        } else {
            value.plan = make_plan(cacheKey.planKey);
        }

        list.push_back(std::make_pair(cacheKey, value));
        return value;
    }

    void FFTMixCache::setCapacity(int64_t maxSize)
    {
        std::lock_guard<std::mutex> guard(fftMutex);
        if (capacity == maxSize) {
            return;
        }

        capacity = maxSize;
        while (static_cast<int>(list.size()) > capacity) {
            if (list.front().first.isAsdSip) {
                destoryHandle(list.front().second.handle);
            }
            list.pop_front();
        }
    }

    int64_t FFTMixCache::getCapacity()
    {
        return capacity;
    }
    
    int64_t FFTMixCache::getSize()
    {
        return list.size();
    }
    
    void FFTMixCache::clear()
    {
        for (auto it = list.begin(); it != list.end(); ++it) {
            FFTPair tmp = *it;
            if (tmp.first.isAsdSip) {
                destoryHandle(tmp.second.handle);
            }
        }
        list.clear();
    }
    
    void setFFTPlanCapacity(int64_t maxSize)
    {
        if (maxSize < 1 || maxSize > 99) {
            return;
        }

        fftCache.setCapacity(maxSize);
    }

    int64_t getFFTPlanCapacity()
    {
        return fftCache.getCapacity();
    }

    int64_t getFFTPlanSize()
    {
        return fftCache.getSize();
    }

    void clearFFTPlanCache()
    {
        fftCache.clear();
    }

    FFTPlanItem get_plan(int64_t prb_size, bool is_forward, PlanMode plan_mode, at::ScalarType scalar_dtype)
    {
        FFTCacheKey cacheKey;
        cacheKey.isAsdSip = false;
        cacheKey.planKey.prb_size = prb_size;
        cacheKey.planKey.is_forward = is_forward;
        cacheKey.planKey.plan_mode = plan_mode;
        cacheKey.planKey.scalar_dtype = scalar_dtype;
        return fftCache.get(cacheKey).plan;
    }

    asdFftHandle getHandle(FFTParam param)
    {
        FFTCacheKey cacheKey;
        cacheKey.isAsdSip = true;
        cacheKey.fftParam = param;
        return fftCache.get(cacheKey).handle;
    }

} // namespace op_api
