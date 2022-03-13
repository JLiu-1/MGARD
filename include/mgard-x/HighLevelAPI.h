/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// #include "compress_cuda.hpp"
#include "mgard-x/Hierarchy.h"
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"

#ifndef MGARD_X_HIGH_LEVEL_API_H
#define MGARD_X_HIGH_LEVEL_API_H

namespace mgard_x {

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated);

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated);

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated);

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated);

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated);

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated);

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, data_type& dtype, 
                std::vector<mgard_x::SIZE>& shape, Config config,
                bool output_pre_allocated);

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, data_type& dtype, 
                std::vector<mgard_x::SIZE>& shape, bool output_pre_allocated);


template <typename DeviceType> void BeginAutoTuning();

template <typename DeviceType> void EndAutoTuning();

} // namespace mgard_x

#endif