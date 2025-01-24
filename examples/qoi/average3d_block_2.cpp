#include <cmath>
#include <cstddef>

#include <array>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/compress.hpp"


template<typename Type>
    void readfile(const char *file, const size_t num, Type *data) {
        std::ifstream fin(file, std::ios::binary);
        if (!fin) {
            std::cout << " Error, Couldn't find the file: " << file << "\n";
            exit(0);
        }
        fin.seekg(0, std::ios::end);
        const size_t num_elements = fin.tellg() / sizeof(Type);
        assert(num_elements == num && "File size is not equals to the input setting");
        fin.seekg(0, std::ios::beg);
        fin.read(reinterpret_cast<char *>(data), num_elements * sizeof(Type));
        fin.close();
    }

  template<typename Type>
  std::unique_ptr<Type[]> readfile(const char *file, size_t &num) {
      std::ifstream fin(file, std::ios::binary);
      if (!fin) {
          std::cout << " Error, Couldn't find the file: " << file << std::endl;
          exit(0);
      }
      fin.seekg(0, std::ios::end);
      const size_t num_elements = fin.tellg() / sizeof(Type);
      fin.seekg(0, std::ios::beg);
//        auto data = QoZ::compat::make_unique<Type[]>(num_elements);
      auto data = std::make_unique<Type[]>(num_elements);
      fin.read(reinterpret_cast<char *>(&data[0]), num_elements * sizeof(Type));
      fin.close();
      num = num_elements;
      return data;
  }

  template<typename Type>
  void writefile(const char *file, const Type *data, size_t num_elements) {
      std::ofstream fout(file, std::ios::binary);
      fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
      fout.close();
  }


class AverageFunctional3D {
public:
  AverageFunctional3D(const std::array<std::size_t, 3> lower_left,
                    const std::array<std::size_t, 3> upper_right)
      : lower_left(lower_left), upper_right(upper_right) {
    for (std::size_t i = 0; i < 3; ++i) {
      //      std::cout << upper_right.at(i) << ", " << lower_left.at(i) <<
      //      "\n";
      if (upper_right.at(i) <= lower_left.at(i)) {
        throw std::invalid_argument("invalid region");
      }
    }
  }

  float operator()(const mgard::TensorMeshHierarchy<3, float> &hierarchy,
                   float const *const u) const {
    const std::array<std::size_t, 3> shape = hierarchy.shapes.back();
    const std::size_t n = shape.at(0);
    const std::size_t m = shape.at(1);
    const std::size_t p = shape.at(2);
    //    std::cout << p << ", " << m << ", " << n << "\n";
    if (upper_right.at(0) > n || upper_right.at(1) > m || upper_right.at(2) > p) {
      throw std::invalid_argument("region isn't contained in domain");
    }
    float total = 0;
    std::size_t count = 0;
    for (std::size_t i = lower_left.at(0); i < upper_right.at(0); ++i) {
      for (std::size_t j = lower_left.at(1); j < upper_right.at(1); ++j) {
        for (std::size_t k = lower_left.at(2); k < upper_right.at(2); ++k) {

          total += u[m * p * i + p * j + k];
          ++count;
	}
      }
    }
    return total / count;
  }

private:
  std::array<std::size_t, 3> lower_left;
  std::array<std::size_t, 3> upper_right;
};

int main(int argc, char **argv) {
  //size_t vx = 128, vy = 128, vz = 128;
  
  float tol = 1.0;
  char *inPath = nullptr;
  char *outPath = nullptr;
  size_t dimx, dimy, dimz;
  size_t block_size;
  if (argc == 8) {
    inPath = argv[1];
    dimx = atoi(argv[2]);
    dimy = atoi(argv[3]);
    dimz = atoi(argv[4]);
    block_size = atoi(argv[5]);
    tol = atof(argv[6]);
    outPath = argv[7];
  }
  else{
    std::cout<<"Usage: average3d_block inputfile dimx dimy dimz block_size qoi_tolerance outputfile"<<std::endl;
    std::cout<<"Wrong arguments"<<std::endl;
    exit(1);
  }
  auto total_element_num = dimx*dimy*dimz;
  float * data = new float[total_element_num];
  readfile<float>(inPath, total_element_num, data);
  //float * decData = new float[total_element_num];



  std::vector<size_t> w3d = {0, 0, 0, block_size, block_size, block_size};
  const mgard::TensorMeshHierarchy<3, float> hierarchy_block({block_size, block_size, block_size});
  const mgard::TensorMeshHierarchy<3, float> hierarchy({dimx, dimy, dimz});
  const AverageFunctional3D average({w3d[0], w3d[1], w3d[2]}, {w3d[3], w3d[4], w3d[5]});
  const mgard::TensorQuantityOfInterest<3, float> Q(hierarchy_block, average);
  const float s = 0;
  float Q_norm = Q.norm(s);
  std::cout << "request error bound on QoI (average) = " << tol << "\n";
  std::cout << "Q_norm = " << Q_norm << "\n";

  //auto average_ori = average(hierarchy, u);
  //std::cout << "average using original data: " << average_ori
  //          << std::endl;
  const float tolerance = tol / Q_norm;
  const mgard::CompressedDataset<3, float> compressed =
      mgard::compress(hierarchy, data, s, tolerance);
  //std::cout << "after compression\n";
  const mgard::DecompressedDataset<3, float> decompressed =
      mgard::decompress(compressed);
 
  
            //<< ", CR = " << (double)(vx * vy * vz * 4) / (compressed.size()) << std::endl;
  auto total_compressed_size = compressed.size();

  float const *dp = decompressed.data();
  
  std::cout<<"Overall compression ratio: "<< (double)total_element_num*sizeof(float)/total_compressed_size << std::endl;

  writefile<float>(outPath, dp, total_element_num);
  delete []data;
  return 0;
}
