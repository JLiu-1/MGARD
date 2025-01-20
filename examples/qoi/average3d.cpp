#include <cmath>
#include <cstddef>

#include <array>
#include <iostream>
#include <stdexcept>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/compress.hpp"

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
  size_t vx = 128, vy = 128, vz = 128;
  std::vector<size_t> w3d = {0, 0, 0, vx, vy, vz};
  float tol = 1.0;
  if (argc > 1) {
    vx = (size_t)std::stoi(argv[1]);
    vy = (size_t)std::stoi(argv[2]);
    vz = (size_t)std::stoi(argv[3]);
    w3d[0] = (size_t)std::stoi(argv[4]);
    w3d[1] = (size_t)std::stoi(argv[5]);
    w3d[2] = (size_t)std::stoi(argv[6]);
    w3d[3] = (size_t)std::stoi(argv[7]);
    w3d[4] = (size_t)std::stoi(argv[8]);
    w3d[5] = (size_t)std::stoi(argv[9]);
    tol = std::stof(argv[10]);
  }
  std::cout<<"p1"<<std::endl;
  const mgard::TensorMeshHierarchy<3, float> hierarchy({vx, vy, vz});

  std::cout<<"p2"<<std::endl;
  const AverageFunctional3D average({w3d[0], w3d[1], w3d[2]}, {w3d[3], w3d[4], w3d[5]});
  std::cout<<"p3"<<std::endl;
  const mgard::TensorQuantityOfInterest<3, float> Q(hierarchy, average);
  std::cout<<"p4"<<std::endl;
  const float s = 0;
  float Q_norm = Q.norm(s);
  std::cout << "request error bound on QoI (average) = " << tol << "\n";
  std::cout << "Q_norm = " << Q_norm << "\n";

  float *const u =
      static_cast<float *>(std::malloc(hierarchy.ndof() * sizeof(*u)));
  {
    float *p = u;
    for (std::size_t i = 0; i < vx; ++i) {
      const float x = 2.5 + static_cast<float>(i) / 60;
      for (std::size_t m = 0; m < vy; ++m) {
        const float y = 0.75 + static_cast<float>(m) / 15;
        for (std::size_t k = 0; k < vz; ++k) {
          const float z = 1.5 + static_cast<float>(m) / 30;
          *p++ = 12 + std::sin(2.1 * x - 1.3 * y + 1.6 * z);
	}
      }
    }
  }
  auto average_ori = average(hierarchy, u);
  std::cout << "average using original data: " << average_ori
            << std::endl;
  const float tolerance = tol / Q_norm;
  const mgard::CompressedDataset<3, float> compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  std::cout << "after compression\n";
  const mgard::DecompressedDataset<3, float> decompressed =
      mgard::decompress(compressed);
 
  std::cout << "average using decompressed data: "
            << average(hierarchy, decompressed.data())
            << ", CR = " << vx * vy * vz * 4 / (compressed.size()) << std::endl;
  auto average_dec = average(hierarchy, decompressed.data());
  float err =
      std::abs(average_dec - average_ori);
  std::cout << "real error of QoI (average) = " << err << "\n";
  if (err < tol)
    std::cout << "********** Successful **********\n";
  else
    std::cout << "********** Fail with error preservation **********\n";
  std::free(u);

  return 0;
}
