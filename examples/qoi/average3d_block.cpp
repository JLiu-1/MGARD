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
  void writefile(const char *file, Type *data, size_t num_elements) {
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

  size_t dim0_offset = dimy * dimz;
  size_t dim1_offset = dimz;
  size_t num_block_1 = (dimx - 1) / block_size + 1;
  size_t num_block_2 = (dimy - 1) / block_size + 1;
  size_t num_block_3 = (dimz - 1) / block_size + 1;
  float * data_x_pos = data;
  size_t total_compressed_size = 0;
  for(size_t i=0; i<num_block_1; i++){
      size_t size_1 = (i == num_block_1 - 1) ? dimx - i * block_size : block_size;
      float * data_y_pos = data_x_pos;
      for(size_t j=0; j<num_block_2; j++){
          size_t size_2 = (j == num_block_2 - 1) ? dimy - j * block_size : block_size;
          float * data_z_pos = data_y_pos;
          for(size_t k=0; k<num_block_3; k++){
              size_t size_3 = (k == num_block_3 - 1) ? dimz - k * block_size : block_size;

              
              size_t n_block_elements = size_1 * size_2 * size_3;

              float *const u = static_cast<float *>(std::malloc(n_block_elements * sizeof(*u)));
              float *p = u;
              float * cur_data_pos = data_z_pos;
              for(size_t ii=0; ii<size_1; ii++){
                  for(size_t jj=0; jj<size_2; jj++){
                      for(size_t kk=0; kk<size_3; kk++){
                          *p++  = *cur_data_pos;
                          cur_data_pos ++;
                      }
                      cur_data_pos += dim1_offset - size_3;
                  }
                  cur_data_pos += dim0_offset - size_2 * dim1_offset;
              }



              std::vector<size_t> w3d = {0, 0, 0, size_1, size_2, size_3};
              const mgard::TensorMeshHierarchy<3, float> hierarchy({size_1, size_2, size_3});
              const AverageFunctional3D average({w3d[0], w3d[1], w3d[2]}, {w3d[3], w3d[4], w3d[5]});
              const mgard::TensorQuantityOfInterest<3, float> Q(hierarchy, average);
              const float s = 0;
              float Q_norm = Q.norm(s);
              std::cout <<"Block: "<<i<<" "<<j<<" "<<k<<std::endl;
              std::cout << "request error bound on QoI (average) = " << tol << "\n";
              std::cout << "Q_norm = " << Q_norm << "\n";

              auto average_ori = average(hierarchy, u);
              std::cout << "average using original data: " << average_ori
                        << std::endl;
              const float tolerance = tol / Q_norm;
              const mgard::CompressedDataset<3, float> compressed =
                  mgard::compress(hierarchy, u, s, tolerance);
              std::cout << "after compression\n";
              const mgard::DecompressedDataset<3, float> decompressed =
                  mgard::decompress(compressed);
             
              
                        //<< ", CR = " << (double)(vx * vy * vz * 4) / (compressed.size()) << std::endl;
              total_compressed_size+=compressed.size();
              cur_data_pos = data_z_pos;
              float const *dp = decompressed.data();
              for(size_t ii=0; ii<size_1; ii++){
                  for(size_t jj=0; jj<size_2; jj++){
                      for(size_t kk=0; kk<size_3; kk++){
                          *cur_data_pos = *dp++;
                          cur_data_pos ++;
                      }
                      cur_data_pos += dim1_offset - size_3;
                  }
                  cur_data_pos += dim0_offset - size_2 * dim1_offset;
              }

              auto average_dec = average(hierarchy, decompressed.data());
              std::cout << "average using decompressed data: "
                        << average_dec<< std::endl;
              float err =
                  std::abs(average_dec - average_ori);
              std::cout << "real error of QoI (average) = " << err << "\n";
              if (err < tol)
                std::cout << "********** Successful **********\n";
              else
                std::cout << "********** Fail with error preservation **********\n";
              std::free(u);


              data_z_pos += size_3;



          }
          data_y_pos += dim1_offset * size_2;

      }
      data_x_pos += dim0_offset * size_1;

  }
  
  std::cout<<"Overall compression ratio: "<< (double)total_element_num*sizeof(float)/total_compressed_size << std::endl;

  writefile<float>(outPath, data, total_element_num);
  delete []data;
  return 0;
}
