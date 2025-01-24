#include <cmath>
#include <cstddef>

#include <array>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <limits>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/compress.hpp"
#ifdef _OPENMP
#include "omp.h"
#endif

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
    double total = 0;
    std::size_t count = 0;
    for (std::size_t i = lower_left.at(0); i < upper_right.at(0); ++i) {
      for (std::size_t j = lower_left.at(1); j < upper_right.at(1); ++j) {
        for (std::size_t k = lower_left.at(2); k < upper_right.at(2); ++k) {

          total += u[m * p * i + p * j + k];
          ++count;
	}
      }
    }
    return (float)(total / count);
  }

private:
  std::array<std::size_t, 3> lower_left;
  std::array<std::size_t, 3> upper_right;
};

int main(int argc, char **argv) {
  //size_t vx = 128, vy = 128, vz = 128;
#ifdef _OPENMP 
  float tol = 1.0;
  char *inPath = nullptr;
  char *outPath = nullptr;
  size_t dimx, dimy, dimz;
  size_t block_size;
  int num_threads = 4;

  if (argc == 8 or argc == 9) {
    inPath = argv[1];
    dimx = atoi(argv[2]);
    dimy = atoi(argv[3]);
    dimz = atoi(argv[4]);
    block_size = atoi(argv[5]);
    tol = atof(argv[6]);
    outPath = argv[7];
    if(argc == 9)
      num_threads=atoi(argv[8]);
  }
  else{
    std::cout<<"Usage: average3d_block_omp inputfile dimx dimy dimz block_size qoi_tolerance outputfile [optional]num_threads"<<std::endl;
    std::cout<<"Wrong arguments"<<std::endl;
    exit(1);
  }
  omp_set_num_threads(num_threads);
  auto total_element_num = dimx*dimy*dimz;
  float * data = new float[total_element_num];
  readfile<float>(inPath, total_element_num, data);

  
  size_t dim0_offset = dimy * dimz;
  size_t dim1_offset = dimz;
  size_t num_block_1 = (dimx - 1) / block_size + 1;
  size_t num_block_2 = (dimy - 1) / block_size + 1;
  size_t num_block_3 = (dimz - 1) / block_size + 1;
  size_t num_blocks = num_block_1*num_block_2*num_block_3;
  std::cout<<"Number of blocks: "<<num_blocks<<std::endl;
  int nThreads = 1;
  size_t total_compressed_size = 0;
  //std::vector<bool> verifications(num_blocks,false);
  //std::vector<char> start(num_blocks,0);
  //std::vector<char> done(num_blocks,0);
  std::vector<float> qoi_errs(num_blocks,0);
  //verifications.resize(num_blocks);
  //float max_err = 0.0;

  //float * decData = new float[total_element_num];
#pragma omp parallel
  {
    //float * data_x_pos = data;
    #pragma omp single
      {
        nThreads = omp_get_num_threads();
        std::cout<<"nThreads = "<<nThreads<<std::endl;;
      }
    int tid = omp_get_thread_num();

    const mgard::TensorMeshHierarchy<3, float> hierarchy_global({size_1, size_2, size_3});
    const AverageFunctional3D average_global({w3d[0], w3d[1], w3d[2]}, {w3d[3], w3d[4], w3d[5]});
    const mgard::TensorQuantityOfInterest<3, float> Q_global(hierarchy, average);
    const float s_global = 0;
    float Q_norm_global = Q.norm(s_global);

    for(size_t block_id = tid; block_id < num_blocks; block_id += nThreads){
      //start[block_id]=true;
     // try{
        auto temp = block_id;
        size_t block_id_x = temp/(num_block_2*num_block_3);
        temp = temp % (num_block_2*num_block_3);
        size_t block_id_y = temp/(num_block_3);
        size_t block_id_z = temp%num_block_3;
        size_t start_id_x = block_size*block_id_x;
        size_t start_id_y = block_size*block_id_y;
        size_t start_id_z = block_size*block_id_z;
        size_t size_1 = std::min(block_size,dimx-start_id_x);
        size_t size_2 = std::min(block_size,dimy-start_id_y);
        size_t size_3 = std::min(block_size,dimz-start_id_z);
        size_t start_offset = start_id_x*dim0_offset+start_id_y*dim1_offset+start_id_z;
        size_t n_block_elements = size_1 * size_2 * size_3;

        float *const u = static_cast<float *>(std::malloc(n_block_elements * sizeof(*u)));
        float *p = u;
        for(size_t ii=0; ii<size_1; ii++){
            for(size_t jj=0; jj<size_2; jj++){
                for(size_t kk=0; kk<size_3; kk++){
                  auto offset = start_offset+ii*dim0_offset+jj*dim1_offset+kk;
                  *p++  = *(data+offset);
                }
            }
        }

        std::vector<size_t> w3d = {0, 0, 0, size_1, size_2, size_3};
        const mgard::TensorMeshHierarchy<3, float> hierarchy({size_1, size_2, size_3});
        const AverageFunctional3D average({w3d[0], w3d[1], w3d[2]}, {w3d[3], w3d[4], w3d[5]});
        const mgard::TensorQuantityOfInterest<3, float> Q(hierarchy, average);
        const float s = 0;
        float Q_norm = Q.norm(s);
        
        auto average_ori = average(hierarchy, u);
        
        const float tolerance = tol / Q_norm;
        const mgard::CompressedDataset<3, float> compressed =
            mgard::compress(hierarchy, u, s, tolerance);
           
        const mgard::DecompressedDataset<3, float> decompressed =
            mgard::decompress(compressed);
        std::free(u);
        
                  //<< ", CR = " << (double)(vx * vy * vz * 4) / (compressed.size()) << std::endl;
        #pragma omp atomic
        total_compressed_size+=compressed.size();
        float const *dp = decompressed.data();
        for(size_t ii=0; ii<size_1; ii++){
            for(size_t jj=0; jj<size_2; jj++){
                for(size_t kk=0; kk<size_3; kk++){
                    auto offset = start_offset+ii*dim0_offset+jj*dim1_offset+kk;
                    *(data+offset) = *dp++;
                }
            }
        }
   
        auto average_dec = average(hierarchy, decompressed.data());
        
        float err =
            std::abs(average_dec - average_ori);
        
        //verifications[block_id] = (err<=tol);
        qoi_errs[block_id] = err;
        //done[block_id] = 1;
        //#pragma omp critical
        //max_err = std::max(max_err,err);
       // std::cout << "real error of QoI (average) = " << err << "\n";
        //if (err < tol)
          //std::cout << "********** Successful **********\n";
        //else
        //  std::cout << "********** Fail with error preservation **********\n";
     // }
     // catch(const std::exception &exc){
     //   std::cerr <<block_id<<std::endl;
     //   std::cerr << exc.what()<<std::endl;

     // }
      

    }


    
    
  }
  
  std::cout<<"Overall compression ratio: "<< (double)total_element_num*sizeof(float)/total_compressed_size << std::endl;
  bool succeessful = true;
  float max_err;
  for(size_t i=0;i<num_blocks;i++){
    max_err = std::max(max_err,qoi_errs[i]);
   // if(!start[i]){
    //  succeessful = false;
    //  std::cout<<"Unstarted at block "<<i<<std::endl;
   // }
    //if(done[i] != 1){
    //  succeessful = false;
    //  std::cout<<"Unfinished at block "<<i<<std::endl;
    //}
    //else 
    if (qoi_errs[i]>tol){
      succeessful = false;
      std::cout<<"QoI unbounded at block "<<i<<std::endl;
    }
  }
  if(succeessful){
    std::cout<<"QoI error bounded successfully."<<std::endl;
  }
  else{
    std::cout<<"Failed to bound QoI error."<<std::endl;
  }
  std::cout<<"Maximum QoI error: "<<max_err<<std::endl;

  writefile<float>(outPath, data, total_element_num);
  delete []data;
  return 0;
#endif
  std::cout<<"OpenMP not found. Please load OpenMP or use sequential version."<<std::endl;
  return 0;
}
