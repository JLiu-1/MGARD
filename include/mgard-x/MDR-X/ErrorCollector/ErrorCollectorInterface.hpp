#ifndef _MDR_ERROR_COLLECTOR_INTERFACE_HPP
#define _MDR_ERROR_COLLECTOR_INTERFACE_HPP
namespace mgard_x {
namespace MDR {
namespace concepts {

// Error estimator: estimate impact of level errors on the final error
template <typename T> class ErrorCollectorInterface {
public:
  virtual ~ErrorCollectorInterface() = default;

  virtual std::vector<double> collect_level_error(T const *data, SIZE n,
                                                  int num_bitplanes,
                                                  T max_level_error) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif