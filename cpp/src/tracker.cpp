#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using State = std::array<double, 4>;  // x, y, vx, vy
using Covariance = std::array<std::array<double, 4>, 4>;

struct Measurement {
  int step{};
  int target{};
  std::string sensor;
  double x{};
  double y{};
  double sigma{};
};

struct Filter {
  State state{};
  Covariance covariance{};
  bool initialized{false};

  void predict(double dt, double acceleration_noise) {
    if (!initialized) return;
    const double f[4][4] = {{1, 0, dt, 0}, {0, 1, 0, dt},
                            {0, 0, 1, 0}, {0, 0, 0, 1}};
    State next{};
    Covariance next_cov{};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) next[i] += f[i][j] * state[j];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b)
            next_cov[i][j] += f[i][a] * covariance[a][b] * f[j][b];
      }
    }
    const double q = acceleration_noise * acceleration_noise;
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    next_cov[0][0] += q * dt4 / 4; next_cov[0][2] += q * dt3 / 2;
    next_cov[2][0] += q * dt3 / 2; next_cov[2][2] += q * dt2;
    next_cov[1][1] += q * dt4 / 4; next_cov[1][3] += q * dt3 / 2;
    next_cov[3][1] += q * dt3 / 2; next_cov[3][3] += q * dt2;
    state = next;
    covariance = next_cov;
  }

  void update(const Measurement& m) {
    const double r = m.sigma * m.sigma;
    if (!initialized) {
      state = {m.x, m.y, 0.0, 0.0};
      covariance = {};
      covariance[0][0] = r; covariance[1][1] = r;
      covariance[2][2] = 100.0; covariance[3][3] = 100.0;
      initialized = true;
      return;
    }

    const double s00 = covariance[0][0] + r;
    const double s01 = covariance[0][1];
    const double s10 = covariance[1][0];
    const double s11 = covariance[1][1] + r;
    const double det = s00 * s11 - s01 * s10;
    if (det <= 1e-12) throw std::runtime_error("non-positive innovation covariance");
    const double inv[2][2] = {{s11 / det, -s01 / det}, {-s10 / det, s00 / det}};
    const double residual[2] = {m.x - state[0], m.y - state[1]};
    double k[4][2]{};
    for (int i = 0; i < 4; ++i) {
      k[i][0] = covariance[i][0] * inv[0][0] + covariance[i][1] * inv[1][0];
      k[i][1] = covariance[i][0] * inv[0][1] + covariance[i][1] * inv[1][1];
      state[i] += k[i][0] * residual[0] + k[i][1] * residual[1];
    }

    // Joseph form keeps the covariance symmetric and positive semidefinite.
    Covariance a{};
    for (int i = 0; i < 4; ++i) a[i][i] = 1.0;
    for (int i = 0; i < 4; ++i) { a[i][0] -= k[i][0]; a[i][1] -= k[i][1]; }
    Covariance ap{}, updated{};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int n = 0; n < 4; ++n) ap[i][j] += a[i][n] * covariance[n][j];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        for (int n = 0; n < 4; ++n) updated[i][j] += ap[i][n] * a[j][n];
        updated[i][j] += r * (k[i][0] * k[j][0] + k[i][1] * k[j][1]);
      }
    covariance = updated;
  }
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: tracker <measurements.csv> <tracks.csv> <innovations.csv> <dt_seconds>\n";
    return 2;
  }
  try {
    const double dt = std::stod(argv[4]);
    if (dt <= 0) throw std::invalid_argument("dt must be positive");
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("could not open input CSV");
    std::string line;
    std::getline(input, line);  // header
    std::vector<Measurement> measurements;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      const auto f = split_csv(line);
      if (f.size() != 6) throw std::runtime_error("expected 6 measurement columns");
      measurements.push_back({std::stoi(f[0]), std::stoi(f[1]), f[2],
                              std::stod(f[3]), std::stod(f[4]), std::stod(f[5])});
      if (measurements.back().sigma <= 0) throw std::runtime_error("sigma must be positive");
    }
    std::map<int, std::vector<Measurement>> by_step;
    std::map<int, Filter> filters;
    for (const auto& m : measurements) by_step[m.step].push_back(m);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("could not open output CSV");
    std::ofstream innovations(argv[3]);
    if (!innovations) throw std::runtime_error("could not open innovations CSV");
    output << "step,time_s,target_id,x,y,vx,vy\n" << std::setprecision(10);
    innovations << "step,target_id,sensor_id,dx,dy,sxx,syy\n" << std::setprecision(10);
    if (!by_step.empty()) {
      const int first = by_step.begin()->first, last = by_step.rbegin()->first;
      for (int step = first; step <= last; ++step) {
        for (auto& [target, filter] : filters) filter.predict(dt, 1.0);
        auto found = by_step.find(step);
        if (found != by_step.end()) {
          std::stable_sort(found->second.begin(), found->second.end(),
                           [](const Measurement& a, const Measurement& b) {
                             return a.sigma < b.sigma;
                           });
          for (const auto& m : found->second) {
            auto& filter = filters[m.target];
            if (filter.initialized) {
              const double variance = m.sigma * m.sigma;
              innovations << step << ',' << m.target << ',' << m.sensor << ','
                          << m.x - filter.state[0] << ',' << m.y - filter.state[1] << ','
                          << filter.covariance[0][0] + variance << ','
                          << filter.covariance[1][1] + variance << '\n';
            }
            filter.update(m);
          }
        }
        for (const auto& [target, filter] : filters) {
          if (!filter.initialized) continue;
          output << step << ',' << step * dt << ',' << target << ',' << filter.state[0] << ','
                 << filter.state[1] << ',' << filter.state[2] << ',' << filter.state[3] << '\n';
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "tracker: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
