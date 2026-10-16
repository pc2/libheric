#include <optimize.h>

#include <cmath>
#include <iostream>
#include <vector>

double f(double x, double y_min)
{
    return std::max(
            y_min,
              (-0.0125 * x * x * x * x + 0.4 * std::exp(0.2 * x) * std::cos(20.0 * x) + 7)
            * (1.0 - std::exp(-(x + 1.0)))
    );
}

int main(int argc, char** argv)
{
    double x_range = 15.0;
    double y_min = -20.0;
    int32_t n_data = 101;
    std::vector<double> x_data(n_data), y_data(n_data);
    for(int32_t i = 0;i < n_data;i++)
    {
        x_data[i] = (i * x_range) / (n_data - 1);
        y_data[i] = f(x_data[i], y_min);
    } 

    double model[4];
    int32_t n_model = 3;
    optimize_screening_function(x_data.data(), y_data.data(), &n_data, y_min, model, n_model);
    std::cout << "used " << n_data << " samples for fitting " << '\n';
    std::cout << "h(x) = ";
    for(int32_t i = 1;i < n_model;i++)
    {
        int32_t exp = (int32_t)std::round(std::pow(2.0, n_model - i));
        std::cout << model[n_model - i] << " * x ^ " << exp << " + ";

    }
    std::cout << model[0] << '\n';

    return 0;
}
