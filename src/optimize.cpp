#include <optimize.h>

#include <highs/Highs.h>

#include <cassert>
#include <vector>

extern "C" 
{
void optimize_screening_function(double* x_data, double* y_data, int32_t* _n_data, double y_cutoff, double* model, int32_t n_model)
{
    if(n_model == 0)return;
    assert(n_model <= 4);
    assert(x_data[0] == 0.0);
   
    int32_t n_data = *_n_data;

    n_data--; // Use n_data as an index
    // Find all data points relevant for fitting
    while(n_data >= 0 && y_data[n_data] <= y_cutoff)n_data--;
    n_data++; // Use n_data as number of elements

    double max_exp = n_model == 1 ? 0.0 : std::pow(2.0, n_model - 1);
    double x_pow_range = std::pow(x_data[n_data - 1], max_exp) / std::pow(x_data[1], max_exp);
    double rescale = 1.0;
    int32_t first_after_0 = 1;
    // Rescale for fitting so that matrix elements span [1e-2, 1e4], i.e. are not separated by more than 1e6 (Suggestion taken from https://jump.dev/JuMP.jl/stable/tutorials/getting_started/tolerances/#Recommended-values)
    if(x_pow_range > 1e6)
    {
        // rescale largest value x^max_exp to 1e4 and cutoff everyting that is below 1e-2 except 0
        rescale = std::pow(1e4 / std::pow(x_data[n_data - 1], max_exp), 1.0 / max_exp);
        while(first_after_0 < n_data && std::pow(x_data[first_after_0] * rescale, max_exp) < 1e-2)first_after_0++;
        assert(first_after_0 < n_data);
    }  
    else
    {
        // rescale so that x^max_exp range is centered around 1e1 
        double x_high = std::sqrt(x_pow_range) * 1e1;
        rescale = std::pow(x_high / std::pow(x_data[n_data - 1], max_exp), 1.0 / max_exp);
    }
    //std::cout << "Using data points 0 + [" << first_after_0 << ", " << n_data - 1 << "]\n"; 

    int32_t n_fit = n_data - first_after_0 + 1;
    //std::cout << "n_fit = " << n_fit << '\n';
    std::vector<double> x_fit(n_fit, 0.0);
    std::vector<double> y_fit(n_fit);
    std::vector<double> x0(n_data, 1.0); 
    std::vector<double> x2(n_data, 0.0);
    std::vector<double> x4(n_data, 0.0);
    std::vector<double> x8(n_data, 0.0);

    y_fit[0] = y_data[0] * rescale;

    for(int32_t i = 1;i < n_fit;i++)
    {
        x_fit[i] = x_data[first_after_0 + i - 1] * rescale;
        y_fit[i] = y_data[first_after_0 + i - 1] * rescale;
        x2[i] = x_fit[i] * x_fit[i];
        x4[i] = x2[i] * x2[i];
        x8[i] = x4[i] * x4[i];
    }

    //std::cout << "y_fit[1] = " << y_fit[1] << '\n';
    //std::cout << "y_fit[n_fit - 1] = " << y_fit[n_fit - 1] << '\n';

    double y_sum = std::accumulate(y_fit.begin(), y_fit.end(), 0.0);

    HighsModel lin;
    // In theory 4 variables are enough, i.e. parameters a, b, c, d
    // with constraints: a * x8[i] + b * x4[i] + c * x2[i] + d >= y[i]
    // and objective: Min(a * sum(x8) + b * sum(x4) + c * sum(x2) + n_data - sum(y))
    // however this is very instable due to a very large sum(x8) and sum(x4)
    // Instead introduce auxiliary varibles: z[n_data] with
    // z[i] == a * x8[i] + b * x4[i] + c * x2[i] + d
    // z[i] >= y[i]
    // Objective: Min(sum(z) - sum(y))
    // Additionally we allow to split the data into a segment that is supposed to be fitted
    // and a segment which acts as an upper bound
    // ^ 
    // |xx
    // |  xx
    // |    x
    // |     x
    // |----------------->
    // |      x
    // |        
    // |       xxxxxxxxxx
    // |   
    // In this example only the first 7-10 data points should be used for fitting, the rest may be used as an upper bound for the function
    // Using remaining points as an upper bound ensures that the fitted function does not do an unexpected turn towards +inf in the given range
    int32_t n_vars = n_fit + n_model;
    //                 2 * z def      , y <= z <= inf
    int32_t n_constr = 2 * n_fit + n_fit;
    int32_t cfit_off = 2 * n_fit;
    lin.lp_.num_col_ = n_vars;
    lin.lp_.num_row_ = n_constr;
    lin.lp_.sense_ = ObjSense::kMinimize;
    lin.lp_.offset_ = -y_sum;

    lin.lp_.col_cost_.resize(n_vars);
    std::fill(lin.lp_.col_cost_.begin()        , lin.lp_.col_cost_.begin() + n_fit, 1.0);
    std::fill(lin.lp_.col_cost_.begin() + n_fit, lin.lp_.col_cost_.end()          , 0.0);
    
    lin.lp_.col_lower_.resize(n_vars);
    lin.lp_.col_upper_.resize(n_vars);
    std::fill(lin.lp_.col_lower_.begin(), lin.lp_.col_lower_.begin() + n_vars, -kHighsInf);
    std::fill(lin.lp_.col_upper_.begin(), lin.lp_.col_upper_.begin() + n_vars, kHighsInf);

    lin.lp_.col_names_.resize(n_vars);
    for(int32_t i = 0;i < n_fit;i++)lin.lp_.col_names_[i] = std::string("z") + std::to_string(i);
    for(int32_t i = 0;i < n_model;i++)lin.lp_.col_names_[n_fit + i] = {char('d' - i)};
    lin.lp_.row_lower_.resize(n_constr);
    lin.lp_.row_upper_.resize(n_constr);
    lin.lp_.row_names_.resize(n_constr);

    std::fill(lin.lp_.row_lower_.begin(), lin.lp_.row_lower_.begin() + cfit_off, 0.0);
    std::fill(lin.lp_.row_upper_.begin(), lin.lp_.row_upper_.begin() + cfit_off, kHighsInf);
    for(int32_t i = 0;i < n_fit;i++)lin.lp_.row_names_[i] = std::string("z_le") + std::to_string(i);
    for(int32_t i = 0;i < n_fit;i++)lin.lp_.row_names_[n_fit + i] = std::string("z_ge") + std::to_string(i);

    std::copy(y_fit.begin(), y_fit.end(), lin.lp_.row_lower_.begin() + cfit_off);
    std::fill(lin.lp_.row_upper_.begin() + cfit_off, lin.lp_.row_upper_.end(), kHighsInf);
    for(int32_t i = 0;i < n_fit;i++)lin.lp_.row_names_[cfit_off + i] = std::string("fit") + std::to_string(i);
    
    lin.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    // Example with 3 data points

    //                z1 z2 z3  d     c      b      a  
    // zle0 : 0    <= -1        1   x2[0]  x4[0]  x8[0] <= inf 
    // zle1 : 0    <=    -1     1   x2[1]  x4[1]  x8[1] <= inf 
    // zle2 : 0    <=       -1  1   x2[2]  x4[2]  x8[2] <= inf 
    // zge0 : 0    <=  1       -1  -x2[0] -x4[0] -x8[0] <= inf 
    // zge1 : 0    <=     1    -1  -x2[1] -x4[1] -x8[1] <= inf 
    // zge2 : 0    <=        1 -1  -x2[2] -x4[2] -x8[2] <= inf 
    // fit0 : y[0] <=  1                                <= inf
    // fit1 : y[1] <=     1                             <= inf
    // fit2  :y[3] <=        1                          <= inf                            


    lin.lp_.a_matrix_.start_.resize(n_vars + 1);
    //                               z cols            abcd cols
    lin.lp_.a_matrix_.index_.resize((2 + 1) * n_fit + n_model * 2 * n_fit);
    lin.lp_.a_matrix_.value_.resize((2 + 1) * n_fit + n_model * 2 * n_fit);
    int32_t abcd_off = (2 + 1) * n_fit;
    
    // z cols
    for(int32_t i = 0;i < n_fit;i++)
    {
        lin.lp_.a_matrix_.start_[i] = 3 * i;
        lin.lp_.a_matrix_.index_[3 * i + 0] = i;
        lin.lp_.a_matrix_.index_[3 * i + 1] = n_fit + i;
        lin.lp_.a_matrix_.index_[3 * i + 2] = 2 * n_fit + i; 
        lin.lp_.a_matrix_.value_[3 * i + 0] = -1.0;
        lin.lp_.a_matrix_.value_[3 * i + 1] = 1.0;
        lin.lp_.a_matrix_.value_[3 * i + 2] = 1.0; 
    }
    for(int32_t i = 0;i < n_model + 1;i++)lin.lp_.a_matrix_.start_[n_fit + i] = abcd_off + i * 2 * n_fit;
   

    std::vector<double>* xs[4] = {&x0, &x2, &x4, &x8};
    for(int32_t param = 0;param < n_model;param++)
    {
        const auto& x_vec = *xs[param];
        for(int32_t i = 0;i < 2 * n_fit;i++)
        {
            lin.lp_.a_matrix_.index_[abcd_off + param * 2 * n_fit + i] = i;
            lin.lp_.a_matrix_.value_[abcd_off + param * 2 * n_fit + i] = i < n_fit ? x_vec[i] : -x_vec[i - n_fit];
        }
    }
    
    //for(int32_t i = 0;i < n_vars;i++)
    //{
    //    std::cout << "col #" << i << ":\n";
    //    for(int32_t j = lin.lp_.a_matrix_.start_[i]; j < lin.lp_.a_matrix_.start_[i + 1];j++)
    //    {
    //        std::cout << "\t idx = " << lin.lp_.a_matrix_.index_[j] << ", value = " << lin.lp_.a_matrix_.value_[j] << '\n';
    //    }
    //    
    //}
    HighsStatus ret;
    Highs highs_handle;
    ret = highs_handle.setOptionValue("log_to_console", "false");
    ret = highs_handle.setOptionValue("kkt_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    ret = highs_handle.setOptionValue("primal_feasibility_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    ret = highs_handle.setOptionValue("dual_feasibility_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    ret = highs_handle.setOptionValue("primal_residual_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    ret = highs_handle.setOptionValue("dual_residual_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    ret = highs_handle.setOptionValue("optimality_tolerance", 1e-10);
    //assert(ret == HighsStatus::kOk);
    

    int32_t solution_stage = 0;
    double y_max = *std::max_element(y_data, y_data + n_data);
    double y_min = *std::min_element(y_data, y_data + n_data);
    double error_tolerance = (y_max - y_min) * 1e-3;
    double max_deviation;
    while(solution_stage != 2)
    {
        ret = highs_handle.passModel(lin);
        assert(ret == HighsStatus::kOk);
        ret = highs_handle.run();
        assert(ret == HighsStatus::kOk);

        const HighsSolution& sol = highs_handle.getSolution();

        std::fill(model, model + n_model, 0.0);
        std::copy(sol.col_value.begin() + n_fit, sol.col_value.begin() + n_fit + n_model, model);
        
        // rescale model
        for(int32_t i = 0;i < n_model;i++)
        {
            double exp = i == 0 ? 1.0 : -std::pow(2.0, i) + 1; 
            model[i] = model[i] * std::pow((1.0 / rescale), exp);
        }

        max_deviation = 0.0;
        int32_t max_deviation_idx = -1;
        for(int32_t i = 0;i < n_data;i++)
        {
            double f = model[0];
            double xx = x_data[i] * x_data[i];
            for(int32_t j = 1;j < n_model;j++)
            {
                f += model[j] * xx;
                xx *= xx;
            }
            if(max_deviation < y_data[i] - f)
            {
                max_deviation_idx = i;
                max_deviation = y_data[i] - f;
            }
        }

        //if(max_deviation > error_tolerance && solution_stage == 0)
        //{
        //    assert(0 < max_deviation_idx && max_deviation_idx < first_after_0);
        //    // try to adjust y[0] and y[first_after_0] by deviation 
        //    solution_stage = 1;
        //    lin.lp_.row_lower_[cfit_off] = (y_data[0] + max_deviation) * rescale;
        //    lin.lp_.row_lower_[cfit_off + 1] = (y_data[first_after_0] + max_deviation) * rescale;
        //}
        //std::cout << "Stage #" << solution_stage << ": Fitted model of size " << n_model
        //          << ", Using data points 0 + [" << first_after_0 << ", " << n_data - 1 << "]" 
        //          << ", max_deviation = " << max_deviation << '\n';   
        if(max_deviation > error_tolerance && solution_stage == 0)
        {
            assert(0 < max_deviation_idx && max_deviation_idx < first_after_0);
            // error might be caused by y_max missing in fitted data, i.e. y_max in [1, first_after_0[
            // Use smaller model to fit range [0, first_after_0] and set y_data[0] and y_data[first_after_0] to the predicted model values -> the idea is that the smaller model will dominate the range [0, first_after_0] in the larger model
            solution_stage = 1;
            
            if(n_model / 2 > 0)
            {
                int32_t small_n_data = first_after_0 + 1;
                double small_model[n_model / 2];
                optimize_screening_function(x_data, y_data, &small_n_data, y_cutoff, small_model, n_model / 2);              
                
                double f0 = small_model[0];
                double f1 = small_model[0];
                double xx0 = x_data[0] * x_data[0];
                double xx1 = x_data[first_after_0] * x_data[first_after_0];
                for(int32_t j = 1;j < n_model;j++)
                {
                    f0 += small_model[j] * xx0;
                    f1 += small_model[j] * xx1;
                    xx0 *= xx0;
                    xx1 *= xx1;
                }

                //std::cout << "\tAdjusting y[0]            from " << y_data[0] << " to " << f0 << '\n';
                //std::cout << "\tAdjusting y[first_after0] from " << y_data[first_after_0] << " to " << f1 << '\n';

                lin.lp_.row_lower_[cfit_off] = f0 * rescale;
                lin.lp_.row_lower_[cfit_off + 1] = f1 * rescale;
            }
        }
        else solution_stage = 2;
    }



    //const auto& opts = highs_handle.getOptions();
    //std::cout << "opts.kkt_tolerance = " << opts.kkt_tolerance << '\n';
    //std::cout << "opts.primal_feasibility_tolerance = " << opts.primal_feasibility_tolerance << '\n';

    //ret = highs_handle.writeModel("model.mps");
    //assert(ret == HighsStatus::kOk || ret == HighsStatus::kWarning);

    // Last resort:
    // Make model true upper bound by adding deviation as offset
    model[0] += max_deviation;
    // Report back the range for which the fit is valid
    *_n_data = n_data;


    //for(int32_t i = 0;i < n_fit_data;i++) std::cout << "fit constr #" << i << " -- value = " << sol.row_value[cfit_off + i] 
    //                                                << ", corresp. y = " << y_data[i] << ", diff = " << sol.row_value[cfit_off + i] - y_data[i] << '\n';

    //std::cout << "f(x) = " << model[0] << "* x^8 + "
    //                       << model[1] << "* x^4 + "
    //                       << model[2] << "* x^2 + "
    //                       << model[3] << '\n';


}

}
