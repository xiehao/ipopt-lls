#include "c_linear_least_squares.hh"

#include <cassert>
#include <iostream>

c_linear_least_squares::c_linear_least_squares(void)
    : m_i_row(0)
{
}

bool c_linear_least_squares::get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                                          Ipopt::Index &nnz_jac_g,
                                          Ipopt::Index &nnz_h_lag,
                                          IndexStyleEnum &index_style)
{
#ifdef DEBUG
    std::cout << "get nlp info..." << std::endl;
#endif // DEBUG
    /*!< number of variables */
    n = m_n_cols;

    /*!< number of constraints */
    m = 0;

    /*!< number of nonzero elements of the Jacobian */
    nnz_jac_g = 0;

    /*!< number of nonzero elements of the Hessian (lower left corner) */
    nnz_h_lag = n * (n + 1) / 2;

    /*!< use the C style indexing (0-based) */
    index_style = Ipopt::TNLP::C_STYLE;

#ifdef DEBUG
    std::cout << "get nlp info done!!" << std::endl;
#endif // DEBUG

    std::cout << "Pre-calculate Hessian matrix..." << std::endl;
    pre_calculate_hessian();
    std::cout << "Pre-calculate Gradient vector..." << std::endl;
    pre_calculate_gradient();

    return true;
}

bool c_linear_least_squares::get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                                             Ipopt::Number *x_u, Ipopt::Index m,
                                             Ipopt::Number *g_l,
                                             Ipopt::Number *g_u)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);
    assert(0 == m);
#ifdef DEBUG
    std::cout << "get bounds info..." << std::endl;
#endif // DEBUG
    /*!< set lower and upper bounds of x */
    for (int i = 0; i < n; ++i)
    {
        x_l[i] = m_x_l;
        x_u[i] = m_x_u;
    }

    /*!< there is no constraints g(x) for this problem */
#ifdef DEBUG
    std::cout << "get bounds info done!!" << std::endl;
#endif // DEBUG

    return true;
}

bool c_linear_least_squares::get_starting_point(Ipopt::Index n, bool init_x,
                                                Ipopt::Number *x, bool init_z,
                                                Ipopt::Number *z_L,
                                                Ipopt::Number *z_U,
                                                Ipopt::Index m,
                                                bool init_lambda,
                                                Ipopt::Number *lambda)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);
    assert(0 == m);

    /*!< ensure that we have the initial values for x,
     * but not for z and lambda */
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

#ifdef DEBUG
    std::cout << "get starting point..." << std::endl;
#endif // DEBUG
    /*!< initialze to the given starting point */
    for (int i = 0; i < n; ++i)
    {
        x[i] = m_x[i];
    }
#ifdef DEBUG
    std::cout << "get starting point done!!" << std::endl;
#endif // DEBUG

    return true;
}

bool c_linear_least_squares::eval_f(Ipopt::Index n, const Ipopt::Number *x,
                                    bool new_x, Ipopt::Number &obj_value)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);

#ifdef DEBUG
    std::cout << "evaluate f..." << std::endl;
#endif // DEBUG
    /*!< assign the objective function:
     * f(x) = sum_{j=0}{(a_{j}^T*x-b_{j})^2} */
    obj_value = 0;
    for (int i = 0; i < m_n_rows; ++i)
    {
        Ipopt::Number temp = 0;
        std::map<int, float>::const_iterator cit = m_csrA[i].begin();
        std::map<int, float>::const_iterator cit_end = m_csrA[i].end();
        for (; cit != cit_end; ++cit)
        {
            temp += x[cit->first] * cit->second;
        }
        temp -= m_b[i];

        obj_value += temp * temp;
    }
#ifdef DEBUG
    std::cout << "evaluate f done!!" << std::endl;
#endif // DEBUG

    return true;
}

bool c_linear_least_squares::eval_grad_f(Ipopt::Index n, const Ipopt::Number *x,
                                         bool new_x, Ipopt::Number *grad_f)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);

#ifdef DEBUG
    std::cout << "evaluate gradient f..." << std::endl;
#endif // DEBUG
    /*!< assign the gradient of objective function f(x):
     * Nebula(f(x))_{j} = 2*sum_{i}{a_{i,j}(a_{i}^T*x-b_{i})} */
#if 0
    for (int j = 0; j < n; ++j)
    {
        grad_f[j] = 0;
        std::map<int, float>::const_iterator cit = m_gradient_order1[j].begin();
        std::map<int, float>::const_iterator cit_end = m_gradient_order1[j].end();
        for (; cit != cit_end; ++cit)
        {
            grad_f[j] += x[cit->first] * cit->second;
        }
        grad_f[j] += m_gradient_order0[j];
    }
#else
    for (int j = 0; j < n; ++j)
    {
        grad_f[j] = 0;
        int index_base = j * (j + 1) / 2;
        for (int i = 0; i <= j; ++i)
        {
            grad_f[j] += x[i] * m_hessian[index_base + i];
        }
        for (int i = j + 1; i < n; ++i)
        {
            index_base = i * (i + 1) / 2;
            grad_f[j] += x[i] * m_hessian[index_base + j];
        }
        grad_f[j] += m_gradient_order0[j];
    }
#endif
#ifdef DEBUG
    std::cout << "evaluate gradient f done!!" << std::endl;
#endif // DEBUG

    return true;
}

bool c_linear_least_squares::eval_g(Ipopt::Index n, const Ipopt::Number *x,
                                    bool new_x, Ipopt::Index m, Ipopt::Number *g)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);
    assert(0 == m);

    /*!< there is no constraints for this problem */

    return true;
}

bool c_linear_least_squares::eval_jac_g(Ipopt::Index n, const Ipopt::Number *x,
                                        bool new_x, Ipopt::Index m,
                                        Ipopt::Index nele_jac,
                                        Ipopt::Index *iRow, Ipopt::Index *jCol,
                                        Ipopt::Number *values)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);
    assert(0 == m);

    /*!< there is no constraints for this problem */

    return true;
}

bool c_linear_least_squares::eval_h(Ipopt::Index n, const Ipopt::Number *x,
                                    bool new_x, Ipopt::Number obj_factor,
                                    Ipopt::Index m, const Ipopt::Number *lambda,
                                    bool new_lambda, Ipopt::Index nele_hess,
                                    Ipopt::Index *iRow, Ipopt::Index *jCol,
                                    Ipopt::Number *values)
{
    /*!< ensure that m and n have correct values, respectively */
    assert(m_n_cols == n);
    assert(0 == m);

    /*!< calculate the Hessian for the combination of objective function f(x)
     * and g(x): H(sigma_{f}+lambda*g(x)) */
    if (!values)
    {
        Ipopt::Index index = 0;
        for (Ipopt::Index i_row = 0; i_row < n; ++i_row)
        {
            for (Ipopt::Index i_col = 0; i_col <= i_row; ++i_col)
            {
                iRow[index] = i_row;
                jCol[index++] = i_col;
            }
        }

        assert(index == nele_hess);
    }
    else
    {
        /*!< add the portion for the first constraint */
#if 0
        int index = 0;
        for (int i_row = 0; i_row < n; ++i_row)
        {
            for (int i_col = 0; i_col <= i_row; ++i_col, ++index)
            {
                float temp = 0;
                for (int k = 0; k < m_n_rows; ++k)
                {
                    if (m_A[k][i_row] && m_A[k][i_col])
                    {
                        temp += m_A[k][i_row] * m_A[k][i_col];
                    }
                }

                values[index] = obj_factor * 2 * temp;
            }
        }
#else /*!< for acceleration */
        size_t size_hessian = m_hessian.size();
        for (size_t i = 0; i < size_hessian; ++i)
        {
            values[i] = obj_factor * m_hessian[i];
        }
#endif
        /*!< there is no constraints g(x), so no additional portions to add */
    }

    return true;
}

void c_linear_least_squares::finalize_solution(Ipopt::SolverReturn status,
                                               Ipopt::Index n,
                                               const Ipopt::Number *x,
                                               const Ipopt::Number *z_L,
                                               const Ipopt::Number *z_U,
                                               Ipopt::Index m,
                                               const Ipopt::Number *g,
                                               const Ipopt::Number *lambda,
                                               Ipopt::Number obj_value,
                                               const Ipopt::IpoptData *ip_data,
                                               Ipopt::IpoptCalculatedQuantities *ip_cq)
{
    /*!< here is where we would store the solution to variables,
     * or write to a file, etc so we could use the solution */

    /*!< For this example, we write the solution to the console */
    std::cout << std::endl << std::endl
              << "Solution of the primal variables, x"
              << std::endl;
    for (int i = 0; i < n; ++i)
    {
        m_x[i] = x[i];
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    std::cout << std::endl << std::endl
              << "Solution of the bound multipliers, z_L and z_U"
              << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
    }
    for (int i = 0; i < n; ++i)
    {
        std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
    }

    std::cout << std::endl << std::endl << "Objective value" << std::endl;
    std::cout << "f(x*) = " << obj_value << std::endl;

    std::cout << std::endl << "Final value of the constraints:" << std::endl;
    for (int i = 0; i < m ; ++i)
    {
        std::cout << "g(" << i << ") = " << g[i] << std::endl;
    }

    return;
}

bool c_linear_least_squares::set_parameters(const int _n_variables,
                                            const int _n_functions,
                                            const int _x_l, const int _x_u)
{
    m_n_cols = _n_variables;
    m_n_rows = _n_functions;

    m_csrA.resize(_n_functions);
    m_cscA.resize(_n_variables);
    m_b.resize(_n_functions);
    m_x.resize(_n_variables, 0);

    m_x_l = _x_l;
    m_x_u = _x_u;

    return true;
}

bool c_linear_least_squares::set_initial_x(const std::vector<float> &_x)
{
    assert(_x.size() == m_x.size());
    m_x.assign(_x.begin(), _x.end());
    return _x.size();
}

bool c_linear_least_squares::add_row(const std::vector<int> &_indices,
                                     const std::vector<float> &_coefficients,
                                     const float _b)
{
    if (_indices.size() != _coefficients.size())
    {
        std::cerr << "Size of indices and coefficients unmatched!!"
                  << std::endl;
        return false;
    }

    if (m_i_row >= m_n_rows)
    {
        std::cerr << "Enough rows, no need to add!!" << std::endl;
        return false;
    }

    /*!< assign the m_i_row-th row of b */
    m_b[m_i_row] = _b;

    /*!< assign the m_i_row-th row of sparse matrix A */
    std::map<int, float> &row = m_csrA[m_i_row];
    size_t n_indices = _indices.size();
    for (size_t i = 0; i < n_indices; ++i)
    {
        row[_indices[i]] = _coefficients[i];
        m_cscA[_indices[i]][m_i_row] = _coefficients[i];
    }
    ++m_i_row;

    return true;
}

bool c_linear_least_squares::pre_calculate_gradient(void)
{
    /*!< assign the gradient of objective function f(x):
     * Nebula(f(x))_{j} = 2*sum_{i}{a_{i,j}(a_{i}^T*x-b_{i})} */
    m_gradient_order1.resize(m_n_cols);
    m_gradient_order0.resize(m_n_cols);
#if 0
    for (int j = 0; j < m_n_cols; ++j)
    {
        float order0 = 0;
        std::map<int, float> &order1 = m_gradient_order1[j];
        for (int i = 0; i < m_n_rows; ++i)
        {
            if (m_csrA[i][j])
            {
                std::map<int, float>::const_iterator cit = m_csrA[i].begin();
                std::map<int, float>::const_iterator cit_end = m_csrA[i].end();
                for (; cit != cit_end; ++cit)
                {
                    order1[cit->first] += cit->second * m_csrA[i][j] * 2;
                }

                order0 -= 2 * m_b[i] * m_csrA[i][j];
            }
        }
        m_gradient_order0[j] = order0;
    }
#else
    for (int j = 0; j < m_n_cols; ++j)
    {
        float order0 = 0;
        std::map<int, float>::const_iterator cit = m_cscA[j].begin();
        std::map<int, float>::const_iterator cit_end = m_cscA[j].end();
        for (; cit != cit_end; ++cit)
        {
            order0 -= cit->second * m_b[cit->first];
        }
        m_gradient_order0[j] = order0 * 2;
    }
#endif
    return true;
}

bool c_linear_least_squares::pre_calculate_hessian(void)
{
    m_hessian.resize(m_n_cols * (m_n_cols + 1) / 2, 0);
#if 1
    int index = 0;
    for (int i_row = 0; i_row < m_n_cols; ++i_row)
    {
        for (int i_col = 0; i_col <= i_row; ++i_col)
        {
            int i_max, i_min;
            if (m_cscA[i_row].size() > m_cscA[i_col].size())
            {
                i_max = i_row;
                i_min = i_col;
            }
            else
            {
                i_max = i_col;
                i_min = i_row;
            }
            const std::map<int, float> &map_max = m_cscA[i_max];
            const std::map<int, float> &map_min = m_cscA[i_min];
            std::map<int, float>::const_iterator cit_min = map_min.begin();
            std::map<int, float>::const_iterator cit_end_min = map_min.end();
            std::map<int, float>::const_iterator cit_end_max = map_max.end();
            std::map<int, float>::const_iterator cit_found_max;
            float temp = 0;
            for (; cit_min != cit_end_min; ++cit_min)
            {
                cit_found_max = map_max.find(cit_min->first);
                if (cit_found_max != cit_end_max)
                {
                    temp += cit_found_max->second * cit_min->second;
                }
            }

            m_hessian[index++] = 2 * temp;
        }
    }
#else
    for (int k = 0; k < m_n_rows; ++k)
    {
        const std::map<int, float> &map_k = m_csrA[k];

        std::map<int, float>::const_iterator cit_row = map_k.begin();
        std::map<int, float>::const_iterator cit_beg = map_k.begin();
        std::map<int, float>::const_iterator cit_end = map_k.end();
        for (; cit_row != cit_end; ++cit_row)
        {
            std::map<int, float>::const_iterator cit_col = cit_beg;
            for (; cit_col != cit_row; ++cit_col)
            {
                int i_row = cit_row->first;
                int i_col = cit_col->first;
                int index = (i_row + 1) * i_row / 2 + i_col;
                m_hessian[index] += 2 * cit_row->second * cit_col->second;
            }

            int i = cit_col->first;
            int v = cit_col->second;
            int index = (i + 1) * i / 2 + i;
            m_hessian[index] += 2 * v * v;
        }
    }
#endif
    return true;
}

#include <coin/IpIpoptApplication.hpp>

int c_linear_least_squares::demo(void)
{
    /*!< solve Ax = b */
    /*!< a simple example:
     * 1) x = 5;
     * 2) x = 10;
     * and the optimal result should be x = 7.5 */

    int n_variables = 1;
    int n_functions = 2;
    std::vector<float> vector_x(n_variables, 50);
    std::vector<float> b(n_functions);
    b[0] = 5, b[1] = 10;

    c_linear_least_squares *lls = new c_linear_least_squares();
    lls->set_parameters(n_variables, n_functions, 0, 100);
    lls->set_initial_x(vector_x);
    for (int i = 0; i < n_functions; ++i)
    {
        std::vector<int> indices(n_variables, 0);
        std::vector<float> coefficients(n_variables, 1);
        lls->add_row(indices, coefficients, b[i]);
    }

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->Options()->SetStringValue("linear_solver", "mumps");

    Ipopt::ApplicationReturnStatus status = app->Initialize();
    if (Ipopt::Solve_Succeeded != status)
    {
        std::cerr << std::endl << std::endl
                  << "*** Error during initialization!" << std::endl;
        return status;
    }

    status = app->OptimizeTNLP(Ipopt::SmartPtr<Ipopt::TNLP>(lls));

    if (Ipopt::Solve_Succeeded == status)
    {
        std::cout << std::endl << std::endl
                  << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl
                  << "*** The problem FAILED!!" << std::endl;
    }

    return 0;
}

int c_linear_least_squares::demo2(void)
{
    /*!< solve Ax = b */
    /*!< a simple example:
     * 1) x + y = 5;
     * 2) x + 2y = 10;
     * 3) 3x + y = 15;
     * and the optimal result should be x = ? */

    int n_variables = 2;
    int n_functions = 3;
    std::vector<float> vector_x(n_variables, 50);
    std::vector<float> b(n_functions);
    b[0] = 5, b[1] = 10, b[2] = 15;

    c_linear_least_squares *lls = new c_linear_least_squares();
    lls->set_parameters(n_variables, n_functions, 0, 100);
    lls->set_initial_x(vector_x);

    std::vector<int> indices(n_variables, 0);
    indices[0] = 0, indices[1] = 1;
    std::vector<float> coefficients(n_variables, 1);
    coefficients[0] = 1, coefficients[1] = 1;
    lls->add_row(indices, coefficients, b[0]);
    coefficients[0] = 1, coefficients[1] = 2;
    lls->add_row(indices, coefficients, b[1]);
    coefficients[0] = 3, coefficients[1] = 1;
    lls->add_row(indices, coefficients, b[2]);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->Options()->SetStringValue("linear_solver", "mumps");

    Ipopt::ApplicationReturnStatus status = app->Initialize();
    if (Ipopt::Solve_Succeeded != status)
    {
        std::cerr << std::endl << std::endl
                  << "*** Error during initialization!" << std::endl;
        return status;
    }

    status = app->OptimizeTNLP(Ipopt::SmartPtr<Ipopt::TNLP>(lls));

    if (Ipopt::Solve_Succeeded == status)
    {
        std::cout << std::endl << std::endl
                  << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl
                  << "*** The problem FAILED!!" << std::endl;
    }

    return 0;
}

int c_linear_least_squares::demo3(void)
{
    /*!< solve Ax = b */
    /*!< a simple example:
     * 1) x + y = 5; * n_repeated
     * 2) x + 2y = 10; * n_repeated
     * 3) 3x + y = 15; * n_repeated
     * and the optimal result should be x = ? */

    int n_repeated = 3000;
    int n_variables = 2;
    int n_functions = 3 * n_repeated;
    std::vector<float> vector_x(n_variables, 50);
    std::vector<float> b(n_functions);
    b[0] = 5, b[1] = 10, b[2] = 15;

    c_linear_least_squares *lls = new c_linear_least_squares();
    lls->set_parameters(n_variables, n_functions, 0, 100);
    lls->set_initial_x(vector_x);

    std::vector<int> indices(n_variables, 0);
    indices[0] = 0, indices[1] = 1;
    std::vector<float> coefficients(n_variables, 1);
    coefficients[0] = 1, coefficients[1] = 1;
    for (int i = 0; i < n_repeated; ++i)
        lls->add_row(indices, coefficients, b[0]);
    coefficients[0] = 1, coefficients[1] = 2;
    for (int i = 0; i < n_repeated; ++i)
        lls->add_row(indices, coefficients, b[1]);
    coefficients[0] = 3, coefficients[1] = 1;
    for (int i = 0; i < n_repeated; ++i)
        lls->add_row(indices, coefficients, b[2]);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->Options()->SetStringValue("linear_solver", "mumps");

    Ipopt::ApplicationReturnStatus status = app->Initialize();
    if (Ipopt::Solve_Succeeded != status)
    {
        std::cerr << std::endl << std::endl
                  << "*** Error during initialization!" << std::endl;
        return status;
    }

    status = app->OptimizeTNLP(Ipopt::SmartPtr<Ipopt::TNLP>(lls));

    if (Ipopt::Solve_Succeeded == status)
    {
        std::cout << std::endl << std::endl
                  << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl
                  << "*** The problem FAILED!!" << std::endl;
    }

    return 0;
}

#include <ctime>

int c_linear_least_squares::demo4(void)
{
    /*!< solve Ax = b */
    /*!< a simple example:
     * 1) x + y = 5; * n_repeated
     * 2) x + 2y = 10; * n_repeated
     * 3) 3x + y = 15; * n_repeated
     * and the optimal result should be x = ? */

    int n_variables = 1000;
    int n_functions = 15000;

    c_linear_least_squares *lls = new c_linear_least_squares();
    lls->set_parameters(n_variables, n_functions, 0, 100);

    /*!< generate a random (n_functions * n_variables) sparse matrix */
    std::cout << "Setting A and b...." << std::endl;
#if 0
    std::vector<std::map<int, float> > &A = lls->get_csrA();
    std::vector<float> &b = lls->get_b();
    srand(static_cast<unsigned int>(time(0)));
    for (int i_row = 0; i_row < n_functions; ++i_row)
    {
//        int n = rand() % n_variables;
        int n = 6;
        for (int i_col = 0; i_col < n; ++i_col)
        {
            int index = rand() % n_variables;
            int value = rand() % 100;
            A[i_row][index] = value;
        }
        b[i_row] = rand() % 1000;
    }
#else
    srand(static_cast<unsigned int>(time(0)));
    for (int i_row = 0; i_row < n_functions; ++i_row)
    {
//        int n = rand() % n_variables;
        int n = 6;
        std::vector<int> indices(n);
        std::vector<float> weights(n);
        for (int i_col = 0; i_col < n; ++i_col)
        {
            int index = rand() % n_variables;
            int weight = rand() % 100;
            indices[i_col] = index;
            weights[i_col] = weight;
        }
        float b = rand() % 1000;

        lls->add_row(indices, weights, b);
    }
#endif

    std::cout << "Setting other parameters..." << std::endl;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->Options()->SetStringValue("linear_solver", "mumps");

    Ipopt::ApplicationReturnStatus status = app->Initialize();
    if (Ipopt::Solve_Succeeded != status)
    {
        std::cerr << std::endl << std::endl
                  << "*** Error during initialization!" << std::endl;
        return status;
    }

    std::cout << "Start solving..." << std::endl;
    status = app->OptimizeTNLP(Ipopt::SmartPtr<Ipopt::TNLP>(lls));

    if (Ipopt::Solve_Succeeded == status)
    {
        std::cout << std::endl << std::endl
                  << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl
                  << "*** The problem FAILED!!" << std::endl;
    }

    return 0;
}
