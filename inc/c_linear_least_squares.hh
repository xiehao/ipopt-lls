#ifndef C_LINEAR_LEAST_SQUARES_HH
#define C_LINEAR_LEAST_SQUARES_HH

#define HAVE_CSTDDEF

#include <vector>
#include <map>
#include <coin/IpTNLP.hpp>

class c_linear_least_squares : public Ipopt::TNLP
{
public:
    /*!< default constructor */
    c_linear_least_squares(void);

    /*!< default destructor (virtual) */
    virtual ~c_linear_least_squares(void) { }

    /*!< Method to return some info about the nlp */
    virtual bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                              Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                              IndexStyleEnum &index_style);

    /*!< Method to return the bounds for this problem */
    virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                                 Ipopt::Number *x_u, Ipopt::Index m,
                                 Ipopt::Number *g_l, Ipopt::Number *g_u);

    /*!< Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Ipopt::Index n, bool init_x,
                                    Ipopt::Number *x, bool init_z,
                                    Ipopt::Number *z_L, Ipopt::Number *z_U,
                                    Ipopt::Index m, bool init_lambda,
                                    Ipopt::Number *lambda);

    /*!< Method to return the objective function value */
    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                        Ipopt::Number &obj_value);

    /*!< Method to return the gradient of the objective function */
    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x,
                             bool new_x, Ipopt::Number *grad_f);

    /*!< Method to return the constraint residuals */
    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                        Ipopt::Index m, Ipopt::Number *g);

    /*!< Method to return:
     * 1) the structure of the Jacobian (if "values" is NULL);
     * 2) the values of the Jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number *x,
                            bool new_x, Ipopt::Index m, Ipopt::Index nele_jac,
                            Ipopt::Index *iRow, Ipopt::Index *jCol,
                            Ipopt::Number *values);

    /*!< Method to return:
     * 1) the structure of the Hessian of the Lagrangian (if "values" is NULL);
     * 2) the values of the Hessian of the lagrangian (if "values" is not NULL)
     */
    virtual bool eval_h(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                        Ipopt::Number obj_factor, Ipopt::Index m,
                        const Ipopt::Number *lambda, bool new_lambda,
                        Ipopt::Index nele_hess, Ipopt::Index *iRow,
                        Ipopt::Index *jCol, Ipopt::Number *values);

    /*!< This method is called when the algorithm is complete so the TNLP can
     * store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                                   const Ipopt::Number *x,
                                   const Ipopt::Number *z_L,
                                   const Ipopt::Number *z_U, Ipopt::Index m,
                                   const Ipopt::Number *g,
                                   const Ipopt::Number *lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData *ip_data,
                                   Ipopt::IpoptCalculatedQuantities *ip_cq);

    /*!< set scale of problem, that is the size of matrix A */
    bool set_parameters(const int _n_variables, const int _n_functions,
                        const int _x_l, const int _x_u);

    /*!< set the initial values for x */
    bool set_initial_x(const std::vector<float> &_x);

    /*!< add one row iteratively */
    bool add_row(const std::vector<int> &_indices,
                 const std::vector<float> &_coefficients, const float _b);

    const std::vector<std::map<int, float> > &get_csrA(void) const
    {
        return m_csrA;
    }

    std::vector<std::map<int, float> > &get_csrA(void)
    {
        return m_csrA;
    }

    const std::vector<float> &get_b(void) const { return m_b; }
    std::vector<float> &get_b(void) { return m_b; }

    const std::vector<float> &get_x(void) const { return m_x; }

    /*!< simply demonstrate usage of this class */
    static int demo(void);
    static int demo2(void);
    static int demo3(void);
    static int demo4(void);

private:
    c_linear_least_squares(const c_linear_least_squares &);
    c_linear_least_squares &operator =(const c_linear_least_squares &);

    bool pre_calculate_gradient(void);
    bool pre_calculate_hessian(void);

    /*!< Ax = b */
    std::vector<std::map<int, float> > m_csrA; /*!< sparse coefficient A (csr) */
    std::vector<std::map<int, float> > m_cscA; /*!< sparse coefficient A (csc) */
    std::vector<float> m_b; /*!< b */
    std::vector<float> m_x; /*!< unknown variables x */

    /*!< size of matrix A */
    int m_n_rows;
    int m_n_cols;

    int m_i_row; /*!< current row index */

    /*!< lower and upper bound of x */
    int m_x_l;
    int m_x_u;

    /*!< Gradient vector (coefficients are constant to be pre-calculated) */
    std::vector<std::map<int, float> > m_gradient_order1;
    std::vector<float> m_gradient_order0;
    /*!< Hessian matrix (constant matrix to be pre-calculated) */
    std::vector<float> m_hessian;
};

#endif // C_LINEAR_LEAST_SQUARES_HH
