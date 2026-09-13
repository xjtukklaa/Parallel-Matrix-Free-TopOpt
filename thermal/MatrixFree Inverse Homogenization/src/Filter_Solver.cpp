#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief 设置热求解器的滤波系统。
 * 
 * 该函数初始化并配置热求解器中滤波系统所需的滤波矩阵和相关数据结构。它执行以下步骤：
 * 
 * 1. 清除现有的滤波矩阵。
 * 2. 分配滤波有限元（FE）和多重网格（MG）自由度（DoFs）。
 * 3. 清除并设置悬挂节点的约束。
 * 4. 配置 MatrixFree 对象的附加数据，包括并行方案和映射更新标志。
 * 5. 使用映射、DoF 处理程序、约束、求积和附加数据初始化 MatrixFree 对象。
 * 6. 使用 MatrixFree 对象初始化滤波矩阵并设置滤波半径。
 * 7. 初始化滤波右手边（RHS）和滤波解的自由度向量。
 */
void Heat_Solver::setup_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_filter_system");
    filter_matrix.clear();
    dof_handler_filter.distribute_dofs(fe_filter);
    dof_handler_filter.distribute_mg_dofs();
    // 
    constraints_filter.clear();
    constraints_filter.reinit(dof_handler_filter.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler_filter));
    DoFTools::make_hanging_node_constraints(dof_handler_filter,constraints_filter);
    constraints_filter.close();
    // 
    typename MatrixFree<dimension, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dimension, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    std::shared_ptr<MatrixFree<dimension, double>> filter_mf_storage(new MatrixFree<dimension, double>());
    filter_mf_storage->reinit(mapping_filter,
                              dof_handler_filter,
                              constraints_filter,
                              QGauss<1>(fe_filter.degree + 1),
                              additional_data);
    filter_matrix.initialize(filter_mf_storage);
    // 
    filter_matrix.Rmin_Pow2 = (double)(Rmin * Rmin);
    filter_matrix.compute_diagonal();
    // 
    filter_matrix.initialize_dof_vector(filter_rhs);
    filter_matrix.initialize_dof_vector(filter_solution);
    // 
    filter_matrix.initialize_dof_vector(Simp_Rho_Continuous_Filter);
}

/**
 * @brief 设置热求解器的多重网格滤波系统。
 *
 * 该函数初始化并配置用于热求解器中滤波系统的多重网格矩阵和约束自由度。
 * 它清除任何现有的元素和约束，调整多重网格矩阵的大小，并初始化约束自由度。
 *
 * 该函数遍历三角剖分的每个层次，设置 MatrixFree 对象所需的约束和附加数据。
 * 然后，它使用适当的设置初始化每个层次的多重网格矩阵。
 *
 * @note 该函数使用 TimerOutput::Scope 来测量设置过程所花费的时间。
 */
void Heat_Solver::setup_multigrid_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_filter_system");
    // 
    mg_matrices_filter.clear_elements();
    mg_constrained_dofs_filter.clear();    
    //
    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices_filter.resize(0, nlevels - 1);
    mg_constrained_dofs_filter.initialize(dof_handler_filter);
    // 
    for (unsigned int level = 0; level < nlevels; level++)
    {    
        AffineConstraints<float> level_constraints(dof_handler_filter.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler_filter, level));;
        level_constraints.close();
        typename MatrixFree<dimension, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dimension, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dimension, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dimension, float>>();
        mg_mf_storage_level->reinit(mapping_filter,
                                    dof_handler_filter,
                                    level_constraints,
                                    QGauss<1>(fe_filter.degree + 1),
                                    additional_data);
        mg_matrices_filter[level].initialize(mg_mf_storage_level,
                                            mg_constrained_dofs_filter,
                                            level);
        mg_matrices_filter[level].Rmin_Pow2 = (float)(Rmin * Rmin);
        mg_matrices_filter[level].compute_diagonal();
    }
}

/**
 * @brief 组装热求解器中滤波操作的右手边 (RHS) 向量。
 *
 * 该函数通过遍历所有单元批次，评估有限元 (FE) 值，并将它们集成形成全局 RHS 向量，从而构建滤波操作的 RHS 向量。
 *
 * @param input 包含用于组装滤波 RHS 的值的输入向量。
 *
 * 该函数执行以下步骤：
 * 1. 将滤波 RHS 向量初始化为零。
 * 2. 更新输入向量中的幽灵值。
 * 3. 初始化滤波矩阵的 FE 评估对象。
 * 4. 遍历所有单元批次：
 *    a. 为当前单元重新初始化 FE 评估对象。
 *    b. 从输入向量中读取自由度 (DOF) 值。
 *    c. 为每个求积点将 DOF 值提交给 FE 评估对象。
 *    d. 集成提交的值。
 *    e. 将局部贡献分配给全局滤波 RHS 向量。
 * 5. 压缩滤波 RHS 向量以完成组装。
 */
void Heat_Solver::assemble_filter_rhs(LinearAlgebra::distributed::Vector<double> input)
{
    TimerOutput::Scope t(computing_timer, "assemble_filter_rhs");
    filter_rhs = 0;    
    QGauss<dimension> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dimension> fe_values_filter(fe_filter,
                                         quadrature_formula_filter,
                                         update_values | 
                                         update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices_filter(1);
    std::vector<double> rho_values_filter(1);
    //
    auto cell_begin_filter = dof_handler_filter.begin_active();
    auto cell_end_filter = dof_handler_filter.end();
    auto base_begin_filter = dof_handler_simp.begin_active();
    //
    for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter, ++base_begin_filter)
    {
        if (cell_begin_filter->is_locally_owned() && cell_begin_filter->level() == base_begin_filter->level())
        {
            cell_rhs = 0;
            fe_values_filter.reinit(cell_begin_filter);
            base_begin_filter->get_dof_indices(local_rho_dof_indices_filter);
            rho_values_filter[0] = input[local_rho_dof_indices_filter[0]];
            for (unsigned int q_point : fe_values_filter.quadrature_point_indices())
            {
                for (unsigned int i : fe_values_filter.dof_indices())
                {
                    cell_rhs(i) += (fe_values_filter.shape_value(i, q_point) *
                                    rho_values_filter[0] *
                                    fe_values_filter.JxW(q_point));
                }
            }
            cell_begin_filter->get_dof_indices(local_dof_indices_filter);
            constraints_filter.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices_filter,
                                                          filter_rhs);
        }
    }
    filter_rhs.compress(VectorOperation::add);
    filter_rhs.update_ghost_values();
}

/**
 * @brief 使用多重网格方法求解滤波问题。
 *
 * 该函数使用多重网格方法设置并求解滤波问题的线性系统。
 * 它初始化多重网格传输、平滑器、粗网格求解器和边界矩阵，
 * 然后使用这些组件通过 BiCGStab 求解器求解系统。
 *
 * 该函数执行以下步骤：
 * 1. 初始化多重网格传输算子。
 * 2. 为多重网格层次结构的每个层次设置平滑器。
 * 3. 初始化粗网格求解器。
 * 4. 构建多重网格矩阵和接口矩阵。
 * 5. 使用矩阵、粗网格求解器、传输算子和平滑器设置多重网格求解器。
 * 6. 使用带有多重网格预处理器的 BiCGStab 求解器求解线性系统。
 * 7. 分配解并更新幽灵值。
 * 8. 输出求解器迭代次数。
 */
void Heat_Solver::solve_filter_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_filter_chebyshev");
    MGTransferMatrixFree<dimension, float> mg_transfer_filter(mg_constrained_dofs_filter);
    mg_transfer_filter.build(dof_handler_filter);
    using SmootherType = PreconditionChebyshev<LeveLFilterMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices_filter[0].m();
        }
        mg_matrices_filter[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices_filter[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_filter, smoother_data);

    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices_filter);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLFilterMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices_filter[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer_filter, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dimension, LeveLVectorType, MGTransferMatrixFree<dimension, float>> preconditioner(dof_handler_filter, mg, mg_transfer_filter);
    // 
    filter_solution = 0;
    SolverControl solver_control_filter(dof_handler_filter.n_dofs(),1e-15,false,false);
    SolverFGMRES<VectorType> solver_filter(solver_control_filter);
    solver_filter.solve(filter_matrix,
                        filter_solution,
                        filter_rhs,
                        preconditioner);
    constraints_filter.distribute(filter_solution);
    filter_solution.update_ghost_values();
    // pcout<<"                             "<< "Filter  Solver Last Steps : "<< solver_control_filter.last_step() << std::endl
    //      <<"                             "<< "Filter  Solver Last Value : "<< solver_control_filter.last_value() << std::endl;
}

/**
 * @brief 生成基于滤波解的平均向量。
 *
 * 该函数计算每个单元的滤波解的平均值，并将其存储在输出向量中。
 * 它遍历 dof_handler_filter 和 dof_handler_simp 中的所有活动单元，
 * 计算每个单元的滤波解的平均值，并将其分配给输出向量中的相应条目。
 *
 * @param output 引用，用于存储计算出的平均值的向量。
 */
void Heat_Solver::generate_average_vector(LinearAlgebra::distributed::Vector<double> &output)
{
    TimerOutput::Scope t(computing_timer, "generate_average_vector");
    //
    auto cell_begin_aver = dof_handler_filter.begin_active();
    auto cell_end_aver = dof_handler_filter.end();
    auto rho_begin_aver = dof_handler_simp.begin_active();
    //
    double average_i;
    std::vector<types::global_dof_index> local_dof_indices_aver(fe_filter.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices_aver(fe_simp.n_dofs_per_cell());
    for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
    {
        if (cell_begin_aver->is_locally_owned() && cell_begin_aver->level() == rho_begin_aver->level())
        {
            average_i = 0;
            cell_begin_aver->get_dof_indices(local_dof_indices_aver);
            rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);
            for (unsigned int i = 0; i < fe_filter.n_dofs_per_cell(); i++)
            {
                average_i += filter_solution[local_dof_indices_aver[i]] / (double)fe_filter.n_dofs_per_cell();
            }
            output[local_rho_dof_indices_aver[0]] = average_i;
        }
    }
    output.compress(VectorOperation::insert);
}

double Nonlinear_Diffusions_Function(double Gradient_Norm)
{
    return 1./pow(1. + Gradient_Norm * Gradient_Norm, 2.);
}

void Heat_Solver::generate_solution_laplace(LinearAlgebra::distributed::Vector<double> &input)
{
    TimerOutput::Scope t(computing_timer, "generate_solution_laplace");
    input = 0.;
    // 
    QGauss<dimension> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dimension> fe_values_filter(mapping_filter,
                                         fe_filter,
                                         quadrature_formula_filter,
                                         update_values | update_gradients |
                                         update_quadrature_points | update_JxW_values | 
                                         update_hessians);
    // 
    auto cell_begin_aver = dof_handler_filter.begin_active();
    auto cell_end_aver = dof_handler_filter.end();
    auto rho_begin_aver = dof_handler_simp.begin_active();
    //
    std::vector<double> laplace_q_point(fe_filter.n_dofs_per_cell());
    std::vector<Tensor<1, dimension,double>> gradient_q_point(fe_filter.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices_aver(fe_simp.n_dofs_per_cell());
    // 
    for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
    {
        if (cell_begin_aver->is_locally_owned() && cell_begin_aver->level() == rho_begin_aver->level())
        {
            rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);

            fe_values_filter.reinit(cell_begin_aver);
            fe_values_filter.get_function_laplacians(filter_solution, laplace_q_point);
            fe_values_filter.get_function_gradients(filter_solution, gradient_q_point);

            for (unsigned int q_point : fe_values_filter.quadrature_point_indices())
            {
                input[local_rho_dof_indices_aver[0]] += Nonlinear_Diffusions_Function(gradient_q_point[q_point].norm()) * 
                                                        laplace_q_point[q_point] *
                                                        fe_values_filter.JxW(q_point) *
                                                        neta;
            }
        }
    }
    input.compress(VectorOperation::add);
}