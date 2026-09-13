#include "../include/Deal_II_MMA.h"
Deal_II_MMA_Solver::Deal_II_MMA_Solver()
{
    Primal_Dual_Epsi = 1e-9;
    Xmax_Xmin_Epsi = 1e-5;
    Rho_Epsi = 5e-5;
    Design_Change_Move = 0.5;
    Alpha_Beta = 0.1;
    Asym_Init = 0.5;
    Asym_Decrease = 0.7;
    Asym_Increase = 1.2;
    Constraint_Convex_Approximation = false;
    RobustAsymptotesType = false;
}
/////////////////////////////////////////////////////////////////////////////////////
////////// 外部接口
/////////////////////////////////////////////////////////////////////////////////////
void Deal_II_MMA_Solver::Deal_II_MMA_SetAsymptotes(double init, double decrease, double increase)
{
    Asym_Init = init;
    Asym_Increase = increase;
    Asym_Decrease = decrease;
}

void Deal_II_MMA_Solver::Deal_II_MMA_SetRobustAsymptotesType(bool robust_type)
{
    RobustAsymptotesType = robust_type;
}

void Deal_II_MMA_Solver::Deal_II_MMA_SetConstraintConvexApproximationType(bool constraint_type)
{
    Constraint_Convex_Approximation = constraint_type;
}

void Deal_II_MMA_Solver::Deal_II_MMA_Init(unsigned int Constraints,
                                          double A_i,
                                          double C_i,
                                          MMAVectorType MMA_Design_Variables)
{
    // 参数初始化
    MMA_Dofs_IndexSet = MMA_Design_Variables.locally_owned_elements();
    //
    Number_Of_Constraints = Constraints;
    // 子问题构建参数
    SubProblem_A.reinit(Constraints);
    SubProblem_A = A_i;
    SubProblem_B.reinit(Constraints);
    SubProblem_C.reinit(Constraints);
    SubProblem_C = C_i;
    // 子问题变量
    SubProblem_Y.reinit(Constraints);
    // 对偶变量和松弛变量
    DualProblem_Lambda.reinit(Constraints);
    DualProblem_Mu.reinit(Constraints);
    // 松弛变量
    Slack_Variables.reinit(2 * Constraints);
    // 移动上下限制Low,Upp,Alpha,Beta
    Moving_Low_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Upp_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Alpha_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Beta_Boundary.reinit(MMA_Design_Variables, true);
    // MMA构建子问题对应的梯度信息
    Object_Function_Positive.reinit(MMA_Design_Variables, true);
    Object_Function_Negative.reinit(MMA_Design_Variables, true);
    Constraint_Function_Positive.resize(Constraints);
    Constraint_Function_Negative.resize(Constraints);
    for (unsigned int i = 0; i < Constraints; i++)
    {
        Constraint_Function_Positive[i].reinit(MMA_Design_Variables, true);
        Constraint_Function_Negative[i].reinit(MMA_Design_Variables, true);
    }
    // MMA储存的前两次的设计变量
    Old_Design_Variables.resize(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        Old_Design_Variables[i].reinit(MMA_Design_Variables, true);
    }
    // 梯度和海斯矩阵
    Object_Function_Gradient.reinit(Constraints);
    Object_Function_Hess.reinit(Constraints,Constraints);
}

void Deal_II_MMA_Solver::Deal_II_MMA_Reinit(MMAVectorType MMA_Design_Variables)
{
    // 参数初始化
    MMA_Dofs_IndexSet = MMA_Design_Variables.locally_owned_elements();
    // 移动上下限制Low,Upp,Alpha,Beta
    Moving_Low_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Upp_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Alpha_Boundary.reinit(MMA_Design_Variables, true);
    Moving_Beta_Boundary.reinit(MMA_Design_Variables, true);
    // MMA构建子问题对应的梯度信息
    Object_Function_Positive.reinit(MMA_Design_Variables, true);
    Object_Function_Negative.reinit(MMA_Design_Variables, true);
    Constraint_Function_Positive.resize(Number_Of_Constraints);
    Constraint_Function_Negative.resize(Number_Of_Constraints);
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Constraint_Function_Positive[i].reinit(MMA_Design_Variables, true);
        Constraint_Function_Negative[i].reinit(MMA_Design_Variables, true);
    }
    // MMA储存的前两次的设计变量
    Old_Design_Variables.resize(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        Old_Design_Variables[i].reinit(MMA_Design_Variables, true);
    }
}

void Deal_II_MMA_Solver::Deal_II_MMA_Dual_Problem_Solve(MMAVectorType &MMA_Design_Variables,
                                                        MMAVectorType MMA_Object_Function_Diff_Values,
                                                        Vector<double>  MMA_Constraints_Function_Values,
                                                        std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
                                                        MMAVectorType MMA_Xmax,
                                                        MMAVectorType MMA_Xmin)
{
    Generate_SubConvexProblem(MMA_Design_Variables,
                              MMA_Object_Function_Diff_Values,
                              MMA_Constraints_Function_Values,
                              MMA_Constraints_Function_Diff_Values,
                              MMA_Xmax,
                              MMA_Xmin);
    Old_Design_Variables[1] = Old_Design_Variables[0];
    Old_Design_Variables[0] = MMA_Design_Variables;
    Interior_Point_Solve(MMA_Design_Variables);
}

double Deal_II_MMA_Solver::Deal_II_MMA_KKT_L2_Norm(MMAVectorType MMA_Design_Variables,
                                                   MMAVectorType MMA_Object_Function_Diff_Values,
                                                   Vector<double>  MMA_Constraints_Function_Values,
                                                   std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
                                                   MMAVectorType MMA_Xmax,
                                                   MMAVectorType MMA_Xmin)
{
    double KKT_L2_Norm = 0;
    double Residual = 0;
    double Mu_Max = 0, Mu_Min = 0;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        Residual = MMA_Object_Function_Diff_Values[i];
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Residual += DualProblem_Lambda[j] * MMA_Constraints_Function_Diff_Values[j][i];
        }
        if (MMA_Design_Variables[i] < MMA_Xmin[i] + 1e-5 && Residual > 0)
        {
            Mu_Min = Residual;
        }
        if (MMA_Design_Variables[i] > MMA_Xmax[i] - 1e-5 && Residual < 0)
        {
            Mu_Max = -Residual;
        }
        Residual += -Mu_Min + Mu_Max;
        KKT_L2_Norm += pow(Residual, 2.);
        KKT_L2_Norm += pow(Mu_Min * (MMA_Design_Variables[i] - MMA_Xmin[i]), 2.);
        KKT_L2_Norm += pow(Mu_Max * (MMA_Xmax[i] - MMA_Design_Variables[i]), 2.);
    }
    KKT_L2_Norm = Utilities::MPI::sum(KKT_L2_Norm,MMA_Design_Variables.get_mpi_communicator());
    // 
    Residual = 0;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Residual += DualProblem_Lambda[i] * (SubProblem_A[i] * SubProblem_Z + SubProblem_Y[i] - MMA_Constraints_Function_Values[i]);
    }
    KKT_L2_Norm += pow(Residual, 2.);
    KKT_L2_Norm = sqrt(KKT_L2_Norm);
    return KKT_L2_Norm;
}
/////////////////////////////////////////////////////////////////////////////////////
////////// 内置函数-不能更改
/////////////////////////////////////////////////////////////////////////////////////
double Deal_II_MMA_Solver::Deal_II_MMA_Get_Change(MMAVectorType MMA_Design_Variables)
{
    double change = 0.;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        change += std::abs(MMA_Design_Variables[i] - Old_Design_Variables[0][i]);
    }
    change = Utilities::MPI::sum(change, MMA_Design_Variables.get_mpi_communicator());
    return change;
}

// MMA子函数构建
void Deal_II_MMA_Solver::Generate_SubConvexProblem(MMAVectorType MMA_Design_Variables,
                                                   MMAVectorType MMA_Object_Function_Diff_Values,
                                                   Vector<double> MMA_Constraints_Function_Values,
                                                   std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
                                                   MMAVectorType MMA_Xmax,
                                                   MMAVectorType MMA_Xmin)
{
    // MMA判断当前的循环次数
    // 如果循环小于三次则不考虑之前迭代的影响
    // 超过三代就需要考虑之前迭代的影响
    // 计算Low和Upp上下界
    if (Loop_Iter < 3)
    {
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            Moving_Low_Boundary[i] = MMA_Design_Variables[i] - Asym_Init * (MMA_Xmax[i] - MMA_Xmin[i]);
            Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + Asym_Init * (MMA_Xmax[i] - MMA_Xmin[i]);
        }
    }
    else
    {
        double Loop_Factor = 0;
        double Asym_Factor = 0;
        double Xmax_Xmin_Epsi_Temp = 0;
        double Xmax_Temp = 0;
        double Xmin_Temp = 0;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            // (x-x_old1)*(x_old1-x_old2)判断迭代序列越来越收敛还是发散
            Loop_Factor = (MMA_Design_Variables[i] - Old_Design_Variables[0][i]) *
                          (Old_Design_Variables[0][i] - Old_Design_Variables[1][i]);
            // 如果迭代序列收敛(迭代变化方向一致)表示上述乘积为正
            // 序列发散表示上述乘积为负
            if (Loop_Factor < 0.)
            {
                Asym_Factor = Asym_Decrease;
            }
            else if (Loop_Factor > 0.)
            {
                Asym_Factor = Asym_Increase;
            }
            else
            {
                Asym_Factor = 1.;
            }
            Moving_Low_Boundary[i] = MMA_Design_Variables[i] - Asym_Factor *
                                                                   (Old_Design_Variables[0][i] - Moving_Low_Boundary[i]);
            Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + Asym_Factor *
                                                                   (Moving_Upp_Boundary[i] - Old_Design_Variables[0][i]);
            Xmax_Xmin_Epsi_Temp = std::max(Xmax_Xmin_Epsi, MMA_Xmax[i] - MMA_Xmin[i]);
            if (RobustAsymptotesType)
            {
                Moving_Low_Boundary[i] = std::max<double>(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 100. * Xmax_Xmin_Epsi_Temp);
                Moving_Low_Boundary[i] = std::min<double>(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 1.e-4 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::max<double>(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 1.e-4 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::min<double>(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 100. * Xmax_Xmin_Epsi_Temp);

                Xmax_Temp = MMA_Xmax[i] + 1.e-5;
                Xmin_Temp = MMA_Xmin[i] - 1.e-5;
                if (MMA_Design_Variables[i] < Xmin_Temp)
                {
                    Moving_Low_Boundary[i] = MMA_Design_Variables[i] - (Xmax_Temp - MMA_Design_Variables[i]) / 0.9;
                    Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + (Xmax_Temp - MMA_Design_Variables[i]) / 0.9;
                }
                if (MMA_Design_Variables[i] > Xmax_Temp)
                {
                    Moving_Low_Boundary[i] = MMA_Design_Variables[i] - (MMA_Design_Variables[i] - Xmin_Temp) / 0.9;
                    Moving_Upp_Boundary[i] = MMA_Design_Variables[i] + (MMA_Design_Variables[i] - Xmin_Temp) / 0.9;
                }                
            }else
            {
                Moving_Low_Boundary[i] = std::max<double>(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 10. * Xmax_Xmin_Epsi_Temp);
                Moving_Low_Boundary[i] = std::min<double>(Moving_Low_Boundary[i], MMA_Design_Variables[i] - 1.e-2 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::max<double>(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 1.e-2 * Xmax_Xmin_Epsi_Temp);
                Moving_Upp_Boundary[i] = std::min<double>(Moving_Upp_Boundary[i], MMA_Design_Variables[i] + 10. * Xmax_Xmin_Epsi_Temp);
            }
        }
    }
	// 计算Beta,Alpha上下界
    double Object_Function_Diff_Value_Positive_Temp = 0;
    double Object_Function_Diff_Value_Negative_Temp = 0;
    double Constraint_Function_Diff_Value_Positive_Temp = 0;
    double Constraint_Function_Diff_Value_Negative_Temp = 0;
    // 这个参数是为了使得子问题的导数不要过于偏离原始梯度
    double Convex_Change_Part_Temp = 0;
    double Xmax_Xmin_Epsi_Inv = 0;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        // Alpha
        Moving_Alpha_Boundary[i] = std::max<double>(MMA_Xmin[i], Moving_Low_Boundary[i] + Alpha_Beta * (MMA_Design_Variables[i] - Moving_Low_Boundary[i]));
        Moving_Alpha_Boundary[i] = std::max<double>(Moving_Alpha_Boundary[i], MMA_Design_Variables[i] - Design_Change_Move * (MMA_Xmax[i] - MMA_Xmin[i]));
        Moving_Alpha_Boundary[i] = std::min<double>(Moving_Alpha_Boundary[i], MMA_Xmax[i]);
        // Beta
        Moving_Beta_Boundary[i] = std::min<double>(MMA_Xmax[i], Moving_Upp_Boundary[i] - Alpha_Beta * (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]));
        Moving_Beta_Boundary[i] = std::min<double>(Moving_Beta_Boundary[i], MMA_Design_Variables[i] + Design_Change_Move * (MMA_Xmax[i] - MMA_Xmin[i]));
        Moving_Beta_Boundary[i] = std::max<double>(Moving_Beta_Boundary[i], MMA_Xmin[i]);
        // 按照符号把目标函数和约束函数的导数分开
        Xmax_Xmin_Epsi_Inv = 1. / std::max(Xmax_Xmin_Epsi, MMA_Xmax[i] - MMA_Xmin[i]);
        // Object Function
        Object_Function_Diff_Value_Positive_Temp = std::max<double>(0., MMA_Object_Function_Diff_Values[i]);
        Object_Function_Diff_Value_Negative_Temp = std::max<double>(0., -1. * MMA_Object_Function_Diff_Values[i]);
        Convex_Change_Part_Temp = 1e-3 * std::abs(MMA_Object_Function_Diff_Values[i]) + Rho_Epsi * Xmax_Xmin_Epsi_Inv;
        // P0_ij
        Object_Function_Positive[i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                      (Object_Function_Diff_Value_Positive_Temp + Convex_Change_Part_Temp);
        // Q0_ij
        Object_Function_Negative[i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                      (Object_Function_Diff_Value_Negative_Temp + Convex_Change_Part_Temp);
        // Constraint Function
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Constraint_Function_Diff_Value_Positive_Temp = std::max<double>(0., MMA_Constraints_Function_Diff_Values[j][i]);
            Constraint_Function_Diff_Value_Negative_Temp = std::max<double>(0., -1. * MMA_Constraints_Function_Diff_Values[j][i]);
            Convex_Change_Part_Temp = 1e-3 * std::abs(MMA_Constraints_Function_Diff_Values[j][i]) + Rho_Epsi * Xmax_Xmin_Epsi_Inv;
            if (Constraint_Convex_Approximation)
            {
                // P_ij
                Constraint_Function_Positive[j][i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Positive_Temp + Convex_Change_Part_Temp);
                // Q_ij
                Constraint_Function_Negative[j][i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Negative_Temp + Convex_Change_Part_Temp);             
            }else{
                // P_ij
                Constraint_Function_Positive[j][i] = std::pow(Moving_Upp_Boundary[i] - MMA_Design_Variables[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Positive_Temp);
                // Q_ij
                Constraint_Function_Negative[j][i] = std::pow(MMA_Design_Variables[i] - Moving_Low_Boundary[i], 2.) *
                                                    (Constraint_Function_Diff_Value_Negative_Temp);
            }
        }
    }
    // 计算约束函数的值
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        SubProblem_B[j] = 0.;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            SubProblem_B[j] += Constraint_Function_Positive[j][i] / (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]) +
                               Constraint_Function_Negative[j][i] / (MMA_Design_Variables[i] - Moving_Low_Boundary[i]);
        }
        SubProblem_B[j] = Utilities::MPI::sum(SubProblem_B[j], MMA_Design_Variables.get_mpi_communicator());
    }
    SubProblem_B.operator-=(MMA_Constraints_Function_Values);
}

// 根据对偶条件
// 得到对偶变量Lambda后计算XYZ的变量值
void Deal_II_MMA_Solver::Generate_XYZ_From_Lambda(MMAVectorType &MMA_Design_Variables)
{
    double Lambda_Mutiply_A = 0;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        if (DualProblem_Lambda[i] < 0.)
        {
            DualProblem_Lambda[i] = 0.;
        }
        SubProblem_Y[i] = std::max(0., DualProblem_Lambda[i] - SubProblem_C[i]);
        Lambda_Mutiply_A += DualProblem_Lambda[i] * SubProblem_A[i];
    }
    SubProblem_Z = std::max(0., 10. * (Lambda_Mutiply_A - 1.));

    double Object_Function_Positive_Mutiply_Lambda = 0;
    double Object_Function_Negative_Mutiply_Lambda = 0;
    for (unsigned int i : MMA_Dofs_IndexSet)
    {
        Object_Function_Positive_Mutiply_Lambda = Object_Function_Positive[i];
        Object_Function_Negative_Mutiply_Lambda = Object_Function_Negative[i];
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Object_Function_Positive_Mutiply_Lambda += Constraint_Function_Positive[j][i] * DualProblem_Lambda[j];
            Object_Function_Negative_Mutiply_Lambda += Constraint_Function_Negative[j][i] * DualProblem_Lambda[j];
        }
        MMA_Design_Variables[i] = (std::sqrt(Object_Function_Positive_Mutiply_Lambda) * Moving_Low_Boundary[i] +
                                   std::sqrt(Object_Function_Negative_Mutiply_Lambda) * Moving_Upp_Boundary[i]) /
                                  (std::sqrt(Object_Function_Positive_Mutiply_Lambda) + std::sqrt(Object_Function_Negative_Mutiply_Lambda));
        if (MMA_Design_Variables[i] < Moving_Alpha_Boundary[i])
        {
            MMA_Design_Variables[i] = Moving_Alpha_Boundary[i];
        }
        if (MMA_Design_Variables[i] > Moving_Beta_Boundary[i])
        {
            MMA_Design_Variables[i] = Moving_Beta_Boundary[i];
        }
    }
    #ifdef COMPRESS_ERROR
    #elif
        MMA_Design_Variables.compress(VectorOperation::insert);    
    #endif
}

// 这个函数用来计算原始对偶系统的梯度
void Deal_II_MMA_Solver::Get_Dual_Gradient(MMAVectorType MMA_Design_Variables)
{
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        Object_Function_Gradient[j] = 0.;
        for (unsigned int i : MMA_Dofs_IndexSet)
        {
            Object_Function_Gradient[j] += Constraint_Function_Positive[j][i] / (Moving_Upp_Boundary[i] - MMA_Design_Variables[i]) +
                                           Constraint_Function_Negative[j][i] / (MMA_Design_Variables[i] - Moving_Low_Boundary[i]);
        }
        Object_Function_Gradient[j] = Utilities::MPI::sum(Object_Function_Gradient[j], MMA_Design_Variables.get_mpi_communicator());
        Object_Function_Gradient[j] -= SubProblem_B[j] + SubProblem_A[j] * SubProblem_Z + SubProblem_Y[j];
    }
}

void Deal_II_MMA_Solver::Get_Dual_Hess(MMAVectorType MMA_Design_Variables)
{
    double Object_Function_Positive_Mutiply_Lambda = 0;
    double Object_Function_Negative_Mutiply_Lambda = 0;
    double Object_Function_Diff_2_Values = 0;
    double Zero_Point_Of_Diff_Function = 0;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        for (unsigned int j = 0; j < Number_Of_Constraints; j++)
        {
            Object_Function_Hess[i][j] = 0.;
            for (unsigned int Index_i : MMA_Dofs_IndexSet)
            {
                Object_Function_Positive_Mutiply_Lambda = Object_Function_Positive[Index_i];
                Object_Function_Negative_Mutiply_Lambda = Object_Function_Negative[Index_i];
                for (unsigned int Index_ii = 0; Index_ii < Number_Of_Constraints; Index_ii++)
                {
                    Object_Function_Positive_Mutiply_Lambda += Constraint_Function_Positive[Index_ii][Index_i] * DualProblem_Lambda[Index_ii];
                    Object_Function_Negative_Mutiply_Lambda += Constraint_Function_Negative[Index_ii][Index_i] * DualProblem_Lambda[Index_ii];
                }
                Zero_Point_Of_Diff_Function = (std::sqrt(Object_Function_Positive_Mutiply_Lambda) * Moving_Low_Boundary[Index_i] +
                                               std::sqrt(Object_Function_Negative_Mutiply_Lambda) * Moving_Upp_Boundary[Index_i]) /
                                              (std::sqrt(Object_Function_Positive_Mutiply_Lambda) + std::sqrt(Object_Function_Negative_Mutiply_Lambda));
                Object_Function_Diff_2_Values = -1. / (2. * Object_Function_Positive_Mutiply_Lambda /
                                                           std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 3.) +
                                                       2. * Object_Function_Negative_Mutiply_Lambda /
                                                           std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 3.));
                if (Zero_Point_Of_Diff_Function < Moving_Alpha_Boundary[Index_i])
                {
                    Object_Function_Diff_2_Values = 0.;
                }
                if (Zero_Point_Of_Diff_Function > Moving_Beta_Boundary[Index_i])
                {
                    Object_Function_Diff_2_Values = 0.;
                }
                Object_Function_Hess[i][j] +=
                    (Constraint_Function_Positive[i][Index_i] / std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 2.) -
                     Constraint_Function_Negative[i][Index_i] / std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 2.)) *
                    Object_Function_Diff_2_Values *
                    (Constraint_Function_Positive[j][Index_i] / std::pow(Moving_Upp_Boundary[Index_i] - MMA_Design_Variables[Index_i], 2.) -
                     Constraint_Function_Negative[j][Index_i] / std::pow(MMA_Design_Variables[Index_i] - Moving_Low_Boundary[Index_i], 2.));
            }
            Object_Function_Hess[i][j] = Utilities::MPI::sum(Object_Function_Hess[i][j], MMA_Design_Variables.get_mpi_communicator());
        }
    }
    
    double Lambda_Mutiply_A = 0.;
    for (unsigned int j = 0; j < Number_Of_Constraints; j++)
    {
        if (DualProblem_Lambda[j] < 0.)
        {
            DualProblem_Lambda[j] = 0.;
        }
        Lambda_Mutiply_A += DualProblem_Lambda[j] * SubProblem_A[j];
        if (DualProblem_Lambda[j] > SubProblem_C[j])
        {
            Object_Function_Hess[j][j] += -1.;
        }
        Object_Function_Hess[j][j] += -DualProblem_Mu[j] / DualProblem_Lambda[j];
    }

    if (Lambda_Mutiply_A > 0.)
    {
        for (unsigned int i = 0; i < Number_Of_Constraints; i++)
        {
            for (unsigned int j = 0; j < Number_Of_Constraints; j++)
            {
                Object_Function_Hess[i][j] += -10. * SubProblem_A[i] * SubProblem_A[j];
            }
        }
    }

    double HessCorr = 1e-4 * Object_Function_Hess.trace() / Number_Of_Constraints;
    if (-1. * HessCorr < 1.e-7)
    {
        HessCorr = -1.e-7;
    }
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Object_Function_Hess[i][i] += HessCorr;
    }
}

// 回溯线性搜索得到步长
void Deal_II_MMA_Solver::Dual_Linear_Search()
{
    double Theta = 1.005;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        if (Theta < -1.01 * Slack_Variables[i] / DualProblem_Lambda[i])
        {
            Theta = -1.01 * Slack_Variables[i] / DualProblem_Lambda[i];
        }
        if (Theta < -1.01 * Slack_Variables[i + Number_Of_Constraints] / DualProblem_Mu[i])
        {
            Theta = -1.01 * Slack_Variables[i + Number_Of_Constraints] / DualProblem_Mu[i];
        }
    }
    Theta = 1. / Theta;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        DualProblem_Lambda[i] = DualProblem_Lambda[i] + Theta * Slack_Variables[i];
        DualProblem_Mu[i] = DualProblem_Mu[i] + Theta * Slack_Variables[i + Number_Of_Constraints];
    }
}

void Deal_II_MMA_Solver::Solve_Linear_System()
{
    if (Number_Of_Constraints > 1)
    {
        // 高斯消元求解
        double  Max_Row_Number = 0;
        unsigned int Max_Row_Index = 0;
        double Swap_Temp = 0;
        for (unsigned int col_index = 0; col_index < Number_Of_Constraints; col_index++)
        {
            Max_Row_Number = 0;
            Max_Row_Index = 0;
            // 选列主元，获得列主元的最大值以及对应的索引
            for (unsigned int row_index = col_index; row_index < Number_Of_Constraints; row_index++)
            {
                if (abs(Object_Function_Hess(row_index,col_index)) > Max_Row_Number)
                {
                    Max_Row_Number = abs(Object_Function_Hess(row_index,col_index));
                    Max_Row_Index = row_index;
                }                      
            }
            // 如果最大值主元小于1e-10报错
            Assert(abs(Max_Row_Number) <= 1e-10, ExcZero());
            // 交换当前行和最大主元的行
            if (Max_Row_Index != col_index)
            {
                for (unsigned int row_index = 0; row_index < Number_Of_Constraints; row_index++)
                {
                    // 交换矩阵的行
                    Swap_Temp = Object_Function_Hess(Max_Row_Index,row_index);
                    Object_Function_Hess(Max_Row_Index,row_index) = Object_Function_Hess(col_index,row_index);
                    Object_Function_Hess(col_index,row_index) = Swap_Temp;
                }
                // 交换向量的行
                Swap_Temp = Object_Function_Gradient[Max_Row_Index];
                Object_Function_Gradient[Max_Row_Index] = Object_Function_Gradient[col_index];
                Object_Function_Gradient[col_index] = Swap_Temp;
            }
            // 开始消元
            for (unsigned int row_index = col_index + 1; row_index < Number_Of_Constraints; row_index++)
            {
                // 矩阵消元
                Swap_Temp = -Object_Function_Hess(row_index,col_index)/Object_Function_Hess(col_index,col_index);
                for (unsigned int col_tmp = 0; col_tmp < Number_Of_Constraints; col_tmp++)
                {
                    Object_Function_Hess(row_index,col_tmp) += Object_Function_Hess(col_index,col_tmp) * Swap_Temp;
                }
                // 开始向量消元
                Object_Function_Gradient[row_index] += Object_Function_Gradient[col_index] * Swap_Temp;
            }   
        }
        for (int i = Number_Of_Constraints - 1; i >= 0; i --)
        {
            // 
            if (i == (int)Number_Of_Constraints - 1)
            {
                Slack_Variables[i] = Object_Function_Gradient[i]/Object_Function_Hess(i,i);                
            }
            // 
            Slack_Variables[i] = Object_Function_Gradient[i];
            for (unsigned int j = i + 1; j < Number_Of_Constraints; j++)
            {
                Slack_Variables[i] -= Slack_Variables[j] * Object_Function_Hess(i,j);
            }
            Slack_Variables[i] /= Object_Function_Hess(i,i);
        }
    }
    else if (Number_Of_Constraints > 0)
    {
        Slack_Variables[0] = Object_Function_Gradient[0] / Object_Function_Hess[0][0];
    }
}

double Deal_II_MMA_Solver::Get_Dual_Residual(MMAVectorType MMA_Design_Variables,
                                             double Epsi)
{
    double Residual = 0.;
    double Error = 0.;
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Residual = 0.;
        for (unsigned int j : MMA_Dofs_IndexSet)
        {
            Residual += Constraint_Function_Positive[i][j] / (Moving_Upp_Boundary[j] - MMA_Design_Variables[j]) +
                        Constraint_Function_Negative[i][j] / (MMA_Design_Variables[j] - Moving_Low_Boundary[j]);
        }
        Residual = Utilities::MPI::sum(Residual, MMA_Design_Variables.get_mpi_communicator());
        Residual -= SubProblem_B[i] + SubProblem_A[i] * SubProblem_Z + SubProblem_Y[i] - DualProblem_Mu[i];
        if (std::abs(Residual) > std::abs(Error))
        {
            Error = std::abs(Residual);
        }
    }
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        Residual = 0.;
        Residual += DualProblem_Mu[i] * DualProblem_Lambda[i] - Epsi;
        if (std::abs(Residual) > std::abs(Error))
        {
            Error = std::abs(Residual);
        }
    }
    return Error;
}

void Deal_II_MMA_Solver::Interior_Point_Solve(MMAVectorType &MMA_Design_Variables)
{
    for (unsigned int i = 0; i < Number_Of_Constraints; i++)
    {
        DualProblem_Lambda[i] = SubProblem_C[i] / 2.;
        DualProblem_Mu[i] = 1.;
    }
    double Tolerance = Primal_Dual_Epsi;
    double Epsi_Init = 1.;
    double Error_Init = 1.;
    unsigned int Loop_Couter = 0;
    while (Epsi_Init > Tolerance)
    {
        Loop_Couter = 0;
        while (Error_Init > 0.9 * Epsi_Init && Loop_Couter < 100)
        {
            Loop_Couter++;
            Generate_XYZ_From_Lambda(MMA_Design_Variables);
            Get_Dual_Gradient(MMA_Design_Variables);
            for (unsigned int i = 0; i < Number_Of_Constraints; i++)
            {
                Object_Function_Gradient[i] = -1. * Object_Function_Gradient[i] - Epsi_Init / DualProblem_Lambda[i];
            }
            Get_Dual_Hess(MMA_Design_Variables);
            Solve_Linear_System();
            for (unsigned int i = 0; i < Number_Of_Constraints; i++)
            {
                Slack_Variables[i + Number_Of_Constraints] = -DualProblem_Mu[i] + Epsi_Init / DualProblem_Lambda[i] - Slack_Variables[i] * DualProblem_Mu[i] / DualProblem_Lambda[i];
            }
            Dual_Linear_Search();
            Generate_XYZ_From_Lambda(MMA_Design_Variables);
            Error_Init = Get_Dual_Residual(MMA_Design_Variables, Epsi_Init);
        }
        Epsi_Init = Epsi_Init * 0.1;
    }
}