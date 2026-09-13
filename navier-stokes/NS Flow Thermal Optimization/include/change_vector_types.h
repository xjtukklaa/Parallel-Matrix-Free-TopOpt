#ifndef CHANGE_VECTOR_TYPES_H
#define CHANGE_VECTOR_TYPES_H

#include "../include/dealiipackage.h"

using namespace dealii;
using VecPETSc = LinearAlgebraPETSc::MPI::Vector;
using VecTrilinos = LinearAlgebraTrilinos::MPI::Vector;
using VecDealII = LinearAlgebra::distributed::Vector<double>;

class ChangeVectorTypes
{
public:
    ChangeVectorTypes();

    void vector_copy(VecTrilinos &out, const VecPETSc &in);
    void vector_copy(VecDealII &out, const VecPETSc &in);

    void vector_copy(VecPETSc &out, const VecTrilinos &in);
    void vector_copy(VecDealII &out, const VecTrilinos &in);

    void vector_copy(VecPETSc &out, const VecDealII &in);
    void vector_copy(VecTrilinos &out, const VecDealII &in);
};

#endif