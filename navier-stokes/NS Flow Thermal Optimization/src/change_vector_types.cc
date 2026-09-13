#include "../include/change_vector_types.h"

ChangeVectorTypes::ChangeVectorTypes() {}


void ChangeVectorTypes::vector_copy(VecTrilinos &out, const VecPETSc &in)
{
    std::pair<unsigned int, unsigned int> local_range = in.local_range();
    for (unsigned int i = local_range.first; i < local_range.second; ++i)
    {
        out[i] = in[i];
    }
    out.compress(VectorOperation::insert);
}
void ChangeVectorTypes::vector_copy(VecDealII &out, const VecPETSc &in)
{
    std::pair<unsigned int, unsigned int> local_range = in.local_range();
    for (unsigned int i = local_range.first; i < local_range.second; ++i)
    {
        out[i] = in[i];
    }
    out.compress(VectorOperation::insert);
}


void ChangeVectorTypes::vector_copy(VecPETSc &out, const VecTrilinos &in)
{
    std::pair<unsigned int, unsigned int> local_range = in.local_range();
    for (unsigned int i = local_range.first; i < local_range.second; ++i)
    {
        out[i] = in[i];
    }
    out.compress(VectorOperation::insert);
}
void ChangeVectorTypes::vector_copy(VecDealII &out, const VecTrilinos &in)
{
    LinearAlgebra::ReadWriteVector<double> rwv;
    rwv.reinit(in);
    out.import_elements(rwv, VectorOperation::insert);
}


void ChangeVectorTypes::vector_copy(VecPETSc &out, const VecDealII &in)
{
    std::pair<unsigned int, unsigned int> local_range = out.local_range();
    for (unsigned int i = local_range.first; i < local_range.second; ++i)
    {
        out[i] = in[i];
    }
    out.compress(VectorOperation::insert);
}
void ChangeVectorTypes::vector_copy(VecTrilinos &out, const VecDealII &in)
{
    LinearAlgebra::ReadWriteVector<double> rwv(out.locally_owned_elements());
    rwv.import_elements(in, VectorOperation::insert);
    out.import_elements(rwv, VectorOperation::insert);
}
