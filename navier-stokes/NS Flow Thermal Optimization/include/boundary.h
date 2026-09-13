#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "../include/prm.h"
#include "../include/dealiipackage.h"

namespace TopOpt
{
  using namespace dealii;

  /*NS方程边界值和右侧*/
  template <int dim>
  class NSBoundaryValues : public Function<dim>
  {
  public:
    NSBoundaryValues(const parallel::distributed::Triangulation<dim> &triangulation)
        : Function<dim>(dim + 1), triangulation(triangulation)
    {
    }
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override;

  private:
    const parallel::distributed::Triangulation<dim> &triangulation;
    const FlatManifold<dim> manifold;
    void calculate_normal_vector(const Point<dim> &p,
                                 Tensor<1, dim> &normal_vector) const;
  };

  template <int dim>
  void NSBoundaryValues<dim>::vector_value(const Point<dim> &p,
                                           Vector<double> &values) const
  {
    Tensor<1, dim> normal_vector;
    calculate_normal_vector(p, normal_vector);
    for (unsigned int i = 0; i < dim; i++)
      values[i] = (-1.0) * PRM::inlet_velocity * normal_vector[i];

    values[dim] = 0;
  }

  template <int dim>
  void NSBoundaryValues<dim>::calculate_normal_vector(const Point<dim> &p,
                                                      Tensor<1, dim> &normal_vector) const
  {
    double distance = 1e7;
    double distance_now = 0.0;
    Tensor<1, dim> cell_face_vector;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary() && face->boundary_id() == 1)
        {
          const Point<dim> cell_center = cell->center();
          const Point<dim> face_center = face->center();
          distance_now = (p - face_center).norm();
          if (distance_now < distance)
          {
            distance = distance_now;
            normal_vector = manifold.normal_vector(face, face_center);

            for (unsigned int i = 0; i < dim; i++)
              cell_face_vector[i] = face_center[i] - cell_center[i];

            if (scalar_product(normal_vector, cell_face_vector) < 0.0)
              normal_vector = -1 * normal_vector;
          }
        }
      }
    }
  }

  /*温度方程边界值和右侧*/
  template <int dim>
  class TempBoundaryValues : public Function<dim>
  {
  public:
    TempBoundaryValues()
        : Function<dim>()
    {
    }
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double TempBoundaryValues<dim>::value(const Point<dim> &p,
                                        const unsigned int) const
  {
    (void)p;
    return PRM::inlet_temperature;
  }

  template <int dim>
  class TempRightHandSide : public Function<dim>
  {
  public:
    TempRightHandSide()
        : Function<dim>()
    {
    }
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double TempRightHandSide<dim>::value(const Point<dim> &p,
                                       const unsigned int) const
  {
    (void)p;
    return PRM::Q0;
  }
}

#endif