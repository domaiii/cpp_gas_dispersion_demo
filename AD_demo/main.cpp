#include "adv_diff.h"
#include <basix/finite-element.h>
#include <cmath>
#include <array>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <petscmat.h>
#include <petscsys.h>
<<<<<<< HEAD
#include <petsctime.h>
#include <petscsystypes.h>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <random>
#include <iostream>
#include <ranges>
=======
#include <vector>
#include <algorithm>
>>>>>>> f78897a (reduced example)

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

struct Params
{
  // Domain
  double len = 50.0;
  double wid = 50.0;

  // Mesh discretization
  std::int64_t nx = 100;
  std::int64_t ny = 100;

  // Gaussian source(s)
  double source_amp = 1.0;
  double source_sigma2 = 0.1;
  int random_locations = 3;
  unsigned long seed = 1;

  // Physics
  double wind_x = 0.0;
  double wind_y = 1.0;
  double D = 0.5;
};

// int parser
static std::int64_t parse_i64(std::string_view s, const char* name)
{
  std::int64_t v{};
  auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (res.ec != std::errc() || res.ptr != s.data() + s.size())
    throw std::runtime_error(std::string("Invalid value for ") + name + ": " + std::string(s));
  return v;
}

// double parser
static double parse_f64(std::string_view s, const char* name)
{
  try
  {
    size_t idx = 0;
    std::string tmp(s);
    double v = std::stod(tmp, &idx);
    if (idx != tmp.size())
      throw std::runtime_error("trailing");
    return v;
  }
  catch (...)
  {
    throw std::runtime_error(std::string("Invalid value for ") + name + ": " + std::string(s));
  }
}

static Params parse_args_strip(int& argc, char* argv[])
{
  Params p;

  auto need_value = [&](int& i, const char* opt) -> std::string_view {
    if (i + 1 >= argc)
      throw std::runtime_error(std::string("Missing value for ") + opt);
    return std::string_view(argv[++i]);
  };

  int out = 1;

  for (int i = 1; i < argc; ++i)
  {
    std::string_view a(argv[i]);

    if (a == "--len")
      p.len = parse_f64(need_value(i, "--len"), "len");
    else if (a == "--wid")
      p.wid = parse_f64(need_value(i, "--wid"), "wid");
    else if (a == "--nx")
      p.nx = parse_i64(need_value(i, "--nx"), "nx");
    else if (a == "--ny")
      p.ny = parse_i64(need_value(i, "--ny"), "ny");
    else if (a == "--windx")
      p.wind_x = parse_f64(need_value(i, "--windx"), "windx");
    else if (a == "--windy")
      p.wind_y = parse_f64(need_value(i, "--windy"), "windy");
    else if (a == "--D")
      p.D = parse_f64(need_value(i, "--D"), "D");
    else if (a == "--amp")
      p.source_amp = parse_f64(need_value(i, "--amp"), "amp");
    else if (a == "--sigma2")
      p.source_sigma2 = parse_f64(need_value(i, "--sigma2"), "sigma2");
    else if (a == "--nloc")
      p.random_locations = static_cast<int>(parse_i64(need_value(i, "--nloc"), "nloc"));
    else if (a == "--seed")
      p.seed = static_cast<unsigned long>(parse_i64(need_value(i, "--seed"), "seed"));
    else if (a.starts_with("--"))
      throw std::runtime_error("Unknown option: " + std::string(a));
    else
      argv[out++] = argv[i]; // pass to PETSc
  }

  argc = out;

  if (p.len <= 0 || p.wid <= 0)
    throw std::runtime_error("len and wid must be > 0");
  if (p.nx <= 0 || p.ny <= 0)
    throw std::runtime_error("nx and ny must be positive integers");
  if (p.D <= 0)
    throw std::runtime_error("D must be > 0");
  if (p.source_sigma2 <= 0)
    throw std::runtime_error("sigma2 must be > 0");
  if (p.random_locations <= 0)
    throw std::runtime_error("nloc must be a positive integer");

  return p;
}

int main(int argc, char* argv[])
{
  Params p;
  try
  {
    p = parse_args_strip(argc, argv);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Argument error: " << e.what() << '\n';
    return 2;
  }

  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  int rank, nrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nrank);

  PetscLogDouble t_mesh = 0.0;
  PetscLogDouble t_assembly_sum = 0.0;
  PetscLogDouble t_solve_sum = 0.0;

<<<<<<< HEAD
  PetscLogDouble t0, t1;
  PetscTime(&t0);

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
=======
    // Rectangle discretization
    auto cell_dicretization = std::array<std::int64_t, 2>{500, 500};

    // Parameters for gaussian source
    const T x0 = 0.2 * len;   // center x
    const T y0 = 0.3 * wid;   // center y
    const T amp = 1.0;        // amplitude
    const T s2 = 0.005;       // sigma^2
>>>>>>> f78897a (reduced example)

  // Domain size
  const T len = static_cast<T>(p.len);
  const T wid = static_cast<T>(p.wid);

  // Rectangle discretization
  auto cell_dicretization = std::array<std::int64_t, 2>{p.nx, p.ny};

  // Wind speeds (constant, uniform)
  const T wind_x = static_cast<T>(p.wind_x);
  const T wind_y = static_cast<T>(p.wind_y);

  // Diffusion constant
  const T D = static_cast<T>(p.D);

  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {len, wid}}},
                                cell_dicretization, mesh::CellType::triangle, part));
  std::shared_ptr<const mesh::Mesh<U>> cmesh = mesh;

  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto basix_elem = std::make_shared<fem::FiniteElement<U>>(element);
  auto elem_V2 = std::make_shared<fem::FiniteElement<U>>(element, std::vector<std::size_t>{2});

  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace<U>(mesh, basix_elem));

  auto V2 = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace<U>(mesh, elem_V2));

  PetscTime(&t1);
  t_mesh = t1 - t0;

  auto D_phys = std::make_shared<fem::Constant<T>>(D);
  auto L_char = std::make_shared<fem::Constant<T>>(len);
  auto f = std::make_shared<fem::Function<T>>(V);
  auto beta = std::make_shared<fem::Function<T>>(V2);

  fem::Form<T> a = fem::create_form<T>(*form_adv_diff_a, {V, V}, {{"beta", beta}},
                                       {{"D_phys", D_phys}, {"L_char", L_char}},
                                       {}, {}, cmesh);

  fem::Form<T> L = fem::create_form<T>(*form_adv_diff_L, {V},
                                       {{"f", f}, {"beta", beta}},
                                       {{"D_phys", D_phys}, {"L_char", L_char}},
                                       {}, {}, cmesh);

  // Wind field
  beta->interpolate(
      [=](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        const std::size_t n = x.extent(1);
        std::vector<T> values(2 * n);
        for (std::size_t j = 0; j < n; ++j)
          values[j] = wind_x;
        for (std::size_t j = 0; j < n; ++j)
          values[n + j] = wind_y;
        return {values, {2, n}};
      });

  // Dirichlet boundary condition (y=0)
  std::vector facets = mesh::locate_entities_boundary(
      *mesh, 1,
      [](auto x)
      {
        using UU = typename decltype(x)::value_type;
        constexpr UU eps = 1.0e-8;
        std::vector<std::int8_t> marker(x.extent(1), false);
        for (std::size_t j = 0; j < x.extent(1); ++j)
        {
          if (std::abs(x(1, j)) < eps)
            marker[j] = true;
        }
        return marker;
      });

  std::vector bdofs = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
  fem::DirichletBC<T> bc(0, bdofs, V);

  auto u = std::make_shared<fem::Function<T>>(V);
  la::petsc::Matrix A(fem::petsc::create_matrix(a), false);
  la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                  L.function_spaces()[0]->dofmap()->index_map_bs());

  MatZeroEntries(A.mat());
  fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES), a, {bc});
  MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V, {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

  la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
  la::petsc::options::set("ksp_type", "gmres");
  la::petsc::options::set("pc_type", "hypre");
  solver.set_from_options();
  solver.set_operator(A.mat());

  la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
  la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);

  std::mt19937 rng;
  std::uniform_real_distribution<double> dist_x(0.2 * p.len, 0.8 * p.len);
  std::uniform_real_distribution<double> dist_y(0.2 * p.wid, 0.8 * p.wid);
  if (rank == 0)
    rng.seed(p.seed);

  const T amp = static_cast<T>(p.source_amp);
  const T s2  = static_cast<T>(p.source_sigma2);

  for (int k = 0; k < p.random_locations; ++k)
  {
    double src_loc_d[2] = {0.0, 0.0};
    if (rank == 0)
    {
      src_loc_d[0] = dist_x(rng);
      src_loc_d[1] = dist_y(rng);
    }
    MPI_Bcast(src_loc_d, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const T x0 = static_cast<T>(src_loc_d[0]);
    const T y0 = static_cast<T>(src_loc_d[1]);

    f->interpolate(
        [=](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          const std::size_t n = x.extent(1);
          std::vector<T> values;
          values.reserve(n);

          for (std::size_t j = 0; j < n; ++j)
          {
            const T dx = x(0, j) - x0;
            const T dy = x(1, j) - y0;
            const T r2 = dx * dx + dy * dy;
            values.push_back(amp * std::exp(-r2 / s2));
          }

          return {values, {n}};
        });

<<<<<<< HEAD
    PetscTime(&t0);
    std::ranges::fill(b.array(), 0);
=======
    std::fill(b.array().begin(), b.array().end(), T(0));
>>>>>>> f78897a (reduced example)
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);
    PetscTime(&t1);
    t_assembly_sum += (t1 - t0);

<<<<<<< HEAD
    PetscTime(&t0);
    solver.solve(_u.vec(), _b.vec());
    PetscTime(&t1);
    t_solve_sum += (t1 - t0);

    u->x()->scatter_fwd();

    std::string fname = "u_" + std::to_string(k) + ".pvd";
    io::VTKFile file(MPI_COMM_WORLD, fname, "w");
    file.write<T>({*u}, 0);

=======
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "gmres");
    la::petsc::options::set("pc_type", "jacobi");
    solver.set_from_options();

    solver.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    solver.solve(_u.vec(), _b.vec());

    // Update ghost values before output
    u->x()->scatter_fwd();

>>>>>>> f78897a (reduced example)
#ifdef HAS_ADIOS2
    std::string bname = "u_" + std::to_string(k) + ".bp";
    io::VTXWriter<U> vtx(MPI_COMM_WORLD, bname, {u}, "bp4");
    vtx.write(0);
#endif
  }

  double t_mesh_max = 0.0, t_assembly_max = 0.0, t_solve_max = 0.0;
  double ta = static_cast<double>(t_assembly_sum);
  double ts = static_cast<double>(t_solve_sum);
  double tm = static_cast<double>(t_mesh);

  MPI_Reduce(&tm, &t_mesh_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ta, &t_assembly_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ts, &t_solve_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::cout << "\n=== Timing summary ===\n";
    std::cout << "MPI processes       : " << nrank << "\n";
    std::cout << "Mesh + spaces setup : " << t_mesh_max << " s\n";
    std::cout << "Assembly (sum)      : " << t_assembly_max << " s\n";
    std::cout << "Solve (sum)         : " << t_solve_max << " s\n";
    std::cout << "Runs (nloc)         : " << p.random_locations << "\n";
    std::cout << "======================\n\n";
  }

  PetscFinalize();
  return 0;
}
