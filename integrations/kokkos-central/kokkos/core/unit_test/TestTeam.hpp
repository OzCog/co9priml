//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <cstdio>
#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType>
struct TestTeamPolicy {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using view_type = Kokkos::View<int **, ExecSpace>;

  view_type m_flags;

  // initialize m_flags first with default view so that the class
  // is fully initialized when *this is used to figure out the length
  // for m_flags
  TestTeamPolicy(const size_t league_size) : m_flags() {
    m_flags = view_type(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "flags"),
    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
        Kokkos::TeamPolicy<ScheduleType, ExecSpace>(
            1, std::is_same_v<ExecSpace, Kokkos::Experimental::OpenMPTarget>
                   ? 32
                   : 1)
            .team_size_max(*this, Kokkos::ParallelReduceTag()),
#else
        Kokkos::TeamPolicy<ScheduleType, ExecSpace>(1, 1).team_size_max(
            *this, Kokkos::ParallelReduceTag()),
#endif
        league_size);
  }

  struct VerifyInitTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    m_flags(member.team_rank(), member.league_rank()) = tid;
    static_assert(
        (std::is_same_v<typename team_member::execution_space, ExecSpace>),
        "TeamMember::execution_space is not the same as "
        "TeamPolicy<>::execution_space");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const VerifyInitTag &, const team_member &member) const {
    const int tid =
        member.team_rank() + member.team_size() * member.league_rank();

    if (tid != m_flags(member.team_rank(), member.league_rank())) {
      Kokkos::printf("TestTeamPolicy member(%d,%d) error %d != %d\n",
                     member.league_rank(), member.team_rank(), tid,
                     m_flags(member.team_rank(), member.league_rank()));
    }
  }

  // Included for test_small_league_size.
  TestTeamPolicy() : m_flags() {}

  // Included for test_small_league_size.
  struct NoOpTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const NoOpTag &, const team_member & /*member*/) const {}

  static void test_small_league_size() {
    int bs = 8;   // batch size (number of elements per batch)
    int ns = 16;  // total number of "problems" to process

    // Calculate total scratch memory space size.
    const int level     = 0;
    int mem_size        = 960;
    const int num_teams = ns / bs;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> policy(num_teams, Kokkos::AUTO());

    Kokkos::parallel_for(
        policy.set_scratch_size(level, Kokkos::PerTeam(mem_size),
                                Kokkos::PerThread(0)),
        TestTeamPolicy());
  }

  static void test_constructors() {
    constexpr const int smallest_work = 1;
    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> none_auto(
        smallest_work,
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? 32
            : smallest_work,
        smallest_work);
#else
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> none_auto(
        smallest_work, smallest_work, smallest_work);
#endif
    (void)none_auto;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> both_auto(
        smallest_work, Kokkos::AUTO(), Kokkos::AUTO());
    (void)both_auto;
    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_vector(
        smallest_work,
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? 32
            : smallest_work,
        Kokkos::AUTO());
#else
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_vector(
        smallest_work, smallest_work, Kokkos::AUTO());
#endif
    (void)auto_vector;
    Kokkos::TeamPolicy<ExecSpace, NoOpTag> auto_team(
        smallest_work, Kokkos::AUTO(), smallest_work);
    (void)auto_team;
  }

  static void test_for(const size_t league_size) {
    {
      TestTeamPolicy functor(league_size);
      using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
      using policy_type_init =
          Kokkos::TeamPolicy<ScheduleType, ExecSpace, VerifyInitTag>;

      // FIXME_OPENMPTARGET temporary restriction for team size to be at least
      // 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
      const int team_size =
          policy_type(
              league_size,
              std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
                  ? 32
                  : 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
      const int team_size_init =
          policy_type_init(
              league_size,
              std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
                  ? 32
                  : 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
#else
      const int team_size =
          policy_type(league_size, 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
      const int team_size_init =
          policy_type_init(league_size, 1)
              .team_size_max(functor, Kokkos::ParallelForTag());
#endif

      Kokkos::parallel_for(policy_type(league_size, team_size), functor);
      Kokkos::parallel_for(policy_type_init(league_size, team_size_init),
                           functor);
    }

    test_small_league_size();
    test_constructors();
  }

  struct ReduceTag {};

  using value_type = int64_t;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &member, value_type &update) const {
    update += member.team_rank() + member.team_size() * member.league_rank();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ReduceTag &, const team_member &member,
                  value_type &update) const {
    update +=
        1 + member.team_rank() + member.team_size() * member.league_rank();
  }

  static void test_reduce(const size_t league_size) {
    TestTeamPolicy functor(league_size);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_reduce =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, ReduceTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    const int team_size =
        policy_type_reduce(
            league_size,
            std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
                ? 32
                : 1)
            .team_size_max(functor, Kokkos::ParallelReduceTag());
#else
    const int team_size =
        policy_type_reduce(league_size, 1)
            .team_size_max(functor, Kokkos::ParallelReduceTag());
#endif

    const int64_t N = team_size * league_size;

    int64_t total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);
    ASSERT_EQ(size_t((N - 1) * (N)) / 2, size_t(total));

    Kokkos::parallel_reduce(policy_type_reduce(league_size, team_size), functor,
                            total);
    ASSERT_EQ((size_t(N) * size_t(N + 1)) / 2, size_t(total));
  }
};

}  // namespace

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

template <typename ScalarType, class DeviceType, class ScheduleType>
class ReduceTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  struct value_type {
    ScalarType value[3];
  };

  const size_type nwork;

  KOKKOS_INLINE_FUNCTION
  ReduceTeamFunctor(const size_type &arg_nwork) : nwork(arg_nwork) {}

  KOKKOS_INLINE_FUNCTION
  ReduceTeamFunctor(const ReduceTeamFunctor &rhs) : nwork(rhs.nwork) {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type &dst) const {
    dst.value[0] = 0;
    dst.value[1] = 0;
    dst.value[2] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    dst.value[0] += src.value[0];
    dst.value[1] += src.value[1];
    dst.value[2] += src.value[2];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &dst) const {
    const int thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    const int thread_size = ind.team_size() * ind.league_size();
    const int chunk       = (nwork + thread_size - 1) / thread_size;

    size_type iwork           = static_cast<size_type>(chunk) * thread_rank;
    const size_type iwork_end = iwork + chunk < nwork ? iwork + chunk : nwork;

    for (; iwork < iwork_end; ++iwork) {
      dst.value[0] += 1;
      dst.value[1] += iwork + 1;
      dst.value[2] += nwork - iwork;
    }
  }
};

template <typename ScalarType, class DeviceType, class ScheduleType>
class ArrayReduceTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  using value_type      = ScalarType[];
  size_type value_count = 3;

  size_type nwork;

  KOKKOS_INLINE_FUNCTION
  ArrayReduceTeamFunctor(const size_type &arg_nwork) : nwork(arg_nwork) {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type dst) const {
    for (size_type i = 0; i < value_count; ++i) dst[i] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type dst, const value_type src) const {
    for (size_type i = 0; i < value_count; ++i) dst[i] += src[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &team,
                  value_type dst) const {
    const int thread_rank =
        team.team_rank() + team.team_size() * team.league_rank();
    const int thread_size = team.team_size() * team.league_size();
    const int chunk       = (nwork + thread_size - 1) / thread_size;

    size_type iwork           = static_cast<size_type>(chunk) * thread_rank;
    const size_type iwork_end = iwork + chunk < nwork ? iwork + chunk : nwork;

    for (; iwork < iwork_end; ++iwork) {
      dst[0] += 1;
      dst[1] += iwork + 1;
      dst[2] += nwork - iwork;
    }
  }
};

}  // namespace Test

namespace {

template <typename ScalarType, class DeviceType, class ScheduleType>
class TestReduceTeam {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using size_type       = typename execution_space::size_type;

  void run_test(const size_type &nwork) {
    enum { Count = 3 };
    enum { Repeat = 100 };

    const uint64_t nw   = nwork;
    const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

    policy_type team_exec(nw, 1);

    {
      using functor_type =
          Test::ReduceTeamFunctor<ScalarType, execution_space, ScheduleType>;
      using value_type = typename functor_type::value_type;
      using result_type =
          Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

      value_type result[Repeat];

      const unsigned team_size = team_exec.team_size_recommended(
          functor_type(nwork), Kokkos::ParallelReduceTag());
      const unsigned league_size = (nwork + team_size - 1) / team_size;

      team_exec = policy_type(league_size, team_size);

      for (unsigned i = 0; i < Repeat; ++i) {
        result_type tmp(&result[i]);
        Kokkos::parallel_reduce(team_exec, functor_type(nwork), tmp);
      }

      execution_space().fence();

      for (unsigned i = 0; i < Repeat; ++i) {
        for (unsigned j = 0; j < Count; ++j) {
          const uint64_t correct = (j == 0) ? nw : nsum;
          ASSERT_EQ((ScalarType)correct, result[i].value[j]);
        }
      }
    }
  }

  void run_array_test(const size_type &nwork) {
    enum { Count = 3 };
    enum { Repeat = 100 };

    const uint64_t nw   = nwork;
    const uint64_t nsum = nw % 2 ? nw * ((nw + 1) / 2) : (nw / 2) * (nw + 1);

    policy_type team_exec(nw, 1);

    {
      using functor_type =
          Test::ArrayReduceTeamFunctor<ScalarType, execution_space,
                                       ScheduleType>;
      using result_type = Kokkos::View<ScalarType *, Kokkos::HostSpace,
                                       Kokkos::MemoryUnmanaged>;

      ScalarType result[Repeat][Count];

      const unsigned team_size = team_exec.team_size_recommended(
          functor_type(nwork), Kokkos::ParallelReduceTag());
      const unsigned league_size = (nwork + team_size - 1) / team_size;

      team_exec = policy_type(league_size, team_size);

      for (unsigned i = 0; i < Repeat; ++i) {
        result_type tmp(&result[i][0], Count);
        Kokkos::parallel_reduce(team_exec, functor_type(nwork), tmp);
      }

      execution_space().fence();

      for (unsigned i = 0; i < Repeat; ++i) {
        for (unsigned j = 0; j < Count; ++j) {
          ASSERT_EQ(j ? nsum : nw, static_cast<uint64_t>(result[i][j]))
              << "failing at repeat " << i << " and index " << j;
        }
      }
    }
  }
};

}  // namespace

/*--------------------------------------------------------------------------*/

namespace Test {

template <class DeviceType, class ScheduleType>
class ScanTeamFunctor {
 public:
  using execution_space = DeviceType;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using value_type      = int64_t;

  Kokkos::View<value_type, execution_space> accum;
  Kokkos::View<value_type, execution_space> total;

  ScanTeamFunctor() : accum("accum"), total("total") {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type &error) const { error = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &error, value_type const &input) const {
    if (input) error = 1;
  }

  struct JoinMax {
    using value_type = int64_t;

    KOKKOS_INLINE_FUNCTION
    void join(value_type &dst, value_type const &input) const {
      if (dst < input) dst = input;
    }
  };

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type ind,
                  value_type &error) const {
    if (0 == ind.league_rank() && 0 == ind.team_rank()) {
      const int64_t thread_count =
          static_cast<int64_t>(ind.league_size()) * ind.team_size();
      total() = (thread_count * (thread_count + 1)) / 2;
    }

    // Team max:
    int64_t m = static_cast<int64_t>(ind.league_rank()) + ind.team_rank();
    ind.team_reduce(Kokkos::Max<int64_t>(m));

    if (m != ind.league_rank() + (ind.team_size() - 1)) {
      Kokkos::printf(
          "ScanTeamFunctor[%i.%i of %i.%i] reduce_max_answer(%li) != "
          "reduce_max(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()),
          static_cast<long>(ind.league_rank()) + ind.team_size() - 1,
          static_cast<long>(m));
    }

    // Scan:
    const int64_t answer = (ind.league_rank() + 1) * ind.team_rank() +
                           (ind.team_rank() * (ind.team_rank() + 1)) / 2;

    const int64_t result =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    const int64_t result2 =
        ind.team_scan(ind.league_rank() + 1 + ind.team_rank() + 1);

    if (answer != result || answer != result2) {
      Kokkos::printf(
          "ScanTeamFunctor[%i.%i of %i.%i] answer(%li) != scan_first(%li) or "
          "scan_second(%li)\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.team_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_size()), static_cast<long>(answer),
          static_cast<long>(result), static_cast<long>(result2));

      error = 1;
    }

    const int64_t thread_rank =
        ind.team_rank() + ind.team_size() * ind.league_rank();
    ind.team_scan(1 + thread_rank, accum.data());
  }
};

template <class DeviceType, class ScheduleType>
class TestScanTeam {
 public:
  using execution_space = DeviceType;
  using value_type      = int64_t;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;
  using functor_type    = Test::ScanTeamFunctor<DeviceType, ScheduleType>;

  TestScanTeam(const size_t nteam) { run_test(nteam); }

  void run_test(const size_t nteam) {
    using result_type =
        Kokkos::View<int64_t, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

    const unsigned REPEAT = 100000;
    unsigned Repeat;

    if (nteam == 0) {
      Repeat = 1;
    } else {
      Repeat = (REPEAT + nteam - 1) / nteam;  // Error here.
    }

    functor_type functor;

    policy_type team_exec(nteam, 1);
    const auto team_size =
        team_exec.team_size_max(functor, Kokkos::ParallelReduceTag());
    team_exec = policy_type(nteam, team_size);

    for (unsigned i = 0; i < Repeat; ++i) {
      int64_t accum = 0;
      int64_t total = 0;
      int64_t error = 0;
      Kokkos::deep_copy(functor.accum, total);

      Kokkos::parallel_reduce(team_exec, functor, result_type(&error));
      DeviceType().fence();

      Kokkos::deep_copy(accum, functor.accum);
      Kokkos::deep_copy(total, functor.total);

      ASSERT_EQ(error, 0);
      ASSERT_EQ(total, accum);
    }

    execution_space().fence();
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

template <class ExecSpace, class ScheduleType>
struct SharedTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_COUNT = 1000 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      Kokkos::View<int *, shmem_space, Kokkos::MemoryUnmanaged>;

  // Tell how much shared memory will be required by this functor.
  inline unsigned team_shmem_size(int /*team_size*/) const {
    return shared_int_array_type::shmem_size(SHARED_COUNT) +
           shared_int_array_type::shmem_size(SHARED_COUNT);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
    const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

    if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
        (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
      Kokkos::printf(
          "member( %i/%i , %i/%i ) Failed to allocate shared memory of size "
          "%lu\n",
          static_cast<int>(ind.league_rank()),
          static_cast<int>(ind.league_size()),
          static_cast<int>(ind.team_rank()), static_cast<int>(ind.team_size()),
          static_cast<unsigned long>(SHARED_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      for (int i = ind.team_rank(); i < SHARED_COUNT; i += ind.team_size()) {
        shared_A[i] = i + ind.league_rank();
        shared_B[i] = 2 * i + ind.league_rank();
      }

      ind.team_barrier();

      if (ind.team_rank() + 1 == ind.team_size()) {
        for (int i = 0; i < SHARED_COUNT; ++i) {
          if (shared_A[i] != i + ind.league_rank()) {
            ++update;
          }

          if (shared_B[i] != 2 * i + ind.league_rank()) {
            ++update;
          }
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestSharedTeam {
  TestSharedTeam() { run(); }

  void run() {
    using Functor = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        Kokkos::View<typename Functor::value_type, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>;

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    const size_t team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? Kokkos::TeamPolicy<ScheduleType, ExecSpace>(64, 32).team_size_max(
                  Functor(), Kokkos::ParallelReduceTag())
            : Kokkos::TeamPolicy<ScheduleType, ExecSpace>(8192, 1)
                  .team_size_max(Functor(), Kokkos::ParallelReduceTag());

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? 32 / team_size
            : 8192 / team_size,
        team_size);
#else
    const size_t team_size =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace>(8192, 1).team_size_max(
            Functor(), Kokkos::ParallelReduceTag());

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);
#endif

    typename Functor::value_type error_count = 0;

    Kokkos::parallel_reduce(team_exec, Functor(), result_type(&error_count));
    Kokkos::fence();

    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

template <class MemorySpace, class ExecSpace, class ScheduleType>
struct TestLambdaSharedTeam {
  TestLambdaSharedTeam() { run(); }

  void run() {
    using Functor     = Test::SharedTeamFunctor<ExecSpace, ScheduleType>;
    using result_type = Kokkos::View<typename Functor::value_type, MemorySpace,
                                     Kokkos::MemoryUnmanaged>;

    using shmem_space = typename ExecSpace::scratch_memory_space;

    // TBD: MemoryUnmanaged should be the default for shared memory space.
    using shared_int_array_type =
        Kokkos::View<int *, shmem_space, Kokkos::MemoryUnmanaged>;

    const int SHARED_COUNT = 1000;
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int team_size    = 1;
#endif

#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<ExecSpace, Kokkos::Cuda>::value) team_size = 128;
#endif

    Kokkos::TeamPolicy<ScheduleType, ExecSpace> team_exec(8192 / team_size,
                                                          team_size);

    int scratch_size = shared_int_array_type::shmem_size(SHARED_COUNT) * 2;
    team_exec = team_exec.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    typename Functor::value_type error_count = 0;

    Kokkos::parallel_reduce(
        team_exec,
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<ScheduleType,
                                              ExecSpace>::member_type &ind,
            int &update) {
          const shared_int_array_type shared_A(ind.team_shmem(), SHARED_COUNT);
          const shared_int_array_type shared_B(ind.team_shmem(), SHARED_COUNT);

          if ((shared_A.data() == nullptr && SHARED_COUNT > 0) ||
              (shared_B.data() == nullptr && SHARED_COUNT > 0)) {
            Kokkos::printf("Failed to allocate shared memory of size %lu\n",
                           static_cast<unsigned long>(SHARED_COUNT));

            ++update;  // Failure to allocate is an error.
          } else {
            for (int i = ind.team_rank(); i < SHARED_COUNT;
                 i += ind.team_size()) {
              shared_A[i] = i + ind.league_rank();
              shared_B[i] = 2 * i + ind.league_rank();
            }

            ind.team_barrier();

            if (ind.team_rank() + 1 == ind.team_size()) {
              for (int i = 0; i < SHARED_COUNT; ++i) {
                if (shared_A[i] != i + ind.league_rank()) {
                  ++update;
                }

                if (shared_B[i] != 2 * i + ind.league_rank()) {
                  ++update;
                }
              }
            }
          }
        },
        result_type(&error_count));

    Kokkos::fence();

    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace Test

namespace Test {

template <class ExecSpace, class ScheduleType>
struct ScratchTeamFunctor {
  using execution_space = ExecSpace;
  using value_type      = int;
  using policy_type     = Kokkos::TeamPolicy<ScheduleType, execution_space>;

  enum { SHARED_TEAM_COUNT = 100 };
  enum { SHARED_THREAD_COUNT = 10 };

  using shmem_space = typename ExecSpace::scratch_memory_space;

  // TBD: MemoryUnmanaged should be the default for shared memory space.
  using shared_int_array_type =
      Kokkos::View<size_t *, shmem_space, Kokkos::MemoryUnmanaged>;

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename policy_type::member_type &ind,
                  value_type &update) const {
    const shared_int_array_type scratch_ptr(ind.team_scratch(1),
                                            3 * ind.team_size());
    const shared_int_array_type scratch_A(ind.team_scratch(1),
                                          SHARED_TEAM_COUNT);
    const shared_int_array_type scratch_B(ind.thread_scratch(1),
                                          SHARED_THREAD_COUNT);

    if ((scratch_ptr.data() == nullptr) ||
        (scratch_A.data() == nullptr && SHARED_TEAM_COUNT > 0) ||
        (scratch_B.data() == nullptr && SHARED_THREAD_COUNT > 0)) {
      Kokkos::printf("Failed to allocate shared memory of size %lu\n",
                     static_cast<unsigned long>(SHARED_TEAM_COUNT));

      ++update;  // Failure to allocate is an error.
    } else {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(ind, 0, (int)SHARED_TEAM_COUNT),
          [&](const int &i) { scratch_A[i] = i + ind.league_rank(); });

      for (int i = 0; i < SHARED_THREAD_COUNT; i++) {
        scratch_B[i] = 10000 * ind.league_rank() + 100 * ind.team_rank() + i;
      }

      scratch_ptr[ind.team_rank()]                   = (size_t)scratch_A.data();
      scratch_ptr[ind.team_rank() + ind.team_size()] = (size_t)scratch_B.data();

      ind.team_barrier();

      for (int i = 0; i < SHARED_TEAM_COUNT; i++) {
        if (scratch_A[i] != size_t(i) + ind.league_rank()) ++update;
      }

      for (int i = 0; i < ind.team_size(); i++) {
        if (scratch_ptr[0] != scratch_ptr[i]) ++update;
      }

      if (scratch_ptr[1 + ind.team_size()] - scratch_ptr[0 + ind.team_size()] <
          SHARED_THREAD_COUNT * sizeof(size_t)) {
        ++update;
      }

      for (int i = 1; i < ind.team_size(); i++) {
        if ((scratch_ptr[i + ind.team_size()] -
             scratch_ptr[i - 1 + ind.team_size()]) !=
            (scratch_ptr[1 + ind.team_size()] -
             scratch_ptr[0 + ind.team_size()])) {
          ++update;
        }
      }
    }
  }
};

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestScratchTeam {
  TestScratchTeam() { run(); }

  void run() {
    using Functor = Test::ScratchTeamFunctor<ExecSpace, ScheduleType>;
    using result_type =
        Kokkos::View<typename Functor::value_type, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>;
    using p_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;

    typename Functor::value_type error_count = 0;

    int thread_scratch_size = Functor::shared_int_array_type::shmem_size(
        Functor::SHARED_THREAD_COUNT);

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    p_type team_exec =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? p_type(64, 32).set_scratch_size(
                  1,
                  Kokkos::PerTeam(Functor::shared_int_array_type::shmem_size(
                      Functor::SHARED_TEAM_COUNT)),
                  Kokkos::PerThread(thread_scratch_size + 3 * sizeof(int)))
            : p_type(8192, 1).set_scratch_size(
                  1,
                  Kokkos::PerTeam(Functor::shared_int_array_type::shmem_size(
                      Functor::SHARED_TEAM_COUNT)),
                  Kokkos::PerThread(thread_scratch_size + 3 * sizeof(int)));
#else
    p_type team_exec = p_type(8192, 1).set_scratch_size(
        1,
        Kokkos::PerTeam(Functor::shared_int_array_type::shmem_size(
            Functor::SHARED_TEAM_COUNT)),
        Kokkos::PerThread(thread_scratch_size + 3 * sizeof(int)));
#endif

    const size_t team_size =
        team_exec.team_size_max(Functor(), Kokkos::ParallelReduceTag());

    int team_scratch_size =
        Functor::shared_int_array_type::shmem_size(Functor::SHARED_TEAM_COUNT) +
        Functor::shared_int_array_type::shmem_size(3 * team_size);

#ifdef KOKKOS_ENABLE_OPENMPTARGET
    team_exec =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value
            ? p_type(64 / team_size, team_size)
            : p_type(8192 / team_size, team_size);
#else
    team_exec     = p_type(8192 / team_size, team_size);
#endif

    Kokkos::parallel_reduce(
        team_exec.set_scratch_size(1, Kokkos::PerTeam(team_scratch_size),
                                   Kokkos::PerThread(thread_scratch_size)),
        Functor(), result_type(&error_count));
    Kokkos::fence();
    ASSERT_EQ(error_count, 0);

    Kokkos::parallel_reduce(
        team_exec.set_scratch_size(1, Kokkos::PerTeam(team_scratch_size),
                                   Kokkos::PerThread(thread_scratch_size)),
        Functor(), Kokkos::Sum<typename Functor::value_type>(error_count));
    Kokkos::fence();
    ASSERT_EQ(error_count, 0);
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
KOKKOS_INLINE_FUNCTION int test_team_mulit_level_scratch_loop_body(
    const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_team1(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_thread1(team.thread_scratch(0), 16);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_team2(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_thread2(team.thread_scratch(0), 16);

  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_team1(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_thread1(team.thread_scratch(1), 1600);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_team2(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_thread2(team.thread_scratch(1), 1600);

  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_team3(team.team_scratch(0), 128);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a_thread3(team.thread_scratch(0), 16);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_team3(team.team_scratch(1), 12800);
  Kokkos::View<double *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      b_thread3(team.thread_scratch(1), 1600);

  // The explicit types for 0 and 128 are here to test TeamThreadRange accepting
  // different types for begin and end.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, int(0), unsigned(128)),
                       [&](const int &i) {
                         a_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         a_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         a_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, int(0), unsigned(16)),
                       [&](const int &i) {
                         a_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         a_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, int(0), unsigned(12800)),
                       [&](const int &i) {
                         b_team1(i) = 1000000 + i + team.league_rank() * 100000;
                         b_team2(i) = 2000000 + i + team.league_rank() * 100000;
                         b_team3(i) = 3000000 + i + team.league_rank() * 100000;
                       });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 1600),
                       [&](const int &i) {
                         b_thread1(i) = 1000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread2(i) = 2000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                         b_thread3(i) = 3000000 + 100000 * team.team_rank() +
                                        16 - i + team.league_rank() * 100000;
                       });

  team.team_barrier();

  int error = 0;
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, 0, 128), [&](const int &i) {
        if (a_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (a_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (a_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 16), [&](const int &i) {
    if (a_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
    if (a_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                            team.league_rank() * 100000)
      error++;
  });

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, 0, 12800), [&](const int &i) {
        if (b_team1(i) != 1000000 + i + team.league_rank() * 100000) error++;
        if (b_team2(i) != 2000000 + i + team.league_rank() * 100000) error++;
        if (b_team3(i) != 3000000 + i + team.league_rank() * 100000) error++;
      });
  team.team_barrier();

  Kokkos::parallel_for(
      Kokkos::ThreadVectorRange(team, 1600), [&](const int &i) {
        if (b_thread1(i) != 1000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread2(i) != 2000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
        if (b_thread3(i) != 3000000 + 100000 * team.team_rank() + 16 - i +
                                team.league_rank() * 100000)
          error++;
      });

  return error;
}

struct TagReduce {};
struct TagFor {};

template <class ExecSpace, class ScheduleType>
struct ClassNoShmemSizeFunction {
  using member_type =
      typename Kokkos::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> errors;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    Kokkos::View<int, ExecSpace> d_errors =
        Kokkos::View<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(128);
    const int per_thread0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(16);

    const int per_team1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(12800);
    const int per_thread1 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(1600);

#ifdef KOKKOS_ENABLE_SYCL
    int team_size = 4;
#else
    int team_size = 8;
#endif
    int const concurrency = ExecSpace().concurrency();
    if (team_size > concurrency) team_size = concurrency;
    {
      Kokkos::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      Kokkos::parallel_for(
          policy
              .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                                Kokkos::PerThread(per_thread0))
              .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                Kokkos::PerThread(per_thread1)),
          *this);
      Kokkos::fence();

      typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
          Kokkos::create_mirror_view(d_errors);
      Kokkos::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      Kokkos::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      Kokkos::parallel_reduce(
          policy
              .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                                Kokkos::PerThread(per_thread0))
              .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                Kokkos::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  }
};

template <class ExecSpace, class ScheduleType>
struct ClassWithShmemSizeFunction {
  using member_type =
      typename Kokkos::TeamPolicy<ExecSpace, ScheduleType>::member_type;

  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> errors;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagFor &, const member_type &team) const {
    int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
    errors() += error;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagReduce &, const member_type &team,
                  int &error) const {
    error += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
  }

  void run() {
    Kokkos::View<int, ExecSpace> d_errors =
        Kokkos::View<int, ExecSpace>("Errors");
    errors = d_errors;

    const int per_team1 =
        3 * Kokkos::View<
                double *, ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(12800);
    const int per_thread1 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(1600);

    int team_size = 8;

    int const concurrency = ExecSpace().concurrency();
    if (team_size > concurrency) team_size = concurrency;

    {
      Kokkos::TeamPolicy<TagFor, ExecSpace, ScheduleType> policy(10, team_size,
                                                                 16);

      Kokkos::parallel_for(
          policy.set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                  Kokkos::PerThread(per_thread1)),
          *this);
      Kokkos::fence();

      typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
          Kokkos::create_mirror_view(d_errors);
      Kokkos::deep_copy(h_errors, d_errors);
      ASSERT_EQ(h_errors(), 0);
    }

    {
      int error = 0;
      Kokkos::TeamPolicy<TagReduce, ExecSpace, ScheduleType> policy(
          10, team_size, 16);

      Kokkos::parallel_reduce(
          policy.set_scratch_size(1, Kokkos::PerTeam(per_team1),
                                  Kokkos::PerThread(per_thread1)),
          *this, error);

      ASSERT_EQ(error, 0);
    }
  }

  unsigned team_shmem_size(int team_size) const {
    const int per_team0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(128);
    const int per_thread0 =
        3 *
        Kokkos::View<double *, ExecSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(16);
    return per_team0 + team_size * per_thread0;
  }
};

template <class ExecSpace, class ScheduleType>
void test_team_mulit_level_scratch_test_lambda() {
  Kokkos::View<int, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> errors;
  Kokkos::View<int, ExecSpace> d_errors("Errors");
  errors = d_errors;

  const int per_team0 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(128);
  const int per_thread0 =
      3 * Kokkos::View<double *, ExecSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(16);

  const int per_team1 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(12800);
  const int per_thread1 =
      3 *
      Kokkos::View<double *, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(1600);

#ifdef KOKKOS_ENABLE_SYCL
  int team_size = 4;
#else
  int team_size = 8;
#endif
  int const concurrency = ExecSpace().concurrency();
  if (team_size > concurrency) team_size = concurrency;

  Kokkos::TeamPolicy<ExecSpace, ScheduleType> policy(10, team_size, 16);

  Kokkos::parallel_for(
      policy
          .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                            Kokkos::PerThread(per_thread0))
          .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                            Kokkos::PerThread(per_thread1)),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
        int error = test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
        errors() += error;
      });
  Kokkos::fence();

  typename Kokkos::View<int, ExecSpace>::HostMirror h_errors =
      Kokkos::create_mirror_view(errors);
  Kokkos::deep_copy(h_errors, d_errors);
  ASSERT_EQ(h_errors(), 0);

  int error = 0;
  Kokkos::parallel_reduce(
      policy
          .set_scratch_size(0, Kokkos::PerTeam(per_team0),
                            Kokkos::PerThread(per_thread0))
          .set_scratch_size(1, Kokkos::PerTeam(per_team1),
                            Kokkos::PerThread(per_thread1)),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team,
          int &count) {
        count += test_team_mulit_level_scratch_loop_body<ExecSpace>(team);
      },
      error);
  ASSERT_EQ(error, 0);
}

}  // namespace Test

namespace {

template <class ExecSpace, class ScheduleType>
struct TestMultiLevelScratchTeam {
  TestMultiLevelScratchTeam() { run(); }

  void run() {
    Test::test_team_mulit_level_scratch_test_lambda<ExecSpace, ScheduleType>();
    Test::ClassNoShmemSizeFunction<ExecSpace, ScheduleType> c1;
    c1.run();

    Test::ClassWithShmemSizeFunction<ExecSpace, ScheduleType> c2;
    c2.run();
  }
};

}  // namespace

namespace Test {

template <class ExecSpace>
struct TestShmemSize {
  TestShmemSize() { run(); }

  void run() {
    using view_type = Kokkos::View<int64_t ***, ExecSpace>;

    size_t d1 = 5;
    size_t d2 = 6;
    size_t d3 = 7;

    size_t size = view_type::shmem_size(d1, d2, d3);

    ASSERT_EQ(size, (d1 * d2 * d3 + 1) * sizeof(int64_t));

    test_layout_stride();
  }

  void test_layout_stride() {
    int rank       = 3;
    int order[3]   = {2, 0, 1};
    int extents[3] = {100, 10, 3};
    auto s1 =
        Kokkos::View<double ***, Kokkos::LayoutStride, ExecSpace>::shmem_size(
            Kokkos::LayoutStride::order_dimensions(rank, order, extents));
    auto s2 =
        Kokkos::View<double ***, Kokkos::LayoutRight, ExecSpace>::shmem_size(
            extents[0], extents[1], extents[2]);
    ASSERT_EQ(s1, s2);
  }
};

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

namespace {

template <class ExecSpace, class ScheduleType, class T, class Enabled = void>
struct TestTeamBroadcast;

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<ExecSpace, ScheduleType, T,
                         std::enable_if_t<(sizeof(T) == sizeof(char)), void>> {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using memory_space = typename ExecSpace::memory_space;
  using value_type   = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        Kokkos::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid % 0xFF) + offset;

    teamMember.team_broadcast([&](value_type &var) { var -= offset; }, value,
                              lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate |= value; },
        Kokkos::BOr<value_type, memory_space>(parUpdate));

    if (teamMember.team_rank() == 0) update |= parUpdate;
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int fake_team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int fake_team_size = 1;
#endif
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                Kokkos::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);

    // team_broadcast with value
    value_type total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            Kokkos::BOr<value_type, Kokkos::HostSpace>(total));

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = (value_type((i % team_size % 0xFF)) + off);
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with value --"
    //"expected_result=%x,"
    //"total=%x\n",expected_result, total);

    // team_broadcast with function object
    total = 0;

    Kokkos::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            Kokkos::BOr<value_type, Kokkos::HostSpace>(total));

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size % 0xFF)));
      expected_result |= val;
    }
    ASSERT_EQ(expected_result, total);
    // printf("team_broadcast with function object --"
    // "expected_result=%x,"
    // "total=%x\n",expected_result, total);
  }
};

template <class ExecSpace, class ScheduleType, class T>
struct TestTeamBroadcast<ExecSpace, ScheduleType, T,
                         std::enable_if_t<(sizeof(T) > sizeof(char)), void>> {
  using team_member =
      typename Kokkos::TeamPolicy<ScheduleType, ExecSpace>::member_type;
  using value_type = T;

  const value_type offset;

  TestTeamBroadcast(const size_t /*league_size*/, const value_type os_)
      : offset(os_) {}

  struct BroadcastTag {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &teamMember, value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0
    bool setValue = ((lid % ts) == tid);

    // broadcast boolean and value to team from source thread
    teamMember.team_broadcast(value, lid % ts);
    teamMember.team_broadcast(setValue, lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const BroadcastTag &, const team_member &teamMember,
                  value_type &update) const {
    int lid = teamMember.league_rank();
    int tid = teamMember.team_rank();
    int ts  = teamMember.team_size();

    value_type parUpdate = 0;
    value_type value     = (value_type)(tid * 3) + offset;

    // setValue is used to determine if the update should be
    // performed at the bottom.  The thread id must match the
    // thread id used to broadcast the value.  It is the
    // thread id that matches the league rank mod team size
    // this way each league rank will use a different thread id
    // which is likely not 0. Note the logic is switched from
    // above because the functor switches it back.
    bool setValue = ((lid % ts) != tid);

    teamMember.team_broadcast([&](value_type &var) { var += var; }, value,
                              lid % ts);
    teamMember.team_broadcast([&](bool &bVar) { bVar = !bVar; }, setValue,
                              lid % ts);

    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(teamMember, ts),
        [&](const int /*j*/, value_type &teamUpdate) { teamUpdate += value; },
        parUpdate);

    if (teamMember.team_rank() == 0 && setValue) update += parUpdate;
  }

  template <class ScalarType>
  static inline std::enable_if_t<!std::is_integral_v<ScalarType>, void>
  compare_test(ScalarType A, ScalarType B, double epsilon_factor) {
    if (std::is_same_v<ScalarType, double> ||
        std::is_same_v<ScalarType, float>) {
      ASSERT_NEAR((double)A, (double)B,
                  epsilon_factor * std::abs(A) *
                      std::numeric_limits<ScalarType>::epsilon());
    } else {
      ASSERT_EQ(A, B);
    }
  }

  template <class ScalarType>
  static inline std::enable_if_t<std::is_integral_v<ScalarType>, void>
  compare_test(ScalarType A, ScalarType B, double) {
    ASSERT_EQ(A, B);
  }

  static void test_teambroadcast(const size_t league_size,
                                 const value_type off) {
    TestTeamBroadcast functor(league_size, off);

    using policy_type = Kokkos::TeamPolicy<ScheduleType, ExecSpace>;
    using policy_type_f =
        Kokkos::TeamPolicy<ScheduleType, ExecSpace, BroadcastTag>;

    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int fake_team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int fake_team_size = 1;
#endif
    const int team_size =
        policy_type_f(league_size, fake_team_size)
            .team_size_max(
                functor,
                Kokkos::
                    ParallelReduceTag());  // printf("team_size=%d\n",team_size);
    // team_broadcast with value
    value_type total = 0;

    Kokkos::parallel_reduce(policy_type(league_size, team_size), functor,
                            total);

    value_type expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val =
          (value_type((i % team_size) * 3) + off) * value_type(team_size);
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));

    // team_broadcast with function object
    total = 0;

    Kokkos::parallel_reduce(policy_type_f(league_size, team_size), functor,
                            total);

    expected_result = 0;
    for (unsigned int i = 0; i < league_size; i++) {
      value_type val = ((value_type)((i % team_size) * 3) + off) *
                       (value_type)(2 * team_size);
      expected_result += val;
    }
    // For comparison purposes treat the reduction as a random walk in the
    // least significant digit, which gives a typical walk distance
    // sqrt(league_size) Add 4x for larger sigma
    compare_test(expected_result, total, 4.0 * std::sqrt(league_size));
  }
};

template <class ExecSpace>
struct TestScratchAlignment {
  struct TestScalar {
    double x, y, z;
  };
  TestScratchAlignment() {
    test_view(true);
    test_view(false);
    test_minimal();
    test_raw();
  }
  using ScratchView =
      Kokkos::View<TestScalar *, typename ExecSpace::scratch_memory_space>;
  using ScratchViewInt =
      Kokkos::View<int *, typename ExecSpace::scratch_memory_space>;
  void test_view(bool allocate_small) {
    int shmem_size = ScratchView::shmem_size(11);
    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int team_size      = 1;
#endif
    if (allocate_small) shmem_size += ScratchViewInt::shmem_size(1);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(1, team_size)
            .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<ExecSpace>::member_type &team) {
          if (allocate_small) ScratchViewInt(team.team_scratch(0), 1);
          ScratchView a(team.team_scratch(0), 11);
          if (ptrdiff_t(a.data()) % sizeof(TestScalar) != 0)
            Kokkos::abort("Error: invalid scratch view alignment\n");
        });
    Kokkos::fence();
  }

  // test really small size of scratch space, produced error before
  void test_minimal() {
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    // FIXME_OPENMPTARGET temporary restriction for team size to be at least 32
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int team_size      = 1;
#endif
    Kokkos::TeamPolicy<ExecSpace> policy(1, team_size);
    size_t scratch_size = sizeof(int);
    Kokkos::View<int, ExecSpace> flag("Flag");

    Kokkos::parallel_for(
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type &team) {
          int *scratch_ptr = (int *)team.team_shmem().get_shmem(scratch_size);
          if (scratch_ptr == nullptr) flag() = 1;
        });
    Kokkos::fence();
    int minimal_scratch_allocation_failed = 0;
    Kokkos::deep_copy(minimal_scratch_allocation_failed, flag);
    ASSERT_EQ(minimal_scratch_allocation_failed, 0);
  }

  // test alignment of successive allocations
  void test_raw() {
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    int team_size =
        std::is_same<ExecSpace, Kokkos::Experimental::OpenMPTarget>::value ? 32
                                                                           : 1;
#else
    int team_size      = 1;
#endif
    Kokkos::TeamPolicy<ExecSpace> policy(1, team_size);
    Kokkos::View<int, ExecSpace> flag("Flag");

    Kokkos::parallel_for(
        policy.set_scratch_size(0, Kokkos::PerTeam(1024)),
        KOKKOS_LAMBDA(const member_type &team) {
          // first get some unaligned allocations, should give back
          // exactly the requested number of bytes
          auto scratch_ptr1 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(24));
          auto scratch_ptr2 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(32));
          auto scratch_ptr3 =
              reinterpret_cast<intptr_t>(team.team_shmem().get_shmem(12));

          if (((scratch_ptr2 - scratch_ptr1) != 24) ||
              ((scratch_ptr3 - scratch_ptr2) != 32))
            flag() = 1;

          // Now request aligned memory such that the allocation after
          // scratch_ptr2 would be unaligned if it doesn't pad correctly.
          // Depending on scratch_ptr3 being 4 or 8 byte aligned
          // we need to request a different amount of memory.
          if ((scratch_ptr3 + 12) % 8 == 4)
            scratch_ptr1 = reinterpret_cast<intptr_t>(
                team.team_shmem().get_shmem_aligned(24, 4));
          else {
            scratch_ptr1 = reinterpret_cast<intptr_t>(
                team.team_shmem().get_shmem_aligned(12, 4));
          }
          scratch_ptr2 = reinterpret_cast<intptr_t>(
              team.team_shmem().get_shmem_aligned(32, 8));
          scratch_ptr3 = reinterpret_cast<intptr_t>(
              team.team_shmem().get_shmem_aligned(8, 4));

          // The difference between scratch_ptr2 and scratch_ptr1 should be 4
          // bytes larger than what we requested in either case.
          if (((scratch_ptr2 - scratch_ptr1) != 28) &&
              ((scratch_ptr2 - scratch_ptr1) != 16))
            flag() = 1;
          // Check that there wasn't unneccessary padding happening. Since
          // scratch_ptr2 was allocated with a 32 byte request and scratch_ptr3
          // is then already aligned, its difference should match 32 bytes.
          if ((scratch_ptr3 - scratch_ptr2) != 32) flag() = 1;

          // check actually alignment of ptrs is as requested
          // cast to int here to avoid failure with icpx in mixed integer type
          // comparison
          if ((int(scratch_ptr1 % 4) != 0) || (int(scratch_ptr2 % 8) != 0) ||
              (int(scratch_ptr3 % 4) != 0))
            flag() = 1;
        });
    Kokkos::fence();
    int raw_get_shmem_alignment_failed = 0;
    Kokkos::deep_copy(raw_get_shmem_alignment_failed, flag);
    ASSERT_EQ(raw_get_shmem_alignment_failed, 0);
  }
};

}  // namespace

namespace {
template <class ExecSpace>
struct TestTeamPolicyHandleByValue {
  using scalar     = double;
  using exec_space = ExecSpace;
  using mem_space  = typename ExecSpace::memory_space;

  TestTeamPolicyHandleByValue() { test(); }

  void test() {
    const int M = 1, N = 1;
    Kokkos::View<scalar **, mem_space> a("a", M, N);
    Kokkos::View<scalar **, mem_space> b("b", M, N);
    Kokkos::deep_copy(a, 0.0);
    Kokkos::deep_copy(b, 1.0);
    Kokkos::parallel_for(
        "test_tphandle_by_value",
        Kokkos::TeamPolicy<exec_space>(M, Kokkos::AUTO(), 1),
        KOKKOS_LAMBDA(
            const typename Kokkos::TeamPolicy<exec_space>::member_type team) {
          const int i = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, N),
                               [&](const int j) { a(i, j) += b(i, j); });
        });
  }
};

}  // namespace

namespace {
template <typename ExecutionSpace>
struct TestRepeatedTeamReduce {
  static constexpr int ncol = 1500;  // nothing special, just some work

  KOKKOS_FUNCTION void operator()(
      const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &team)
      const {
    // non-divisible by power of two to make triggering problems easier
    constexpr int nlev = 129;
    constexpr auto pi  = Kokkos::numbers::pi;
    double b           = 0.;
    for (int ri = 0; ri < 10; ++ri) {
      // The contributions here must be sufficiently complex, simply adding ones
      // wasn't enough to trigger the bug.
      const auto g1 = [&](const int k, double &acc) {
        acc += Kokkos::cos(pi * double(k) / nlev);
      };
      const auto g2 = [&](const int k, double &acc) {
        acc += Kokkos::sin(pi * double(k) / nlev);
      };
      double a1, a2;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nlev), g1, a1);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nlev), g2, a2);
      b += a1;
      b += a2;
    }
    const auto h = [&]() {
      const auto col = team.league_rank();
      v(col)         = b + col;
    };
    Kokkos::single(Kokkos::PerTeam(team), h);
  }

  KOKKOS_FUNCTION void operator()(const int i, int &bad) const {
    if (v(i) != v(0) + i) {
      ++bad;
      Kokkos::printf("Failing at %d!\n", i);
    }
  }

  TestRepeatedTeamReduce() : v("v", ncol) { test(); }

  void test() {
    int team_size_recommended =
        Kokkos::TeamPolicy<ExecutionSpace>(1, 1).team_size_recommended(
            *this, Kokkos::ParallelForTag());
    // Choose a non-recommened (non-power of two for GPUs) team size
    int team_size = team_size_recommended > 1 ? team_size_recommended - 1 : 1;

    // The failure was non-deterministic so run the test a bunch of times
    for (int it = 0; it < 100; ++it) {
      Kokkos::parallel_for(
          Kokkos::TeamPolicy<ExecutionSpace>(ncol, team_size, 1), *this);

      int bad = 0;
      Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, ncol),
                              *this, bad);
      ASSERT_EQ(bad, 0) << " Failing in iteration " << it;
    }
  }

  Kokkos::View<double *, ExecutionSpace> v;
};

}  // namespace

}  // namespace Test

namespace Test {

struct SimpleTestValueType {
  using ScalarType = int;

  ScalarType value[2];
};

struct TestTeamReducerFunctor {
  using value_type = SimpleTestValueType;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &init) const {
    init.value[0] = 1;
    init.value[1] = 10;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, value_type const &src) const {
    dst.value[0] *= src.value[0];
    dst.value[1] += src.value[1];
  }

  KOKKOS_INLINE_FUNCTION
  void final(value_type &dst) const {
    dst.value[0] /= -2;
    dst.value[1] /= -2;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &update) const {
    update.value[0] *= (i + 1);
    update.value[1] *= (i + 2);
  }
};

struct TestTeamReducer {
  using reducer    = TestTeamReducer;
  using value_type = SimpleTestValueType;

  KOKKOS_INLINE_FUNCTION
  TestTeamReducer(value_type &val) : local(val) {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type &init) const {
    init.value[0] = 1;
    init.value[1] = 10;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, value_type const &src) const {
    dst.value[0] *= src.value[0];
    dst.value[1] += src.value[1];
  }

  KOKKOS_INLINE_FUNCTION
  void final(value_type &dst) const {
    dst.value[0] /= -2;
    dst.value[1] /= -2;
  }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const { return local; }

  value_type &local;
};

namespace {

template <typename ExecSpace>
class TestTeamNestedReducerFunctor {
 public:
  using execution_space  = ExecSpace;
  using team_policy_type = Kokkos::TeamPolicy<execution_space>;
  using member_type      = typename team_policy_type::member_type;
  using value_type       = SimpleTestValueType;
  using functor_type     = TestTeamReducerFunctor;
  using reducer_type     = TestTeamReducer;
  using index_type       = int;

  void run_test_team_thread() {
    auto policy = KOKKOS_LAMBDA(member_type const &member, index_type count) {
      return Kokkos::TeamThreadRange(member, count);
    };
    run_test_team_policies(policy);
  }

  void run_test_thread_vector() {
    auto policy = KOKKOS_LAMBDA(member_type const &member, index_type count) {
      return Kokkos::ThreadVectorRange(member, count);
    };
    run_test_team_policies(policy);
  }

  void run_test_team_vector() {
    auto policy = KOKKOS_LAMBDA(member_type const &member, index_type count) {
      return Kokkos::TeamVectorRange(member, count);
    };
    run_test_team_policies(policy);
  }

  template <typename Policy>
  void run_test_team_policies(Policy &policy) {
    constexpr index_type league_size = 3;
    constexpr index_type test_count  = 8;

    Kokkos::View<value_type[league_size], execution_space>
        reducer_functor_result("reducer_functor_result");
    Kokkos::View<value_type[league_size], execution_space> reducer_result(
        "reducer_result");

    Kokkos::parallel_for(
        team_policy_type(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(member_type const &team) {
          const int league = team.league_rank();

          // Using a functor as reducer
          value_type result1{};
          Kokkos::parallel_reduce(policy(team, test_count), functor_type{},
                                  result1);

          // Using a reducer
          value_type result2{};
          reducer_type reducer(result2);
          Kokkos::parallel_reduce(
              policy(team, test_count),
              [&](const int i, value_type &update) {
                update.value[0] *= (i + 1);
                update.value[1] *= (i + 2);
              },
              reducer);

          Kokkos::single(Kokkos::PerTeam(team), [=]() {
            reducer_functor_result(league).value[0] = result1.value[0];
            reducer_functor_result(league).value[1] = result1.value[1];

            reducer_result(league).value[0] = result2.value[0];
            reducer_result(league).value[1] = result2.value[1];
          });
        });
    Kokkos::fence();

    auto test1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace{}, reducer_functor_result);
    auto test2 = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace{}, reducer_result);

    for (unsigned i = 0; i < test1.extent(0); ++i) {
      EXPECT_EQ(test1(i).value[0], test2(i).value[0]);
      EXPECT_EQ(test1(i).value[1], test2(i).value[1]);
    }
  }
};

}  // namespace

}  // namespace Test

/*--------------------------------------------------------------------------*/
