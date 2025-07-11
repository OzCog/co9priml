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

#ifndef KOKKOSSPARSE_STATICCRSGRAPH_HPP_
#define KOKKOSSPARSE_STATICCRSGRAPH_HPP_

#include <Kokkos_Core.hpp>
#include "KokkosKernels_default_types.hpp"

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4

#define KOKKOS_IMPL_DO_NOT_WARN_INCLUDE_STATIC_CRS_GRAPH
#include <Kokkos_StaticCrsGraph.hpp>
#undef KOKKOS_IMPL_DO_NOT_WARN_INCLUDE_STATIC_CRS_GRAPH

namespace KokkosSparse {
using Kokkos::create_mirror;
using Kokkos::create_mirror_view;
using Kokkos::create_staticcrsgraph;
using Kokkos::GraphRowViewConst;
using Kokkos::maximum_entry;
using Kokkos::StaticCrsGraph;
}  // namespace KokkosSparse

#else

namespace KokkosSparse {

namespace Impl {
template <class RowOffsetsType, class RowBlockOffsetsType>
struct StaticCrsGraphBalancerFunctor {
  using int_type = typename RowOffsetsType::non_const_value_type;
  RowOffsetsType row_offsets;
  RowBlockOffsetsType row_block_offsets;

  int_type cost_per_row, num_blocks;

  StaticCrsGraphBalancerFunctor(RowOffsetsType row_offsets_, RowBlockOffsetsType row_block_offsets_,
                                int_type cost_per_row_, int_type num_blocks_)
      : row_offsets(row_offsets_),
        row_block_offsets(row_block_offsets_),
        cost_per_row(cost_per_row_),
        num_blocks(num_blocks_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int_type& iRow) const {
    const int_type num_rows    = row_offsets.extent(0) - 1;
    const int_type num_entries = row_offsets(num_rows);
    const int_type total_cost  = num_entries + num_rows * cost_per_row;

    const double cost_per_workset = 1.0 * total_cost / num_blocks;

    const int_type row_cost = row_offsets(iRow + 1) - row_offsets(iRow) + cost_per_row;

    int_type count = row_offsets(iRow + 1) + cost_per_row * iRow;

    if (iRow == num_rows - 1) row_block_offsets(num_blocks) = num_rows;

    if (true) {
      int_type current_block = (count - row_cost - cost_per_row) / cost_per_workset;
      int_type end_block     = count / cost_per_workset;

      // Handle some corner cases for the last two blocks.
      if (current_block >= num_blocks - 2) {
        if ((current_block == num_blocks - 2) && (count >= (current_block + 1) * cost_per_workset)) {
          int_type row   = iRow;
          int_type cc    = count - row_cost - cost_per_row;
          int_type block = cc / cost_per_workset;
          while ((block > 0) && (block == current_block)) {
            cc    = row_offsets(row) + row * cost_per_row;
            block = cc / cost_per_workset;
            row--;
          }
          if ((count - cc - row_cost - cost_per_row) < num_entries - row_offsets(iRow + 1)) {
            row_block_offsets(current_block + 1) = iRow + 1;
          } else {
            row_block_offsets(current_block + 1) = iRow;
          }
        }
      } else {
        if ((count >= (current_block + 1) * cost_per_workset) || (iRow + 2 == int_type(row_offsets.extent(0)))) {
          if (end_block > current_block + 1) {
            int_type num_block                   = end_block - current_block;
            row_block_offsets(current_block + 1) = iRow;
            for (int_type block = current_block + 2; block <= end_block; block++)
              if ((block < current_block + 2 + (num_block - 1) / 2))
                row_block_offsets(block) = iRow;
              else
                row_block_offsets(block) = iRow + 1;
          } else {
            row_block_offsets(current_block + 1) = iRow + 1;
          }
        }
      }
    }
  }
};
}  // namespace Impl

/// \class GraphRowViewConst
/// \brief View of a row of a sparse graph.
/// \tparam GraphType Sparse graph type, such as (but not limited to)
/// StaticCrsGraph.
///
/// This class provides a generic view of a row of a sparse graph.
/// We intended this class to view a row of a StaticCrsGraph, but
/// GraphType need not necessarily be CrsMatrix.
///
/// The row view is suited for computational kernels like sparse
/// matrix-vector multiply, as well as for modifying entries in the
/// sparse matrix.  The view is always const as it does not allow graph
/// modification.
///
/// Here is an example loop over the entries in the row:
/// \code
/// using ordinal_type = typename GraphRowViewConst<MatrixType>::ordinal_type;
///
/// GraphRowView<GraphType> G_i = ...;
/// const ordinal_type numEntries = G_i.length;
/// for (ordinal_type k = 0; k < numEntries; ++k) {
///   ordinal_type j = G_i.colidx (k);
///   // ... do something with A_ij and j ...
/// }
/// \endcode
///
/// GraphType must provide the \c data_type
/// aliases. In addition, it must make sense to use GraphRowViewConst to
/// view a row of GraphType. In particular, column
/// indices of a row must be accessible using the <tt>entries</tt>
/// resp. <tt>colidx</tt> arrays given to the constructor of this
/// class, with a constant <tt>stride</tt> between successive entries.
/// The stride is one for the compressed sparse row storage format (as
/// is used by CrsMatrix), but may be greater than one for other
/// sparse matrix storage formats (e.g., ELLPACK or jagged diagonal).
template <class GraphType>
struct GraphRowViewConst {
  //! The type of the column indices in the row.
  using ordinal_type = const typename GraphType::data_type;

 private:
  //! Array of (local) column indices in the row.
  ordinal_type* colidx_;
  /// \brief Stride between successive entries in the row.
  ///
  /// For compressed sparse row (CSR) storage, this is always one.
  /// This might be greater than one for storage formats like ELLPACK
  /// or Jagged Diagonal.  Nevertheless, the stride can never be
  /// greater than the number of rows or columns in the matrix.  Thus,
  /// \c ordinal_type is the correct type.
  const ordinal_type stride_;

 public:
  /// \brief Constructor
  ///
  /// \param colidx_in [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each of the above arrays.
  /// \param count [in] Number of entries in the row.
  KOKKOS_INLINE_FUNCTION
  GraphRowViewConst(ordinal_type* const colidx_in, const ordinal_type& stride, const ordinal_type& count)
      : colidx_(colidx_in), stride_(stride), length(count) {}

  /// \brief Constructor with offset into \c colidx array
  ///
  /// \param colidx_in [in] Array of the row's column indices.
  /// \param stride [in] (Constant) stride between matrix entries in
  ///   each of the above arrays.
  /// \param count [in] Number of entries in the row.
  /// \param idx [in] Start offset into \c colidx array
  ///
  /// \tparam OffsetType The type of \c idx (see above).  Must be a
  ///   built-in integer type.  This may differ from ordinal_type.
  ///   For example, the matrix may have dimensions that fit in int,
  ///   but a number of entries that does not fit in int.
  template <class OffsetType>
  KOKKOS_INLINE_FUNCTION GraphRowViewConst(const typename GraphType::entries_type& colidx_in,
                                           const ordinal_type& stride, const ordinal_type& count, const OffsetType& idx,
                                           const std::enable_if_t<std::is_integral_v<OffsetType>, int>& = 0)
      : colidx_(&colidx_in(idx)), stride_(stride), length(count) {}

  /// \brief Number of entries in the row.
  ///
  /// This is a public const field rather than a public const method,
  /// in order to avoid possible overhead of a method call if the
  /// compiler is unable to inline that method call.
  ///
  /// We assume that rows contain no duplicate entries (i.e., entries
  /// with the same column index).  Thus, a row may have up to
  /// A.numCols() entries.  This means that the correct type of
  /// 'length' is ordinal_type.
  const ordinal_type length;

  /// \brief (Const) reference to the column index of entry i in this
  ///   row of the sparse matrix.
  ///
  /// "Entry i" is not necessarily the entry with column index i, nor
  /// does i necessarily correspond to the (local) row index.
  KOKKOS_INLINE_FUNCTION
  ordinal_type& colidx(const ordinal_type& i) const { return colidx_[i * stride_]; }

  /// \brief An alias for colidx
  KOKKOS_INLINE_FUNCTION
  ordinal_type& operator()(const ordinal_type& i) const { return colidx(i); }
};

/// \class StaticCrsGraph
/// \brief Compressed row storage array.
///
/// \tparam DataType The type of stored entries.  If a StaticCrsGraph is
///   used as the graph of a sparse matrix, then this is usually an
///   integer type, the type of the column indices in the sparse
///   matrix.
///
/// \tparam Arg1Type The second template parameter, corresponding
///   either to the Device type (if there are no more template
///   parameters) or to the Layout type (if there is at least one more
///   template parameter).
///
/// \tparam Arg2Type The third template parameter, which if provided
///   corresponds to the Device type.
///
/// \tparam Arg3Type The third template parameter, which if provided
///   corresponds to the MemoryTraits.
///
/// \tparam SizeType The type of row offsets.  Usually the default
///   parameter suffices.  However, setting a nondefault value is
///   necessary in some cases, for example, if you want to have a
///   sparse matrices with dimensions (and therefore column indices)
///   that fit in \c int, but want to store more than <tt>INT_MAX</tt>
///   entries in the sparse matrix.
///
/// A row has a range of entries:
/// <ul>
/// <li> <tt> row_map[i0] <= entry < row_map[i0+1] </tt> </li>
/// <li> <tt> 0 <= i1 < row_map[i0+1] - row_map[i0] </tt> </li>
/// <li> <tt> entries( entry ,            i2 , i3 , ... ); </tt> </li>
/// <li> <tt> entries( row_map[i0] + i1 , i2 , i3 , ... ); </tt> </li>
/// </ul>
template <class DataType, class Arg1Type, class Arg2Type = void, class Arg3Type = void,
          typename SizeType = ::KokkosKernels::default_size_type>
class StaticCrsGraph {
 private:
  using traits = Kokkos::ViewTraits<DataType*, Arg1Type, Arg2Type, Arg3Type>;

 public:
  using data_type       = DataType;
  using array_layout    = typename traits::array_layout;
  using execution_space = typename traits::execution_space;
  using device_type     = typename traits::device_type;
  using memory_traits   = typename traits::memory_traits;
  using size_type       = SizeType;

  using staticcrsgraph_type = StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>;
  using HostMirror =
      StaticCrsGraph<data_type, array_layout, typename traits::host_mirror_space, memory_traits, size_type>;

  using row_map_type   = Kokkos::View<const size_type*, array_layout, device_type, memory_traits>;
  using entries_type   = Kokkos::View<data_type*, array_layout, device_type, memory_traits>;
  using row_block_type = Kokkos::View<const size_type*, array_layout, device_type, memory_traits>;

  entries_type entries;
  row_map_type row_map;
  row_block_type row_block_offsets;

  KOKKOS_DEFAULTED_FUNCTION
  StaticCrsGraph() = default;

  template <class EntriesType, class RowMapType>
  KOKKOS_INLINE_FUNCTION StaticCrsGraph(const EntriesType& entries_, const RowMapType& row_map_)
      : entries(entries_), row_map(row_map_) {}

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION StaticCrsGraph(const StaticCrsGraph<Args...>& other)
      : entries(other.entries), row_map(other.row_map), row_block_offsets(other.row_block_offsets) {}

  /**  \brief  Return number of rows in the graph
   */
  KOKKOS_INLINE_FUNCTION
  size_type numRows() const {
    return (row_map.extent(0) != 0) ? row_map.extent(0) - static_cast<size_type>(1) : static_cast<size_type>(0);
  }

  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated() const {
    return (row_map.is_allocated() && entries.is_allocated());
  }

  /// \brief Return a const view of row i of the graph.
  ///
  /// If row i does not belong to the graph, return an empty view.
  ///
  /// The returned object \c view implements the following interface:
  /// <ul>
  /// <li> \c view.length is the number of entries in the row </li>
  /// <li> \c view.colidx(k) returns a const reference to the
  ///      column index of the k-th entry in the row </li>
  /// </ul>
  /// k is not a column index; it just counts from 0 to
  /// <tt>view.length - 1</tt>.
  ///
  /// Users should not rely on the return type of this method.  They
  /// should instead assign to 'auto'.  That allows compile-time
  /// polymorphism for different kinds of sparse matrix formats (e.g.,
  /// ELLPACK or Jagged Diagonal) that we may wish to support in the
  /// future.
  KOKKOS_INLINE_FUNCTION
  GraphRowViewConst<StaticCrsGraph> rowConst(const data_type i) const {
    const size_type start = row_map(i);
    // count is guaranteed to fit in ordinal_type, as long as no row
    // has duplicate entries.
    const data_type count = static_cast<data_type>(row_map(i + 1) - start);

    if (count == 0) {
      return GraphRowViewConst<StaticCrsGraph>(nullptr, 1, 0);
    } else {
      return GraphRowViewConst<StaticCrsGraph>(entries, 1, count, start);
    }
  }

  /**  \brief  Create a row partitioning into a given number of blocks
   *           balancing non-zeros + a fixed cost per row.
   */
  void create_block_partitioning(size_type num_blocks, size_type fix_cost_per_row = 4) {
    Kokkos::View<size_type*, array_layout, device_type> block_offsets("StatisCrsGraph::load_balance_offsets",
                                                                      num_blocks + 1);

    Impl::StaticCrsGraphBalancerFunctor<row_map_type, Kokkos::View<size_type*, array_layout, device_type> > partitioner(
        row_map, block_offsets, fix_cost_per_row, num_blocks);

    Kokkos::parallel_for("Kokkos::StaticCrsGraph::create_block_partitioning",
                         Kokkos::RangePolicy<execution_space>(0, numRows()), partitioner);
    typename device_type::execution_space().fence(
        "Kokkos::StaticCrsGraph::create_block_partitioning:: fence after "
        "partition");

    row_block_offsets = block_offsets;
  }
};

//----------------------------------------------------------------------------

template <class StaticCrsGraphType, class InputSizeType>
typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(const std::string& label,
                                                                       const std::vector<InputSizeType>& input);

template <class StaticCrsGraphType, class InputSizeType>
typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(
    const std::string& label, const std::vector<std::vector<InputSizeType> >& input);

//----------------------------------------------------------------------------

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>::HostMirror create_mirror_view(
    const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& input);

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>::HostMirror create_mirror(
    const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& input);

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>::HostMirror create_mirror_view(
    const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& view,
    std::enable_if_t<Kokkos::ViewTraits<DataType, Arg1Type, Arg2Type, Arg3Type>::is_hostspace>* = 0) {
  return view;
}

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>::HostMirror create_mirror(
    const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& view) {
  // Force copy:
  // using alloc = Impl::ViewAssignment<Impl::ViewDefault>; // unused
  using staticcrsgraph_type = StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>;

  typename staticcrsgraph_type::HostMirror tmp;
  typename staticcrsgraph_type::row_map_type::HostMirror tmp_row_map = create_mirror(view.row_map);
  typename staticcrsgraph_type::row_block_type::HostMirror tmp_row_block_offsets =
      create_mirror(view.row_block_offsets);

  // Allocation to match:
  tmp.row_map           = tmp_row_map;  // Assignment of 'const' from 'non-const'
  tmp.entries           = create_mirror(view.entries);
  tmp.row_block_offsets = tmp_row_block_offsets;  // Assignment of 'const' from 'non-const'

  // Deep copy:
  deep_copy(tmp_row_map, view.row_map);
  deep_copy(tmp.entries, view.entries);
  deep_copy(tmp_row_block_offsets, view.row_block_offsets);

  return tmp;
}

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
inline typename StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>::HostMirror create_mirror_view(
    const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& view,
    std::enable_if_t<!Kokkos::ViewTraits<DataType, Arg1Type, Arg2Type, Arg3Type>::is_hostspace>* = 0) {
  return create_mirror(view);
}

template <class StaticCrsGraphType, class InputSizeType>
inline typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(const std::string& label,
                                                                              const std::vector<InputSizeType>& input) {
  using output_type  = StaticCrsGraphType;
  using entries_type = typename output_type::entries_type;
  using work_type    = Kokkos::View<typename output_type::size_type[], typename output_type::array_layout,
                                 typename output_type::device_type, typename output_type::memory_traits>;

  output_type output;

  // Create the row map:

  const size_t length = input.size();

  {
    work_type row_work("tmp", length + 1);

    typename work_type::HostMirror row_work_host = create_mirror_view(row_work);

    size_t sum       = 0;
    row_work_host[0] = 0;
    for (size_t i = 0; i < length; ++i) {
      row_work_host[i + 1] = sum += input[i];
    }

    deep_copy(row_work, row_work_host);

    output.entries = entries_type(label, sum);
    output.row_map = row_work;
  }

  return output;
}

//----------------------------------------------------------------------------

template <class StaticCrsGraphType, class InputSizeType>
inline typename StaticCrsGraphType::staticcrsgraph_type create_staticcrsgraph(
    const std::string& label, const std::vector<std::vector<InputSizeType> >& input) {
  using output_type  = StaticCrsGraphType;
  using entries_type = typename output_type::entries_type;

  static_assert(entries_type::rank == 1, "Graph entries view must be rank one");

  using work_type = Kokkos::View<typename output_type::size_type[], typename output_type::array_layout,
                                 typename output_type::device_type, typename output_type::memory_traits>;

  output_type output;

  // Create the row map:

  const size_t length = input.size();

  {
    work_type row_work("tmp", length + 1);

    typename work_type::HostMirror row_work_host = create_mirror_view(row_work);

    size_t sum       = 0;
    row_work_host[0] = 0;
    for (size_t i = 0; i < length; ++i) {
      row_work_host[i + 1] = sum += input[i].size();
    }

    deep_copy(row_work, row_work_host);

    output.entries = entries_type(label, sum);
    output.row_map = row_work;
  }

  // Fill in the entries:
  {
    typename entries_type::HostMirror host_entries = create_mirror_view(output.entries);

    size_t sum = 0;
    for (size_t i = 0; i < length; ++i) {
      for (size_t j = 0; j < input[i].size(); ++j, ++sum) {
        host_entries(sum) = input[i][j];
      }
    }

    deep_copy(output.entries, host_entries);
  }

  return output;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Impl {

template <class GraphType>
struct StaticCrsGraphMaximumEntry {
  using execution_space = typename GraphType::execution_space;
  using value_type      = typename GraphType::data_type;

  const typename GraphType::entries_type entries;

  StaticCrsGraphMaximumEntry(const GraphType& graph) : entries(graph.entries) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned i, value_type& update) const {
    if (update < entries(i)) update = entries(i);
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& update) const { update = 0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& update, const value_type& input) const {
    if (update < input) update = input;
  }
};

}  // namespace Impl

template <class DataType, class Arg1Type, class Arg2Type, class Arg3Type, typename SizeType>
DataType maximum_entry(const StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>& graph) {
  using GraphType   = StaticCrsGraph<DataType, Arg1Type, Arg2Type, Arg3Type, SizeType>;
  using FunctorType = Impl::StaticCrsGraphMaximumEntry<GraphType>;

  DataType result = 0;
  Kokkos::parallel_reduce("Kokkos::maximum_entry", graph.entries.extent(0), FunctorType(graph), result);
  return result;
}

}  // namespace KokkosSparse

#endif

#endif
