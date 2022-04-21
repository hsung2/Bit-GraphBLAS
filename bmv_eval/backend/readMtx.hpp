// from GraphBLAST library: https://github.com/gunrock/graphblast/blob/master/graphblas/util.hpp

#ifndef READMTX_HPP_
#define READMTX_HPP_
#include "mmio.hpp"

#include <sys/resource.h>
#include <sys/time.h>
#include <libgen.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <string>
#include <iostream>
#include <string.h>

#define GRB_MAXLEN 256

template<typename T>
inline T getEnv(const char *key, T default_val) {
  const char *val = std::getenv(key);
  if (val == NULL)
    return default_val;
  else
    return static_cast<T>(atoi(val));
}

template<typename T>
void setEnv(const char *key, T default_val) {
  std::string s = std::to_string(default_val);
  const char *val = s.c_str();
  setenv(key, val, 0);
}

template<typename T>
bool compare(const std::tuple<int,
                              int,
                              T,
                              int> &lhs,
             const std::tuple<int,
                              int,
                              T,
                              int> &rhs) {
  int a = std::get<0>(lhs);
  int b = std::get<0>(rhs);
  int c = std::get<1>(lhs);
  int d = std::get<1>(rhs);
  if (a == b)
    return c < d;
  else
    return a < b;
}

template<typename T>
void customSort(std::vector<int>* row_indices,
                std::vector<int>* col_indices,
                std::vector<T>*                values) {
  int nvals = row_indices->size();
  std::vector<std::tuple<int,
                         int,
                         T,
                         int> > my_tuple;

  for (int i = 0; i < nvals; ++i)
    my_tuple.push_back(std::make_tuple( (*row_indices)[i], (*col_indices)[i],
        (*values)[i], i));

  std::sort(my_tuple.begin(), my_tuple.end(), compare<T>);

  std::vector<int> v1 = *row_indices;
  std::vector<int> v2 = *col_indices;
  std::vector<T>                v3 = *values;

  for (int i = 0; i < nvals; ++i) {
    int index = std::get<3>(my_tuple[i]);
    (*row_indices)[i] = v1[index];
    (*col_indices)[i] = v2[index];
    (*values)[i]      = v3[index];
  }
}

template<typename T, typename mtxT>
void readTuples(std::vector<int>* row_indices,
                std::vector<int>* col_indices,
                std::vector<T>*                values,
                int               nvals,
                FILE*                          f) {
  int row_ind, col_ind;
  T value;
  mtxT raw_value;
  char type_str[3];
  type_str[0] = '%';
  if (typeid(mtxT) == typeid(int))
    type_str[1] = 'd';
  else if (typeid(mtxT) == typeid(float))
    type_str[1] = 'f';
//  else if (typeid(mtxT) == typeid(double))
//    type_str[1] = 'f';

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for (int i = 0; i < nvals; i++) {
    if (fscanf(f, "%d", &row_ind) == EOF) {
      std::cout << "Error: Not enough rows in mtx file!\n";
      return;
    } else {
      int u = fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices->push_back(row_ind-1);
      col_indices->push_back(col_ind-1);

      u = fscanf(f, type_str, &raw_value);
      value = static_cast<T>(raw_value);

      values->push_back(value);
    }
  }
}

template<typename T>
void readTuples(std::vector<int>* row_indices,
                std::vector<int>* col_indices,
                std::vector<T>*                values,
                int               nvals,
                FILE*                          f) {
  int row_ind, col_ind;
  T value = (T) 1.0;

  // Currently checks if there are fewer rows than promised
  // Could add check for edges in diagonal of adjacency matrix
  for (int i = 0; i < nvals; i++) {
    if (fscanf(f, "%d", &row_ind) == EOF) {
      std::cout << "Error: Not enough rows in mtx file!\n";
      return;
    } else {
      int u = fscanf(f, "%d", &col_ind);

      // Convert 1-based indexing MTX to 0-based indexing C++
      row_indices->push_back(row_ind-1);
      col_indices->push_back(col_ind-1);
      values->push_back(value);
    }
  }
}

/*!
 * Remove self-loops, duplicates and make graph undirected if option is set
 */
template<typename T>
void removeSelfloop(std::vector<int>* row_indices,
                    std::vector<int>* col_indices,
                    std::vector<T>*                values,
                    int*              nvals,
                    bool                           undirected) {
  bool remove_self_loops = getEnv("GRB_UTIL_REMOVE_SELFLOOP", true);

  if (undirected) {
    for (int i = 0; i < *nvals; i++) {
      if ((*col_indices)[i] != (*row_indices)[i]) {
        row_indices->push_back((*col_indices)[i]);
        col_indices->push_back((*row_indices)[i]);
        values->push_back((*values)[i]);
      }
    }
  }

  *nvals = row_indices->size();

  // Sort
  customSort<T>(row_indices, col_indices, values);

  int curr = (*col_indices)[0];
  int last;
  int curr_row = (*row_indices)[0];
  int last_row;

  // Detect self-loops and duplicates
  for (int i = 0; i < *nvals; i++) {
    last = curr;
    last_row = curr_row;
    curr = (*col_indices)[i];
    curr_row = (*row_indices)[i];

    // Self-loops
    if (remove_self_loops && curr_row == curr)
      (*col_indices)[i] = -1;

  // Duplicates
    if (i > 0 && curr == last && curr_row == last_row)
      (*col_indices)[i] = -1;
  }

  int shift = 0;

  // Remove self-loops and duplicates marked -1.
  int back = 0;
  for (int i = 0; i + shift < *nvals; i++) {
    if ((*col_indices)[i] == -1) {
      for (; back <= *nvals; shift++) {
        back = i+shift;
        if ((*col_indices)[back] != -1) {
          (*col_indices)[i] = (*col_indices)[back];
          (*row_indices)[i] = (*row_indices)[back];
          (*col_indices)[back] = -1;
          break;
        }
      }
    }
  }

  *nvals = *nvals - shift;
  row_indices->resize(*nvals);
  col_indices->resize(*nvals);
  values->resize(*nvals);
}

bool exists(const char *fname) {
  FILE *file;
  if (file = fopen(fname, "r")) {
    fclose(file);
    return 1;
  }
  return 0;
}

char* convert(const char* fname, bool is_undirected = true) {
  char* dat_name = reinterpret_cast<char*>(malloc(GRB_MAXLEN));

  // separate the graph path and the file name
  char *temp1 = strdup(fname);
  char *temp2 = strdup(fname);
  char *file_path = dirname(temp1);
  char *file_name = basename(temp2);
  bool remove_self_loops = getEnv("GRB_UTIL_REMOVE_SELFLOOP", true);
  //std::cout << "Remove self-loop: " << remove_self_loops << std::endl;

  snprintf(dat_name, GRB_MAXLEN, "%s/.%s.%s.%s.%sbin", file_path, file_name,
      (is_undirected ? "ud" : "d"),
      (remove_self_loops ? "nosl" : "sl"),
      ((sizeof(int) == 8) ? "64bVe." : ""));

  return dat_name;
}

template <typename T>
void coo2csr(int*                    csrRowPtr,
             int*                    csrColInd,
             T*                        csrVal,
             const std::vector<int>& row_indices,
             const std::vector<int>& col_indices,
             const std::vector<T>&     values,
             int                     nrows,
             int                     ncols) {
  int temp, row, col, dest, cumsum = 0;
  int nvals = row_indices.size();

  std::vector<int> row_indices_t = row_indices;
  std::vector<int> col_indices_t = col_indices;
  std::vector<T>     values_t = values;

  customSort<T>(&row_indices_t, &col_indices_t, &values_t);

  // Set all rowPtr to 0
  for (int i = 0; i <= nrows; i++)
    csrRowPtr[i] = 0;

  // Go through all elements to see how many fall in each row
  for (int i = 0; i < nvals; i++) {
    row = row_indices_t[i];
    if (row >= nrows) std::cout << "Error: Index out of bounds!\n";
    csrRowPtr[row]++;
  }

  // Cumulative sum to obtain rowPtr
  for (int i = 0; i < nrows; i++) {
    temp = csrRowPtr[i];
    csrRowPtr[i] = cumsum;
    cumsum += temp;
  }
  csrRowPtr[nrows] = nvals;

  // Store colInd and val
  for (int i = 0; i < nvals; i++) {
    row = row_indices_t[i];
    dest = csrRowPtr[row];
    col = col_indices_t[i];
    if (col >= ncols) std::cout << "Error: Index out of bounds!\n";
    csrColInd[dest] = col;
    csrVal[dest] = values_t[i];
    csrRowPtr[row]++;
  }
  cumsum = 0;

  // Undo damage done to rowPtr
  for (int i = 0; i < nrows; i++) {
    temp = csrRowPtr[i];
    csrRowPtr[i] = cumsum;
    cumsum = temp;
  }
  temp = csrRowPtr[nrows];
  csrRowPtr[nrows] = cumsum;
  cumsum = temp;
}

template <typename T>
void coo2csc(int*                    cscColPtr,
             int*                    cscRowInd,
             T*                        cscVal,
             const std::vector<int>& row_indices,
             const std::vector<int>& col_indices,
             const std::vector<T>&     values,
             int                     nrows,
             int                     ncols) {
  return coo2csr(cscColPtr, cscRowInd, cscVal, col_indices, row_indices, values,
      ncols, nrows);
}

template <typename T>
void csr2csc(int*       cscColPtr,
             int*       cscRowInd,
             T*           cscVal,
             const int* csrRowPtr,
             const int* csrColInd,
             const T*     csrVal,
             int        nrows,
             int        ncols) {
  int nvals = csrRowPtr[nrows];
  std::vector<int> row_indices(nvals, 0);
  std::vector<int> col_indices(nvals, 0);
  std::vector<T>     values(nvals, 0);

  for (int i = 0; i < nrows; ++i) {
    int row_start = csrRowPtr[i];
    int row_end   = csrRowPtr[i+1];
    for (; row_start < row_end; ++row_start) {
      row_indices[row_start] = i;
      col_indices[row_start] = csrColInd[row_start];
      values[row_start] = csrVal[row_start];
    }
  }

  return coo2csc(cscColPtr, cscRowInd, cscVal, row_indices, col_indices, values,
      ncols, nrows);
}

// Directed controls how matrix is interpreted:
// 0: If it is marked symmetric, then double the edges. Else do nothing.
// 1: Force matrix to be unsymmetric.
// 2: Force matrix to be symmetric.
template<typename T>
int readMtx(const char*                    fname,
            std::vector<int>*              row_indices,
            std::vector<int>*              col_indices,
            std::vector<T>*                values,
            int*                           nrows,
            int*                           ncols,
            int*                           nvals,
            int                            directed,
            bool                           mtxinfo,
            char**                         dat_name = NULL)
{
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  if ((f = fopen(fname, "r")) == NULL) {
    printf("File %s not found\n", fname);
    exit(1);
  }

  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Read MTX Size
  if ((ret_code = mm_read_mtx_crd_size(f, nrows, ncols, nvals)) != 0)
    exit(1);

  //printf("Undirected due to mtx: %d\n", mm_is_symmetric(matcode));
  //printf("Undirected due to cmd: %d\n", directed == 2);
  bool is_undirected = mm_is_symmetric(matcode) || directed == 2;
  is_undirected = (directed == 1) ? false : is_undirected;
  //printf("Undirected: %d\n", is_undirected);
  if (dat_name != NULL)
    *dat_name = convert(fname, is_undirected);

  if (dat_name != NULL && exists(*dat_name)) {
    // The size of the file in bytes is in results.st_size
    // -unserialize vector
    std::ifstream ifs(*dat_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
      std::cout << "Error: Unable to open file for reading!\n";
    } else {
    // Empty arrays indicate to Matrix::build that binary file exists
//      row_indices->clear();
//      col_indices->clear();
//      values->clear();
    }
//  } else {
//
  }

    if (mm_is_integer(matcode))
        readTuples<T, int>(row_indices, col_indices, values, *nvals, f);
    else if (mm_is_real(matcode))
        readTuples<T, float>(row_indices, col_indices, values, *nvals, f);
    else if (mm_is_pattern(matcode))
        readTuples<T>(row_indices, col_indices, values, *nvals, f);

    removeSelfloop<T>(row_indices, col_indices, values, nvals, is_undirected);
    customSort<T>(row_indices, col_indices, values);

    if (mtxinfo) mm_write_banner(stdout, matcode);
    if (mtxinfo) mm_write_mtx_crd_size(stdout, *nrows, *ncols, *nvals);

  return ret_code;
}

#endif